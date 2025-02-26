import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import wandb
from src.environment import GameEnvironment
from src.logging_config import logger
from src.utils import (
    calculate_flatten_size,
    capture_screen_to_video,
    preprocess_frames,
    sample_expert_states,
    save_model,
)


class Policy(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(Policy, self).__init__()
        self.state_seq_len = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(self.state_seq_len, 8, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.flatten_size = calculate_flatten_size(input_shape, self.cnn)
        logger.info(f"Policy flatten_size: {self.flatten_size}")

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        cnn_out = self.cnn(x)
        logits = self.fc(cnn_out)
        return logits

    def get_action(self, state: torch.Tensor) -> Tuple[int, float, np.ndarray]:
        logits = self.forward(state)
        action_dist = Categorical(logits=logits)
        probs = action_dist.probs.detach().cpu().numpy()

        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action), probs


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        state_seq_len = input_shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(state_seq_len, 8, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        self.flatten_size = calculate_flatten_size(input_shape, self.cnn)
        logger.info(f"Discriminator flatten_size: {self.flatten_size}")
        self.fc = nn.Sequential(nn.Linear(self.flatten_size, 1), nn.Sigmoid())

    def forward(self, x):
        cnn_out = self.cnn(x)
        return self.fc(cnn_out)


class GAIfO:
    def __init__(self, action_dim, device, args):

        self.use_wandb = args.use_wandb
        self.device = device

        self.gamma = args.gamma
        self.epsilon = args.ppo_epsilon
        self.policy_epochs = args.policy_epochs

        self.policy = Policy(
            input_shape=(args.state_seq_length, args.img_height, args.img_width),
            action_dim=action_dim,
        ).to(self.device)
        self.discriminator = Discriminator(
            input_shape=(args.state_seq_length, args.img_height, args.img_width),
        ).to(self.device)

        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=args.lr_policy,
            weight_decay=args.policy_weight_decay,
        )
        self.discr_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=args.lr_disc,
            weight_decay=args.disc_weight_decay,
        )

    def compute_returns(self, rewards) -> np.ndarray:
        returns = []
        running_return = 0
        for r in reversed(rewards):
            running_return = r + self.gamma * running_return
            returns.insert(0, running_return)
        returns = np.array(returns)
        return (returns - returns.mean()) / (returns.std() + 1e-8)

    def update_discriminator(
        self, agent_transitions: torch.Tensor, expert_transitions: torch.Tensor
    ):
        agent_transitions = agent_transitions.float().to(self.device)
        expert_transitions = expert_transitions.float().to(self.device)
        agent_preds = self.discriminator(agent_transitions)
        expert_preds = self.discriminator(expert_transitions)

        discr_loss = -(
            torch.log(agent_preds + 1e-8).mean()
            + torch.log(1 - expert_preds + 1e-8).mean()
        )

        self.discr_optimizer.zero_grad()
        discr_loss.backward()
        self.discr_optimizer.step()

        return discr_loss.item()

    def update_policy(
        self,
        states: List[torch.Tensor],
        actions: List[int],
        old_log_probs: List[float],
        returns: np.ndarray,
    ):
        states = torch.stack(states).to(self.device)
        actions = torch.IntTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        policy_loss = 0
        for _ in range(self.policy_epochs):
            action_logits = self.policy(states)
            dist = Categorical(logits=action_logits)
            curr_log_probs = dist.log_prob(actions)
            ratios = torch.exp(curr_log_probs - old_log_probs)
            surr1 = ratios * returns
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * returns
            policy_loss = -torch.min(surr1, surr2).mean()
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        return policy_loss.item()


def train_gaifo(environment: GameEnvironment, agent: GAIfO, args, device):
    for episode in range(args.n_episodes):
        logger.debug(f"EPISODE {episode}")
        start_frame = environment.reset()

        # Initialize the frame stack with the starting frame
        frame_stack = [start_frame for _ in range(args.state_seq_length)]
        state = preprocess_frames(frame_stack, args.img_height, args.img_width).to(
            device
        )

        states = [state]
        actions = []
        rewards = []
        old_log_probs = []
        action_distibutions = []
        step_times = []

        for i in range(args.episode_len):
            step_start_time = time.time()

            with torch.no_grad():
                action, action_log_prob, action_dist = agent.policy.get_action(state)
            logger.debug(f"STEP {i}, action {action}")
            next_frame = environment.step(action)
            frame_stack.append(next_frame)

            next_state = preprocess_frames(
                frame_stack[-args.state_seq_length :], args.img_height, args.img_width
            ).to(device)

            with torch.no_grad():
                disc_output = agent.discriminator(
                    next_state.unsqueeze(0)
                )  # unsqueeze to add batch dimension [1, state_seq_len, height, width]
                reward = -torch.log(disc_output + 1e-8).item()

            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            old_log_probs.append(action_log_prob)
            action_distibutions.append(action_dist)

            state = next_state

            step_time = time.time() - step_start_time
            step_times.append(step_time)

        environment.un_pause(sleep=1)

        if agent.use_wandb:
            wandb.log(
                {
                    "losses/Policy LR": agent.policy_optimizer.param_groups[0]["lr"],
                    "losses/Discr LR": agent.discr_optimizer.param_groups[0]["lr"],
                    "Episode": episode,
                }
            )

        mean_step_interval = int(np.array(step_times).mean() * 1000)
        batch_expert = sample_expert_states(
            batch_size=args.episode_len,
            state_seq_length=args.state_seq_length,
            desired_frames_interval_ms=mean_step_interval,
            img_height=args.img_height,
            img_width=args.img_width,
        )

        disc_loss = agent.update_discriminator(
            agent_transitions=torch.stack(states[1:]), expert_transitions=batch_expert
        )

        #  Compute the cumulative discounted rewards for the episode
        returns = agent.compute_returns(rewards)
        # Update the policy using PPO
        policy_loss = agent.update_policy(states[:-1], actions, old_log_probs, returns)

        logger.info(np.array(action_distibutions).mean(axis=0))
        if agent.use_wandb:
            wandb.log(
                {
                    "episode": episode,
                    "charts/Cumulative Reward": sum(rewards),
                    "losses/Discr Loss": disc_loss,
                    "losses/Policy Loss": policy_loss,
                    "charts/Mean Step Interval": mean_step_interval,
                }
            )

        if episode == args.n_episodes - 1 or episode % args.save_freq == 0:
            save_model(
                agent.discriminator, agent.discr_optimizer, "discriminator", episode
            )
            save_model(agent.policy, agent.policy_optimizer, "policy", episode)

            if args.use_wandb:
                save_name = f"{wandb.run.name}_{episode}"
            else:
                save_name = f"{time.strftime("%Y%m%d_%H%M%S")}"
            test_episode(
                environment,
                agent,
                args,
                device,
                is_upload=agent.use_wandb,
                save_name=save_name,
            )


def test_episode(
    environment: GameEnvironment,
    agent: GAIfO,
    args,
    device,
    is_upload: bool,
    save_name: str,
):
    start_frame = environment.reset()
    frame_stack = [start_frame for _ in range(args.state_seq_len)]
    state = preprocess_frames(frame_stack, args.img_height, args.img_width).to(device)

    for i in range(args.episode_len):
        with torch.no_grad():
            action, _, _ = agent.get_action(state)
        logger.debug(f"STEP {i}, action {action}")
        next_frame = environment.step(action)
        frame_stack.append(next_frame)
        next_state = preprocess_frames(
            frame_stack[-args.state_seq_len :], args.img_height, args.img_width
        ).to(device)
        state = next_state

    capture_screen_to_video(frame_stack, output_filename=save_name, is_upload=is_upload)
