import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sympy.stats.sampling.sample_numpy import numpy
from torch.distributions import Categorical

import wandb
from src.control import do_nothing
from src.environment import GameEnvironment
from src.logging_config import logger
from src.utils import (
    calculate_flatten_size,
    capture_screen_to_video,
    preprocess_frames,
    sample_expert_states,
    save_model,
)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Policy(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(Policy, self).__init__()
        self.state_seq_len = input_shape[0]
        self.action_dim = action_dim
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(self.state_seq_len, 8, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(8, 16, kernel_size=4, stride=2)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.flatten_size = calculate_flatten_size(input_shape, self.cnn)
        logger.info(f"Policy flatten_size: {self.flatten_size}")

        self.fc = nn.Sequential(
            layer_init(nn.Linear(self.flatten_size, 128)),
            nn.ReLU(),
        )

        self.policy_head = layer_init(
            nn.Linear(128 + self.state_seq_len, action_dim), std=0.01
        )  # actor
        self.value_head = layer_init(
            nn.Linear(128 + self.state_seq_len, 1), std=1
        )  # critic

    def get_action_and_value(
        self, state: torch.Tensor, action_history: torch.Tensor, action=None
    ):
        hidden = self.fc(self.cnn(state))
        hidden = torch.cat((hidden, action_history), dim=-1)
        logits = self.policy_head(hidden)
        action_dist = Categorical(logits=logits)
        if action is None:
            action = action_dist.sample()
        return (
            action,
            action_dist.log_prob(action),
            action_dist.entropy(),
            self.value_head(hidden),
        )

    def get_value(self, state: torch.Tensor, action_history: torch.Tensor):
        hidden = self.fc(self.cnn(state))
        hidden = torch.cat((hidden, action_history), dim=-1)
        return self.value_head(hidden).item()


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
        self.state_seq_len = args.state_seq_len
        self.episode_len = args.episode_len

        # policy hyperparameters
        self.policy_epochs = args.policy_epochs
        self.policy_batch_size = args.episode_len
        self.policy_minibatch_size = args.policy_minibatch_size
        self.gae_lambda = args.gae_lambda
        self.clip_vloss = args.clip_vloss
        self.norm_adv = args.norm_adv
        self.clip_coef = args.clip_coef
        self.ent_coef = args.ent_coef
        self.vf_coef = args.vf_coef
        self.max_grad_norm = args.max_grad_norm
        self.target_kl = args.target_kl

        self.policy = Policy(
            input_shape=(args.state_seq_len, args.img_height, args.img_width),
            action_dim=action_dim,
        ).to(self.device)
        self.discriminator = Discriminator(
            input_shape=(args.state_seq_len, args.img_height, args.img_width),
        ).to(self.device)

        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=args.policy_lr,
        )
        self.policy_scheduler = optim.lr_scheduler.ExponentialLR(
            self.policy_optimizer, gamma=args.policy_anneal_gamma
        )

        self.discr_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=args.discr_lr,
            eps=1e-5,
        )
        self.discr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.discr_optimizer, gamma=args.discr_anneal_gamma
        )

    def update_discriminator(
        self,
        agent_transitions: torch.Tensor,
        expert_transitions: torch.Tensor,
        i_episode: int,
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

        self.discr_scheduler.step()

        if self.use_wandb:
            wandb.log(
                {
                    "episode": i_episode,
                    "discriminator/loss": discr_loss,
                }
            )

    def update_policy(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_state: torch.Tensor,
        i_episode: int,
    ):
        # bootstrap value if not done
        with torch.no_grad():
            episode_len = len(rewards)
            action_history = actions[
                self.episode_len : self.episode_len + self.state_seq_len
            ].unsqueeze(0)
            next_value = self.policy.get_value(
                next_state.unsqueeze(0),
                action_history=action_history,
            )
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(episode_len)):
                if t == episode_len - 1:
                    nextvalues = next_value
                else:
                    nextvalues = values[t + 1]
                delta = rewards[t] + self.gamma * nextvalues - values[t]
                advantages[t] = lastgaelam = (
                    delta + self.gamma * self.gae_lambda * lastgaelam
                )
            returns = advantages + values

        # Optimizing the policy and value network
        b_inds = np.arange(self.policy_batch_size)
        clipfracs = []
        for epoch in range(self.policy_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.policy_batch_size, self.policy_minibatch_size):
                end = start + self.policy_minibatch_size
                mb_inds = b_inds[start:end]
                action_slice_idx = numpy.stack(
                    (mb_inds, mb_inds + self.state_seq_len), axis=1
                )
                actions_histories = torch.stack(
                    [actions[slice(idx[0], idx[1])] for idx in action_slice_idx]
                )

                _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(
                    states[mb_inds],
                    action_history=actions_histories,
                    action=actions.int()[mb_inds + self.state_seq_len],
                )

                logratio = newlogprob - log_probs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    ]

                mb_advantages = advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(
                    -1
                )  # from shape [policy_minibatch_size, 1] to [policy_minibatch_size]
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                    v_clipped = values[mb_inds] + torch.clamp(
                        newvalue - values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        self.policy_scheduler.step()

        y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if self.use_wandb:
            wandb.log(
                {
                    "policy/actor_loss": pg_loss.item(),
                    "policy/critic_loss": v_loss.item(),
                    "policy/entropy_loss": entropy_loss.item(),
                    "policy/old_approx_kl": old_approx_kl.item(),
                    "policy/approx_kl": approx_kl.item(),
                    "policy/explained_var": explained_var,
                    "policy/clipfrac": np.mean(clipfracs),
                    "episode": i_episode,
                }
            )


def train_gaifo(environment: GameEnvironment, agent: GAIfO, args, device):
    for episode in range(args.n_episodes):
        logger.info(f"EPISODE {episode}")

        # 1. Run episode to collect policy trajectory
        states = torch.zeros(
            (args.episode_len + 1, args.state_seq_len, args.img_height, args.img_width)
        ).to(device)
        actions = torch.full(
            (args.episode_len + args.state_seq_len,),
            fill_value=environment.action2id[do_nothing],
        ).to(
            device
        )  # Initialize previous actions with do_nothing
        values = torch.zeros((args.episode_len,)).to(device)
        rewards = torch.zeros((args.episode_len,)).to(device)
        log_probs = torch.zeros((args.episode_len,)).to(device)

        # Initialize the frame stack with the starting frame
        start_frame = environment.reset()
        frames = [start_frame for _ in range(args.state_seq_len)]
        state = preprocess_frames(frames, args.img_height, args.img_width).to(device)
        states[0] = state
        step_times = []

        for step in range(args.episode_len):
            step_start_time = time.time()

            with torch.no_grad():
                action, action_log_prob, _, value = agent.policy.get_action_and_value(
                    state.unsqueeze(0),
                    action_history=actions[step : step + args.state_seq_len].unsqueeze(
                        0
                    ),  # take state_seq_len prev actions as history
                )
            logger.info(f"STEP {step}, action {action.item()}")
            next_frame = environment.step(action.item())
            frames.append(next_frame)

            next_state = preprocess_frames(
                frames[-args.state_seq_len :], args.img_height, args.img_width
            ).to(
                device
            )  # take last state_seq_len frames and stack them

            with torch.no_grad():
                disc_output = agent.discriminator(
                    next_state.unsqueeze(0)
                )  # unsqueeze to add batch dimension [1, state_seq_len, height, width]
                reward = -torch.log(disc_output + 1e-8).item()

            states[step + 1] = next_state
            actions[step + args.state_seq_len] = action
            values[step] = value
            rewards[step] = reward
            log_probs[step] = action_log_prob

            state = next_state

            step_time = time.time() - step_start_time
            step_times.append(step_time)

        environment.un_pause(sleep=1)
        mean_step_interval = int(np.array(step_times).mean() * 1000)

        if agent.use_wandb:
            wandb.log(
                {
                    "policy/lr": agent.policy_optimizer.param_groups[0]["lr"],
                    "discriminator/lr": agent.discr_optimizer.param_groups[0]["lr"],
                    "episode": episode,
                }
            )

        # 2. Sample expert states and update discriminator
        batch_expert = sample_expert_states(
            batch_size=args.episode_len,
            state_seq_len=args.state_seq_len,
            desired_frames_interval_ms=mean_step_interval,
            img_height=args.img_height,
            img_width=args.img_width,
        )
        agent.update_discriminator(
            agent_transitions=states[1:],
            expert_transitions=batch_expert,
            i_episode=episode,
        )

        # 3. Update the policy using PPO
        agent.update_policy(
            rewards=rewards,
            values=values,
            log_probs=log_probs,
            states=states,
            actions=actions,
            next_state=next_state,
            i_episode=episode,
        )

        if args.use_wandb:
            wandb.log(
                {
                    "episode": episode,
                    "cumulative reward": sum(rewards),
                    "mean step reward": rewards.mean(),
                    "mean step interval": mean_step_interval,
                }
            )
            save_name = f"{wandb.run.name}_{episode}"
        else:
            save_name = f"{time.strftime("%Y%m%d_%H%M%S")}"
        if episode == args.n_episodes - 1 or episode % args.save_freq == 0:
            save_model(
                agent.discriminator, agent.discr_optimizer, "discriminator", episode
            )
            save_model(agent.policy, agent.policy_optimizer, "policy", episode)
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
    actions = torch.full(
        (args.episode_len + args.state_seq_len,),
        fill_value=environment.action2id[do_nothing],
    ).to(
        device
    )  # Initialize previous actions with do_nothing

    for step in range(args.episode_len):
        with torch.no_grad():
            action, _, _, _ = agent.policy.get_action_and_value(
                state.unsqueeze(0),
                actions[step : step + args.state_seq_len].unsqueeze(0),
            )
        logger.info(f"STEP {step}, action {action.item()}")
        next_frame = environment.step(action.item())
        frame_stack.append(next_frame)

        next_state = preprocess_frames(
            frame_stack[-args.state_seq_len :], args.img_height, args.img_width
        ).to(device)
        state = next_state
        actions[step + args.state_seq_len] = action

    capture_screen_to_video(frame_stack, output_filename=save_name, is_upload=is_upload)
