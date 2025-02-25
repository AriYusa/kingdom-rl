import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from numpy.ma.core import shape
from torch.distributions import Categorical
import numpy as np
import wandb
from torchvision.transforms import transforms
from typing import List, Tuple
from PIL import Image
from src.environment import GameEnvironment
from src.logging_config import logger
from types import SimpleNamespace

# Get the directory of the current script (model.py)
script_dir = os.path.dirname(os.path.abspath(__file__))

model_config = {
        "gamma": 0.99,  # discount coef
        "seq_length": 3,
        "n_episodes": 3,
        "episode_length": 64,
        "img_height": 72,
        "img_width": 128,
        "log_freq": 10,
        # discriminator
        "lr_disc": 1e-4,
        # policy specific
        "gae": True,
        "lr_policy": 3e-4,
        "policy_epochs": 4,
        "minibatch_size": 64, # = episode_length, as episode_length is not big
        "clip_coef": 0.2,
        "norm_adv": True,
        "clip_vloss":True,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": False,
}
wandb.init(
    project="kingdom-rl",
    config=model_config,
)

model_config = SimpleNamespace(**model_config)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_flatten_size(input_shape, conv_layers):
    dummy_input = torch.zeros(1, *input_shape).to(device)
    with torch.no_grad():
        output = conv_layers(dummy_input)
    return output.numel()

def ppo_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Policy(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(Policy, self).__init__()
        self.seq_length = input_shape[0]
        self.cnn = nn.Sequential(
            ppo_layer_init(nn.Conv2d(self.seq_length, 8, kernel_size=8, stride=4)),
            nn.ReLU(),
            ppo_layer_init(nn.Conv2d(8, 16, kernel_size=4, stride=2)),
            nn.ReLU(),
            nn.Flatten(),
            ppo_layer_init(nn.Linear(1568, 128)),
            nn.ReLU(),
        )
        # self.flatten_size = calculate_flatten_size(input_shape, self.cnn.to(device))
        # logger.info(f"Policy flatten_size: {self.flatten_size}")  # 1568

        self.actor = nn.Sequential(
            ppo_layer_init(nn.Linear(128, action_dim)),
        )

        self.critic = nn.Sequential(
            ppo_layer_init(nn.Linear(128, 1)),
        )

    def get_value(self, x):
        encoded = self.cnn(x)
        return self.critic(encoded)

    def get_action_and_value(self, x, action=None):
        encoded = self.cnn(x)
        logits = self.actor(encoded)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(encoded)


class Discriminator(nn.Module):
    """For grayscale images"""
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        seq_length = input_shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(seq_length, 8, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        self.flatten_size = calculate_flatten_size(input_shape, self.cnn.to(device))
        logger.info(f"Discriminator flatten_size {self.flatten_size}")
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        cnn_out = self.cnn(x)
        return self.fc(cnn_out)


class GAIfO:
    def __init__(self, model_config, action_dim):
        self.policy = Policy(
            input_shape=(model_config.seq_length, model_config.img_height, model_config.img_width),
            action_dim=action_dim
        ).to(device)
        self.discriminator = Discriminator(
            input_shape=(model_config.seq_length, model_config.img_height, model_config.img_width)
        ).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=model_config.lr_policy, eps=1e-5)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=model_config.lr_disc)
        wandb.watch(self.policy, criterion=self.policy_optimizer, log="all", log_freq=model_config.log_freq, idx=1)
        wandb.watch(self.discriminator, criterion=self.disc_optimizer, log="all", log_freq=model_config.log_freq, idx=2)

    def compute_returns(self, rewards) -> np.ndarray:
        returns = []
        running_return = 0
        for r in reversed(rewards):
            running_return = r + model_config.gamma * running_return
            returns.insert(0, running_return)
        returns = np.array(returns)
        return (returns - returns.mean()) / (returns.std() + 1e-8)

    def update_discriminator(self, i_episode: int, agent_transitions: torch.Tensor, expert_transitions: torch.Tensor):
        agent_transitions = agent_transitions.float().to(device)
        expert_transitions = expert_transitions.float().to(device)
        agent_preds = self.discriminator(agent_transitions)
        expert_preds = self.discriminator(expert_transitions)

        agent_loss = -torch.log(agent_preds + 1e-8).mean()
        expert_loss = -torch.log(1 - expert_preds + 1e-8).mean()

        disc_loss = agent_loss + expert_loss

        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        # Log Discriminator Loss
        wandb.log({
            "episode": i_episode,
            "discriminator/learning_rate": self.disc_optimizer.param_groups[0]["lr"],
            "discriminator/loss": disc_loss.item(),
            "discriminator/agent_loss": agent_loss.item(),
            "discriminator/expert_loss": expert_loss.item()
        })

    def update_policy(self,
                      i_episode: int,
                      states: List[torch.Tensor],
                      actions: torch.Tensor,
                      logprobs: torch.Tensor,
                      returns: torch.Tensor,
                      values: torch.Tensor,
                      advantages: torch.Tensor
                      ):
        states = torch.stack(states).to(device)
        # 2. Optimizing the policy and value network
        b_inds = np.arange(model_config.episode_length)
        clipfracs = []
        for epoch in range(model_config.policy_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, model_config.episode_length, model_config.minibatch_size):
                end = start + model_config.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(states[mb_inds],
                                                                              actions.long()[mb_inds])
                logratio = newlogprob - logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > model_config.clip_coef).float().mean().item()]

                mb_advantages = advantages[mb_inds]
                if model_config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - model_config.clip_coef, 1 + model_config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if model_config.clip_vloss:
                    v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                    v_clipped = values[mb_inds] + torch.clamp(
                        newvalue - values[mb_inds],
                        -model_config.clip_coef,
                        model_config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - model_config.ent_coef * entropy_loss + model_config.vf_coef * v_loss

                self.policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), model_config.max_grad_norm)
                self.policy_optimizer.step()

            if model_config.target_kl is not None and approx_kl > model_config.target_kl:
                break

            # just for info: how good critic predicts returns
            y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            wandb.log({
                "episode": i_episode,
                "policy/learning_rate": self.policy_optimizer.param_groups[0]["lr"],
                "policy/loss": loss.item(),
                "policy/value_loss": v_loss.item(),
                "policy/policy_loss": pg_loss.item(),
                "policy/entropy_loss": entropy_loss.item(),
                "policy/explained_var": explained_var,
                "policy/old_approx_kl": old_approx_kl.item(),
                "policy/approx_kl": approx_kl.item(),
                "policy/clipfrac": np.mean(clipfracs),
                "policy/explained_variance": explained_var
            })

def preprocess_frames(frames: List[Image]) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((model_config.img_height, model_config.img_width)),
        transforms.ToTensor(),
    ])
    processed = [transform(frame).squeeze(0) for frame in frames]
    return torch.stack(processed).to(device)

def train_gaifo(environment: GameEnvironment, agent: GAIfO):
    for i_episode in range(model_config.n_episodes):
        logger.debug(f"EPISODE {i_episode}")
        start_frame = environment.reset()
        frame_stack = [start_frame for _ in range(model_config.seq_length)]  # TODO try no-ops
        state = preprocess_frames(frame_stack)
        states = [state]

        actions = torch.zeros(model_config.episode_length,).to(device)
        rewards = torch.zeros(model_config.episode_length,).to(device)
        values = torch.zeros(model_config.episode_length,).to(device)
        log_probs = torch.zeros(model_config.episode_length,).to(device)

        # 1. run episode
        for i in range(model_config.episode_length):
            with torch.no_grad():
                action, logprob, _, value = agent.policy.get_action_and_value(state)
            logger.debug(f"STEP {i}, action {action}")
            next_frame = environment.step(action)
            frame_stack.append(next_frame)
            next_state = preprocess_frames(frame_stack[-model_config.seq_length:])
            with torch.no_grad():
                disc_output = agent.discriminator(next_state.unsqueeze(0))
                reward = -torch.log(disc_output + 1e-8).item()
            states.append(next_state)
            actions[i] = action
            rewards[i] = reward
            log_probs[i] = logprob
            values[i] = value

            state = next_state

        environment.un_pause(sleep=1) # pause

        with torch.no_grad():
            next_value = agent.policy.get_value(state)
            # gae
            advantages = torch.zeros(size=(len(rewards),)).to(device)
            lastgaelam = 0
            for t in reversed(range(model_config.episode_length)):
                if t == model_config.episode_length - 1:
                    nextvalues = next_value
                else:
                    nextvalues = values[t + 1]
                delta = rewards[t] + model_config.ppo_gamma * nextvalues - values[t]
                advantages[t] = lastgaelam = delta + model_config.ppo_gamma * model_config.gae_lambda * lastgaelam
            returns = advantages + values

        # 2. Update networks
        # 2.1 update the discriminator
        batch_expert = sample_expert_states(batch_size=model_config.episode_length, seq_length=model_config.seq_length)
        agent.update_discriminator(agent_transitions=torch.stack(states[1:]), expert_transitions=batch_expert)

        # 2.2. Optimizing the policy and value network
        agent.update_policy(i_episode, states[:-1], actions, log_probs, returns, values, advantages)

        wandb.log(
            {
                "Episode": i_episode,
                "Cumulative Reward": sum(rewards),
                # "Avg action distribution": wandb.Histogram(np.array(action_distibutions).mean(axis=0), num_bins=8),  # не так как я хочу
            }
        )
    # save last models
    save_model(agent.discriminator, "discriminator")
    save_model(agent.policy, "policy")

def sample_expert_states(batch_size, seq_length) -> torch.Tensor:
    """
    Load and preprocess expert trajectories for training.

    Args:
        batch_size: Number of sequences to load in a batch.
        seq_length: Length of each sequence of frames (e.g., 3 frames).

    Returns:
        A batch of preprocessed sequences as a tensor of shape (batch_size, seq_length, h, w).
    """
    expert_observations_dir = os.path.join(script_dir, "../expert_observations")
    expert_observations_dir = os.path.abspath(expert_observations_dir)
    all_episodes = os.listdir(expert_observations_dir)  # List all episodes in the directory
    if not all_episodes:
        raise ValueError("No expert trajectories found in the specified directory.")
    batch = []
    for _ in range(batch_size):
        # Randomly select an episode
        selected_episode = random.choice(all_episodes)
        episode_path = os.path.join(expert_observations_dir, selected_episode, 'screenshots')

        frame_files = sorted(os.listdir(episode_path))  # Sorting ensures timestamp order

        if len(frame_files) < seq_length:
            raise ValueError(f"Episode {selected_episode} has fewer frames than seq_length {seq_length}.")
        start_idx = random.randint(0, len(frame_files) - seq_length)
        frames = [Image.open(os.path.join(episode_path, frame_files[idx])) for idx in range(start_idx, start_idx + seq_length)]
        preprocessed_sequence = preprocess_frames(frames)
        batch.append(preprocessed_sequence)
    return torch.stack(batch)


def save_model(model, model_name):
    model_path = os.path.join(script_dir, "../models", f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved & uploaded")

def capture_screen_to_video(episode_frames: List[Image], output_filename: str):
    first_frame = episode_frames[0]
    gif_path = os.path.join(script_dir, "../gifs", f"{output_filename}.gif")
    first_frame.save(gif_path,
                     save_all=True,
                     append_images=episode_frames,
                     duration=300,  # Duration between frames in milliseconds
                     loop=0
    )

    wandb.log({"screen_capture": wandb.Video(gif_path)})

def test_episode(environment: GameEnvironment, agent: GAIfO):
    start_frame = environment.reset()
    frame_stack = [start_frame for _ in range(model_config.seq_length)]
    state = preprocess_frames(frame_stack)

    for i in range(model_config.episode_length):
        action, log_prob, _ = agent.get_action(state)
        logger.debug(f"STEP {i}, action {action}")
        next_frame = environment.step(action)
        frame_stack.append(next_frame)
        next_state = preprocess_frames(frame_stack[-model_config.seq_length:])
        state = next_state

    capture_screen_to_video(frame_stack, output_filename=f"{wandb.run.name}")

env = GameEnvironment()
model = GAIfO(model_config, action_dim=8)
time.sleep(10)
train_gaifo(env, model)
test_episode(env, model)

# policy = Policy(input_shape=(3, model_config.img_height, model_config.img_width), action_dim=8).to(device)
# print(policy)

# disr = Discriminator(input_shape=(3, model_config.img_height, model_config.img_width)).to(device)
