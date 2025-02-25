import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
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
        "lr_policy": 3e-4,
        "lr_disc": 3e-4,
        "gamma": 0.99,
        "epsilon": 0.2,
        "seq_length": 3,
        "n_episodes": 100,
        "episode_length": 64,
        "policy_epochs": 4,
        "img_height": 72,
        "img_width": 128,
        "log_freq": 10,
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

class Policy(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(Policy, self).__init__()
        self.seq_length = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(self.seq_length, 8, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.flatten_size = calculate_flatten_size(input_shape, self.cnn.to(device))
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
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=model_config.lr_policy)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=model_config.lr_disc)
        self.gamma = model_config.gamma
        self.epsilon = model_config.epsilon
        wandb.watch(self.policy, criterion=self.policy_optimizer, log="all", log_freq=model_config.log_freq, idx=1)
        wandb.watch(self.discriminator, criterion=self.disc_optimizer, log="all", log_freq=model_config.log_freq, idx=2)

    def get_action(self, state: torch.Tensor) -> Tuple[int, float, np.ndarray]:
        state = state.unsqueeze(0).to(device)
        logits = self.policy(state)
        action_dist = Categorical(logits=logits)
        probs = action_dist.probs.detach().cpu().numpy()

        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action).detach().item(), probs

    def compute_returns(self, rewards) -> np.ndarray:
        returns = []
        running_return = 0
        for r in reversed(rewards):
            running_return = r + self.gamma * running_return
            returns.insert(0, running_return)
        returns = np.array(returns)
        return (returns - returns.mean()) / (returns.std() + 1e-8)

    def update_discriminator(self, agent_transitions: torch.Tensor, expert_transitions: torch.Tensor):
        agent_transitions = agent_transitions.float().to(device)
        expert_transitions = expert_transitions.float().to(device)
        agent_preds = self.discriminator(agent_transitions)
        expert_preds = self.discriminator(expert_transitions)

        disc_loss = -(torch.log(agent_preds + 1e-8).mean() +
                      torch.log(1 - expert_preds + 1e-8).mean())

        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        # Log Discriminator Loss
        return disc_loss.item()

    def update_policy(self, states: List[torch.Tensor], actions: List[int], old_log_probs: List[float], returns: np.ndarray):
        states = torch.stack(states).to(device)
        actions = torch.IntTensor(actions).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        returns = torch.FloatTensor(returns).to(device)

        policy_loss = 0
        for _ in range(model_config.policy_epochs):
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

def preprocess_frames(frames: List[Image]) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((model_config.img_height, model_config.img_width)),
        transforms.ToTensor(),
    ])
    processed = [transform(frame).squeeze(0) for frame in frames]
    return torch.stack(processed).to(device)

def train_gaifo(environment: GameEnvironment, agent: GAIfO):
    for episode in range(model_config.n_episodes):
        logger.debug(f"EPISODE {episode}")
        start_frame = environment.reset()
        frame_stack = [start_frame for _ in range(model_config.seq_length)]
        state = preprocess_frames(frame_stack)
        states, actions, rewards = [state], [], []
        old_log_probs = []
        action_distibutions = []

        for i in range(model_config.episode_length):
            action, action_log_prob, action_dist = agent.get_action(state)
            logger.debug(f"STEP {i}, action {action}")
            next_frame = environment.step(action)
            frame_stack.append(next_frame)
            next_state = preprocess_frames(frame_stack[-model_config.seq_length:])
            with torch.no_grad():
                disc_output = agent.discriminator(next_state.unsqueeze(0))
                reward = -torch.log(disc_output + 1e-8).item()
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            old_log_probs.append(action_log_prob)  # for PPO
            action_distibutions.append(action_dist)

            state = next_state

        environment.un_pause(sleep=1) # pause

        # After the episode is finished, we update the discriminator
        batch_expert = sample_expert_states(batch_size=model_config.episode_length, seq_length=model_config.seq_length)
        disc_loss = agent.update_discriminator(agent_transitions=torch.stack(states[1:]), expert_transitions=batch_expert)

        #  Compute the cumulative discounted rewards for the episode
        returns = agent.compute_returns(rewards)

        # Update the policy using PPO
        policy_loss = agent.update_policy(states[:-1], actions, old_log_probs, returns)

        logger.info(np.array(action_distibutions).mean(axis=0))
        wandb.log(
            {
                "Episode": episode,
                "Cumulative Reward": sum(rewards),
                "Discriminator Loss": disc_loss,
                "Policy Loss": policy_loss,
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

    wandb.finish()
    environment.close_game()

env = GameEnvironment()
model = GAIfO(model_config, action_dim=8)
time.sleep(10)
train_gaifo(env, model)
test_episode(env, model)

# policy = Policy(input_shape=(3, model_config.img_height, model_config.img_width), action_dim=8).to(device)
# print(policy)

# disr = Discriminator(input_shape=(3, model_config.img_height, model_config.img_width)).to(device)
