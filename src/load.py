import os
import torch
import wandb

from src.environment import GameEnvironment
from src.model import model_config, GAIfO, test_episode

script_dir = os.path.dirname(os.path.abspath(__file__))

env = GameEnvironment()
model = GAIfO(model_config, action_dim=8)
model.policy.load_state_dict(torch.load(os.path.join(script_dir, "../models", "policy.pth")))
model.discriminator.load_state_dict(torch.load(os.path.join(script_dir, "../models", "discriminator.pth")))

test_episode(env, model, is_upload=True, save_name=f"{wandb.run.name}_99")