import os
import torch
from src.model import model_config, GAIfO

script_dir = os.path.dirname(os.path.abspath(__file__))

model = GAIfO(model_config, action_dim=8)
model.policy.load_state_dict(torch.load(os.path.join(script_dir, "../models", "policy.pth")))
model.discriminator.load_state_dict(torch.load(os.path.join(script_dir, "../models", "discriminator.pth")))

