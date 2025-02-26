import os
import random
from typing import List

import numpy as np
import torch
from PIL.Image import Image
from torchvision.transforms import transforms

import wandb


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_frames(frames: List[Image], img_height, img_width) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
        ]
    )
    processed = [transform(frame).squeeze(0) for frame in frames]
    return torch.stack(processed)


def calculate_flatten_size(input_shape, conv_layers):
    conv_layers = conv_layers
    dummy_input = torch.zeros(1, *input_shape)
    with torch.no_grad():
        output = conv_layers(dummy_input)
    return output.numel()


def sample_expert_states(
    batch_size, state_seq_len, desired_frames_interval_ms, img_height, img_width
) -> torch.Tensor:
    expert_observations_dir = os.path.join("expert_observations")
    expert_observations_dir = os.path.abspath(expert_observations_dir)
    all_episodes = os.listdir(expert_observations_dir)
    if not all_episodes:
        raise ValueError("No expert trajectories found in the specified directory.")

    batch = []
    for _ in range(batch_size):
        selected_episode = random.choice(all_episodes)
        episode_path = os.path.join(
            expert_observations_dir, selected_episode, "screenshots"
        )

        frame_files = sorted(os.listdir(episode_path))

        # The interval between the frames in the expert observations,
        # can be different than the agent interval,
        # therefore discriminator can exploit the difference in the intervals
        # To prevent this, sample expert observations with the same interval as agent
        if (
            len(selected_episode.split("_")) == 3
        ):  # If the episode name contains the expert frames interval
            frames_interval_ms = int(
                selected_episode.split("_")[0]
            )  # Extract the frames interval
            step = int(desired_frames_interval_ms / frames_interval_ms)
        else:
            step = 1

        timeframe_len = (state_seq_len - 1) * step + 1

        if len(frame_files) < timeframe_len:
            raise ValueError(
                f"Episode {selected_episode} has fewer frames than "
                f"({state_seq_len}-1)*{step} + 1 = {timeframe_len}."
            )

        start_idx = random.randint(0, len(frame_files) - timeframe_len)

        frames = [
            Image.open(os.path.join(episode_path, frame_files[idx]))
            for idx in range(start_idx, start_idx + timeframe_len, step)
        ]
        preprocessed_sequence = preprocess_frames(frames, img_height, img_width)
        batch.append(preprocessed_sequence)
    return torch.stack(batch)


def save_model(model, optimizer, model_name, i_episode):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(
        script_dir, "../models", f"{model_name}_{wandb.run.name}_{i_episode}.pth"
    )
    checkpoint = {
        "task": "recruit",
        "episode": i_episode,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, model_path)
    print("Model saved & uploaded")


def capture_screen_to_video(
    episode_frames: List[Image], output_filename: str, is_upload: bool
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    first_frame = episode_frames[0]
    gif_path = os.path.join(script_dir, "../gifs", f"{output_filename}.gif")
    first_frame.save(
        gif_path, save_all=True, append_images=episode_frames, duration=300, loop=0
    )
    if is_upload:
        wandb.log({"screen_capture": wandb.Video(gif_path)})
