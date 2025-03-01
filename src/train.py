from src.environment import GameEnvironment
from src.model import GAIfO, train_gaifo
from src.utils import get_device, set_random_seed


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train GAIfO agent")
    # General arguments
    parser.add_argument(
        "--n_episodes", type=int, default=50, help="Number of episodes to train"
    )
    parser.add_argument(
        "--episode_len", type=int, default=64, help="Length of each episode"
    )
    parser.add_argument(
        "--state_seq_len",
        type=int,
        default=3,
        help="Number of frames to be stacked to form state",
    )
    parser.add_argument(
        "--seed", type=int, default=31, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--img_height", type=int, default=44, help="Height of input images"
    )
    parser.add_argument(
        "--img_width", type=int, default=120, help="Width of input images"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor for rewards"
    )
    # Discriminator arguments
    parser.add_argument(
        "--discr_lr",
        type=float,
        default=1e-4,
        help="Learning rate for the discriminator network",
    )
    parser.add_argument(
        "--discr_anneal_gamma",
        type=float,
        default=0.95,
        help="Annealing factor for the discriminator gamma",
    )
    # Policy arguments
    parser.add_argument(
        "--policy_lr",
        type=float,
        default=3e-4,
        help="Learning rate for the policy network",
    )
    parser.add_argument(
        "--policy_anneal_gamma",
        type=float,
        default=0.99,
        help="Annealing factor for the policy gamma",
    )
    parser.add_argument(
        "--policy_epochs",
        type=int,
        default=4,
        help="Number of epochs for policy update",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="the lambda for the general advantage estimation",
    )
    parser.add_argument(
        "--clip_vloss",
        type=bool,
        default=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.",
    )
    parser.add_argument(
        "--norm_adv",
        type=bool,
        default=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument(
        "--clip_coef",
        type=float,
        default=0.2,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=0.01,
        help="coefficient of the entropy",
    )
    parser.add_argument(
        "--vf_coef",
        type=float,
        default=0.5,
        help="coefficient of the value function",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--target_kl",
        type=float,
        default=None,
        help="Discount factor for rewards",
    )

    # Logging arguments
    parser.add_argument(
        "--use_wandb",
        type=bool,
        default=True,
        help="Flag to indicate if wandb should be used",
    )
    parser.add_argument(
        "--run_id", type=str, default="", help="Run ID for resuming a previous run"
    )
    parser.add_argument(
        "--save_freq", type=int, default=20, help="Frequency of saving models"
    )

    args = parser.parse_args()

    if args.use_wandb:
        import wandb

        if args.run_id:
            wandb.init(
                project="kingdom-rl",
                config=vars(args),
                id=args.run_id,
                resume="must",
            )
        else:
            wandb.init(
                project="kingdom-rl",
                config=vars(args),
            )

    set_random_seed(args.seed)
    device = get_device()

    env = GameEnvironment()
    model = GAIfO(action_dim=len(env.id2action), device=device, args=args)
    train_gaifo(env, model, args, device)
    if args.use_wandb:
        wandb.finish()
    env.close_game()


if __name__ == "__main__":
    main()
