import argparse
from breakout_simulation import BreakoutTrainer

def train(args):
    """Train the agent."""
    trainer = BreakoutTrainer(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        target_update_frequency=args.target_update_frequency,
        continue_training=args.continue_training,
        model_path=args.model_path,
        history_path=args.history_path,
        replay_start_size=args.replay_start_size
    )

    trainer.train(n_episodes=args.episodes, one_epoch=args.epoch_size)
    trainer.env.close()

def play(args):
    """Load a trained model and play games."""
    trainer = BreakoutTrainer(render_type="human")
    trainer.agent.load_saved_model(args.model_path)

    print(f"Playing {args.episodes} episodes...")
    trainer.play_games(num_episodes=args.episodes)
    trainer.env.close()

def main():
    """Args parsing"""
    parser = argparse.ArgumentParser(description='DQN Breakout training and vizualisation')
    subparsers = parser.add_subparsers(dest='command', help='Possible commands:')

    # Training arguments
    train_parser = subparsers.add_parser('train', help='Train the agent')
    train_parser.add_argument('--learning-rate', type=float, default=0.0000625)
    train_parser.add_argument('--gamma', type=float, default=0.99)
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--memory-size', type=int, default=100000)
    train_parser.add_argument('--target-update-frequency', type=int, default=5000)
    train_parser.add_argument('--episodes', type=int, default=500000)
    train_parser.add_argument('--epoch-size', type=int, default=50000)
    train_parser.add_argument('--continue_training', type=bool, default=False)
    train_parser.add_argument('--model-path', type=str, default=None)
    train_parser.add_argument('--history-path', type=str, default=None)
    train_parser.add_argument('--replay-start-size', type=int, default=80000)

    # Eval arguments
    play_parser = subparsers.add_parser('play', help='Play using a trained model')
    play_parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model file')
    play_parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to play')

    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    elif args.command == 'play':
        play(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
