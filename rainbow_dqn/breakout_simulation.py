import gymnasium as gym
import numpy as np
from deep_q_learning_agent import DeepQLearningAgent
from gymnasium.wrappers import AtariPreprocessing
import ale_py
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from PIL import Image

class BreakoutTrainer:
    """Handles training and evaluation of DQN agent for Atari Breakout."""

    def __init__(
            self,
            learning_rate: float = 0.0000625,
            gamma: float = 0.99,
            batch_size: int = 32,
            memory_size: int = 100000,
            target_update_frequency: int = 5000,
            render_type: str = None,
            continue_training: bool = False,
            model_path: str = None,
            history_path: str = None,
            replay_start_size: int = 80000
        ):
        """Initialize the Breakout trainer.

        Args:
            learning_rate (float): Learning rate for the Adam optimizer.
            gamma (float): Discount factor for future rewards.
            batch_size (int): Size of training batches.
            memory_size (int): Size of the replay memory buffer.
            target_update_frequency (int): Number of steps between updates to the target network.
            render_type (str, optional): Rendering mode ('human' for live view or None).
            continue_training (bool, optional): Whether to resume training from a saved model.
            model_path (str, optional): Path to the saved model file for continued training.
            history_path (str, optional): Path to the training history file for continued training.
            replay_start_size (int): Minimum number of transitions in replay memory before training starts.
        """
        # Setup environment
        gym.register_envs(ale_py)

        if render_type is not None:
            print("Human rendering...")
            self.env = gym.make("ALE/Breakout-v5", frameskip=4, render_mode=render_type)
        else:
            print("Training rendering...")
            self.env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)

        self.env = AtariPreprocessing(self.env, grayscale_obs=True, frame_skip=1, scale_obs=True)
        self.env = gym.wrappers.FrameStackObservation(self.env, 4)
        n_actions = self.env.action_space.n


        # Init agent
        self.agent = DeepQLearningAgent(
            learning_rate=learning_rate,
            gamma=gamma,
            n_actions=n_actions,
            batch_size=batch_size,
            memory_size=memory_size,
            replay_start_size=replay_start_size
        )

        # Training parameters
        self.target_update_frequency = target_update_frequency

        self.epoch_values = [0]
        self.std_values = [0]
        self.max_rewards = [0]
        self.min_rewards = [0]
        self.mean_reward = 0

        if continue_training and model_path:
            self._load_training_state(model_path, history_path)

        print("Action meanings:", self.env.unwrapped.get_action_meanings())
        print("Target net update frquency:", target_update_frequency, end="\n\n")

    def train(self, n_episodes: int = 500000, one_epoch: int = 50000):
        """Train the DQN agent on Breakout.

        Args:
            n_episodes (int): Number of episodes to train for
            one_epoch (int): Number of steps per training epoch
        """
        step = 0
        max_reward = float('-inf')
        min_reward = float('inf')

        rewards_epoch = []
        start_epoch = len(self.epoch_values)

        pbar = tqdm(range(1, n_episodes), desc="Starting")
        for episode in pbar:
            state, _ = self.env.reset()
            done = False
            total_reward = 0.0
            current_frame = step

            while not done and step < current_frame + 108_000:
                self.agent.policy_net.reset_noise()

                # Get action and step environment
                action = 1 if step == current_frame else self.agent.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated or info['lives'] != 5

                total_reward -= 1 if done else - max(-1, min(1, reward))

                # Update agent
                self.agent.update(state, action, reward, next_state, done)
                state = next_state
                step += 1

                if done:
                    rewards_epoch.append(total_reward)

                    # Update progression description
                    avg_reward = round(np.mean(rewards_epoch).item(), 2)
                    max_reward = round(max(max_reward, total_reward), 2)
                    min_reward = round(min(min_reward, total_reward), 2)
                    std_reward = round(np.std(rewards_epoch).item(), 2)

                    pbar.set_description(
                        desc=f"Episode: {episode} (mean: {avg_reward}, std: {std_reward}, min: {min_reward}, max: {max_reward}) |"
                    )

                # Print and save statistics (every 50000 batchs)
                if step % one_epoch == 0:
                    current_epoch = start_epoch + (len(self.epoch_values) - start_epoch)

                    avg_reward = round(np.mean(rewards_epoch).item(), 2)
                    max_reward = round(max(max_reward, total_reward), 2)
                    min_reward = round(min(min_reward, total_reward), 2)
                    std_reward = round(np.std(rewards_epoch).item(), 2)

                    print(f"\n===== Epoch {current_epoch} stats =====")
                    print(f"Min reward: {min_reward}")
                    print(f"Max reward: {max_reward}")
                    print(f"Mean reward: {avg_reward}")
                    print(f"Std reward: {std_reward}\n")

                    self.epoch_values.append(avg_reward)
                    self.std_values.append(std_reward)
                    self.min_rewards.append(min_reward)
                    self.max_rewards.append(max_reward)

                    # Plot and save training progress
                    self._plot_progress()
                    self._save_training_state(current_epoch)

                    rewards_epoch.clear()
                    max_reward = float('-inf')
                    min_reward = float('inf')

                # Update target network
                if step % self.target_update_frequency == 0:
                    save_max = False
                    if self.epoch_values[-1] > self.mean_reward:
                        self.mean_reward = self.epoch_values[-1]
                        save_max = True
                        print(f"==> New best model with avg reward: {self.mean_reward}")
                    current_epoch = start_epoch + (len(self.epoch_values) - start_epoch)
                    self.agent.update_target_network(current_epoch, save_max=save_max, max_reward=self.mean_reward)



    def play_games(self, num_episodes: int = 5):
        """Play games without learning.

        Args:
            num_episodes (int): Number of episodes to play
        """
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            overide_action = True
            current_lives = 5

            while not done:
                action = 1 if overide_action else self.agent.get_action(state)
                state, reward, terminated, truncated, info = self.env.step(action)

                overide_action = current_lives != info['lives']
                if overide_action:
                    current_lives = info['lives']

                done = terminated or truncated
                total_reward += reward

            print(f"Episode {episode + 1} reward: {total_reward}")

    def _plot_progress(self):
        """Plot and save training progress graph.

        Args:
            epoch_values (list): List of average rewards per epoch
        """
        plt.figure(figsize=(15, 10))
        epochs = np.arange(len(self.epoch_values))
        epoch_values = np.array(self.epoch_values)
        std_dev = np.array(self.std_values)
        mins = np.array(self.min_rewards)
        maxs = np.array(self.max_rewards)

        plt.plot(epochs, epoch_values, label="average reward", color='blue', linewidth=2)
        plt.plot(epochs, mins, label="minimum reward", color='red', linestyle='--', marker='o', markersize=2)
        plt.plot(epochs, maxs, label="maximum reward", color='green', linestyle='--', marker='o', markersize=2)
        plt.fill_between(epochs, epoch_values - std_dev, epoch_values + std_dev,
                     color='blue', alpha=0.15, label="std dev")
        plt.axhline(y=31.8, color='purple', linestyle='--', label="Human performance")

        plt.title("Average reward with std dev and extreme values on breakout")
        plt.xlabel("Training epochs")
        plt.ylabel("Average reward per episode")
        plt.grid(visible=True, linestyle='--', linewidth=0.5)
        plt.legend()
        plt.savefig(f"figures/reward_breakout_epoch_{len(self.epoch_values) - 1}.png")
        plt.close()

    def _load_training_state(self, model_path: str, history_path: str = None):
        """Load previous training state including model and history.

        Args:
            model_path: Path to the saved model file
            history_path: Path to the saved training history (optional)
        """
        print(f"Loading model from {model_path}")
        self.agent.load_saved_model(model_path)

        # Load training history
        if history_path and os.path.exists(history_path):
            print(f"Loading training history from {history_path}...")
            history_data = np.load(history_path, allow_pickle=True).item()
            self.epoch_values = history_data.get('reward', [0])
            self.mean_reward = history_data.get('mean_reward', 0)
            print(f"Restored training from epoch {len(self.epoch_values) - 1}")
            print(f"Previous max reward: {self.mean_reward}")

    def _save_training_state(self, epoch: int):
        """Save current training state including history.

        Args:
            epoch: Current training epoch
        """
        history_data = {
            'mean_rewards': self.epoch_values,
            'best_mean_reward': self.mean_reward,
            'std_values': self.std_values,
            'min_rewards': self.min_rewards,
            'max_rewards': self.max_rewards
        }
        np.save(f'meta_data/training_history_epoch_{epoch}.npy', history_data)
