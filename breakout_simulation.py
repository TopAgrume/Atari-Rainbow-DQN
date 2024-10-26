import gymnasium as gym
import numpy as np
from deep_Q_learning_agent import DeepQLearningAgent
from gymnasium.wrappers import AtariPreprocessing
import ale_py
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def preprocess_observation(obs: np.ndarray) -> np.ndarray:
    """Preprocess Atari game frame for the DQN.

    Normalizes pixel values to range [0,1] and converts to float32.

    Args:
        obs (np.ndarray): Raw observation frame from the Atari environment

    Returns:
        np.ndarray: Preprocessed observation as float32 array with values in [0,1]
    """
    return np.array(obs).astype(np.float32) / 255.0


class BreakoutTrainer:
    """Handles training and evaluation of DQN agent for Atari Breakout."""

    def __init__(
            self,
            learning_rate: float = 0.0000625,
            epsilon: float = 1.0,
            gamma: float = 0.99,
            batch_size: int = 32,
            memory_size: int = 100000,
            epsilon_decay: float = 0.9999,
            epsilon_min: float = 0.05,
            target_update_frequency: int = 100,
            render_type: str = None,
            continue_training: bool = False,
            model_path: str = None,
            history_path: str = None
        ):
        """Initialize the Breakout trainer.

        Args:
            learning_rate: Learning rate for the Adam optimizer
            epsilon: Initial exploration rate
            gamma: Discount factor for future rewards
            batch_size: Size of training batches
            memory_size: Size of replay memory
            epsilon_decay: Rate at which epsilon decays
            epsilon_min: Minimum value for epsilon
            target_update_frequency: How often to update target network
        """
        # Setup environment
        gym.register_envs(ale_py)

        if render_type:
            self.env = gym.make("ALE/Breakout-v5", obs_type="rgb", frameskip=4, render_mode=render_type)
        else:
            self.env = gym.make("ALE/Breakout-v5", obs_type="grayscale", frameskip=4)

        self.env = AtariPreprocessing(self.env, frame_skip=1)
        self.env = gym.wrappers.FrameStackObservation(self.env, 4)
        n_actions = self.env.action_space.n


        # Init agent
        self.agent = DeepQLearningAgent(
            learning_rate=learning_rate,
            epsilon=epsilon,
            gamma=gamma,
            n_actions=n_actions,
            batch_size=batch_size,
            memory_size=memory_size
        )

        # Training parameters
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_frequency = target_update_frequency

        self.epoch_values = [0]
        self.max_reward = 0

        if continue_training and model_path:
            self._load_training_state(model_path, history_path)

        print("Action meanings:", self.env.unwrapped.get_action_meanings())


    def train(self, n_episodes: int = 30000, one_epoch: int = 50000):
        """Train the DQN agent on Breakout.

        Args:
            n_episodes (int): Number of episodes to train for
            one_epoch (int): Number of steps per training epoch
        """
        mean_reward = [0, 0]  # [total_reward, count]
        mini_batch = 0
        start_epoch = len(self.epoch_values)

        pbar = tqdm(range(n_episodes), desc="Starting")
        for episode in pbar:
            obs, _ = self.env.reset()
            state = preprocess_observation(obs)
            done = False
            total_reward = 0

            while not done:
                # Get action and step environment
                action = self.agent.get_action(state)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                # Update agent
                next_state = preprocess_observation(obs)
                self.agent.update(state, action, reward, next_state, done)
                state = next_state
                mini_batch += 1

                if done:
                    mean_reward[0] += total_reward
                    mean_reward[1] += 1

                # Print statistics (every 50000 batchs)
                if mini_batch % one_epoch == 0:
                    avg_reward = round(mean_reward[0] / mean_reward[1], 2)
                    current_epoch = start_epoch + (len(self.epoch_values) - start_epoch)

                    print(f"\nEpoch {current_epoch} stats:")
                    print(f"Mean reward: {avg_reward}")
                    print(f"Current epsilon: {round(self.agent.epsilon, 2)}")

                    self.epoch_values.append(avg_reward)
                    mean_reward = [0, 0]

                    self._save_training_state(current_epoch)

            agent_eps = round(self.agent.epsilon, 2)
            avg_reward = round(mean_reward[0] / mean_reward[1], 2)
            pbar.set_description(desc=f"Episode: {episode}, Reward: {avg_reward}, Epsilon: {agent_eps} |")

            # Decay epsilon
            self.agent.epsilon = max(self.epsilon_min, self.agent.epsilon * self.epsilon_decay)

            # Update target network
            if (episode + 1) % self.target_update_frequency == 0:
                save_max = False
                if self.epoch_values[-1] > self.max_reward:
                    self.max_reward = self.epoch_values[-1]
                    save_max = True
                    print(f"New best model with avg reward: {self.max_reward}")
                self.agent.update_target_network(save_max=save_max, max_reward=self.max_reward)

                # Plot training progress
                self._plot_progress()

    def play_games(self, num_episodes: int = 5):
        """Let the trained agent play games without learning.

        Args:
            num_episodes (int): Number of episodes to play
        """
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            state = preprocess_observation(obs)
            done = False
            total_reward = 0

            while not done:
                action = self.agent.get_action(state)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = preprocess_observation(obs)

            print(f"Episode {episode + 1} reward: {total_reward}")

    def _plot_progress(self):
        """Plot and save training progress graph.

        Args:
            epoch_values (list): List of average rewards per epoch
        """
        plt.figure(figsize=(15, 10))
        plt.plot(np.arange(len(self.epoch_values)), np.array(self.epoch_values))
        plt.title("Average reward on breakout")
        plt.xlabel("Training epochs")
        plt.ylabel("Average reward per episode")
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
            self.epoch_values = history_data.get('epoch_values', [0])
            self.max_reward = history_data.get('max_reward', 0)
            self.agent.epsilon = history_data.get('epsilon', self.agent.epsilon)
            print(f"Restored training from epoch {len(self.epoch_values)-1}")
            print(f"Continuing with epsilon: {self.agent.epsilon:.4f}")
            print(f"Previous max reward: {self.max_reward}")

    def _save_training_state(self, epoch: int):
        """Save current training state including history.

        Args:
            epoch: Current training epoch
        """
        history_data = {
            'epoch_values': self.epoch_values,
            'max_reward': self.max_reward,
            'epsilon': self.agent.epsilon
        }
        np.save(f'meta_data/training_history_epoch_{epoch}.npy', history_data)








