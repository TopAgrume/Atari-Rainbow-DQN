# exercise.py
import gymnasium as gym
import numpy as np
from agent import DeepQLearningAgent
from gymnasium.wrappers import AtariPreprocessing
import ale_py
from tqdm import tqdm
import matplotlib.pyplot as plt

def preprocess_observation(obs):
    return np.array(obs).astype(np.float32) / 255.0

def main():
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", obs_type="grayscale", frameskip=4)
    env = AtariPreprocessing(env, frame_skip=1)
    env = gym.wrappers.FrameStackObservation(env, 4)

    n_actions = env.action_space.n
    print("Available actions:", env.unwrapped.get_action_meanings(), end="\n\n")

    agent = DeepQLearningAgent(
        learning_rate=0.0001,
        epsilon=1.0,
        gamma=0.99,
        n_actions=n_actions,
        batch_size=32,
        memory_size=100000
    )

    n_episodes = 500000
    target_update_frequency = 50000
    epsilon_decay = 0.9995
    epsilon_min = 0.05
    one_epoch = 50000

    mean_reward = []
    mini_batch = 0
    max_reward = float('-inf')
    min_reward = float('inf')

    epoch_values = [0]
    std_values = [0]
    max_rewards = [0]
    min_rewards = [0]

    best_reward = 0
    pbar = tqdm(range(1, n_episodes), desc="Starting")
    for episode in pbar:
        obs, _ = env.reset()
        state = preprocess_observation(obs)
        done = False
        total_reward = 0.0
        current_frame = mini_batch

        while not done:
            action = 1 if current_frame == mini_batch else agent.get_action(state)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated or info['lives'] != 5

            total_reward -= 1 if done else - max(-1, min(1, reward))

            next_state = preprocess_observation(obs)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            mini_batch += 1

            if done:
                mean_reward.append(total_reward)

                # Update progression description
                avg_reward = round(np.mean(mean_reward).item(), 2)
                max_reward = round(max(max_reward, total_reward), 2)
                min_reward = round(min(min_reward, total_reward), 2)
                std_reward = round(np.std(mean_reward).item(), 2)

                pbar.set_description(
                    desc=f"Episode: {episode} (mean: {avg_reward}, std: {std_reward}, min: {min_reward}, max: {max_reward}) |"
                )

            if mini_batch % one_epoch == 0:
                current_epoch = len(epoch_values)
                avg_reward = round(np.mean(mean_reward).item(), 2)
                max_reward = round(max(max_reward, total_reward), 2)
                min_reward = round(min(min_reward, total_reward), 2)
                std_reward = round(np.std(mean_reward).item(), 2)

                print(f"\n===== Epoch {current_epoch} stats =====")
                print(f"Min reward: {min_reward}")
                print(f"Max reward: {max_reward}")
                print(f"Mean reward: {avg_reward}")
                print(f"Std reward: {std_reward}\n")
                print("Current agent epsilon:", round(agent.epsilon, 2))
                mean_reward.clear()

                epoch_values.append(avg_reward)
                std_values.append(std_reward)
                min_rewards.append(min_reward)
                max_rewards.append(max_reward)

                max_reward = float('-inf')
                min_reward = float('inf')

            # Update target network and plot stats
            if mini_batch % target_update_frequency == 0:
                current_epoch = len(epoch_values)
                save_max_network = False
                if epoch_values[-1] > best_reward:
                    best_reward = epoch_values[-1]
                    save_max_network = True
                    print(f"New max_reward net with r={max_reward}")
                agent.update_target_network(save_max=save_max_network, max_reward=max_reward, epoch=current_epoch)

                plt.figure(figsize=(15, 10))
                epochs = np.arange(len(epoch_values))
                eeepoch_values = np.array(epoch_values)
                std_dev = np.array(std_values)
                mins = np.array(min_rewards)
                maxs = np.array(max_rewards)

                plt.plot(epochs, eeepoch_values, label="average reward", color='blue', linewidth=2)
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
                plt.savefig(f"figures/reward_breakout_epoch_{len(epoch_values) - 1}.png")
                plt.close()

        # Decay epsilon
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)


    env.close()

if __name__ == "__main__":
    main()
