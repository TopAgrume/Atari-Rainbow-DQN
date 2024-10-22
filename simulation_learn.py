# exercise.py
import gymnasium as gym
import numpy as np
from deep_Q_learning_agent import DeepQLearningAgent
from gymnasium.wrappers import AtariPreprocessing
import ale_py
from tqdm import tqdm
import matplotlib.pyplot as plt

def preprocess_observation(obs):
    """_summary_

    Args:
        obs (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.array(obs).astype(np.float32) / 255.0

def main():
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5", obs_type="grayscale", frameskip=4)
    env = AtariPreprocessing(env, frame_skip=1)
    env = gym.wrappers.FrameStackObservation(env, 4)

    n_actions = env.action_space.n
    print("Available actions:", env.unwrapped.get_action_meanings(), end="\n\n")

    agent = DeepQLearningAgent(
        learning_rate=2.5e-4,
        epsilon=1.0,
        gamma=0.99,
        n_actions=n_actions,
        batch_size=32,
        memory_size=100000
    )

    n_episodes = 30000
    target_update_frequency = 100
    epsilon_decay = 0.9999
    epsilon_min = 0.05
    one_epoch = 50000

    mean_reward = [0, 0]
    mini_batch = 0
    epoch_value = [0]

    max_reward = 0

    for episode in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        state = preprocess_observation(obs)
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            next_state = preprocess_observation(obs)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            mini_batch += 1

            if done:
                mean_reward[0] += total_reward
                mean_reward[1] += 1

            if mini_batch % one_epoch == 0:
                mean_reward = round(mean_reward[0] / mean_reward[1], 2)
                print(f"Mean reward for a new epoch (minibatch {mini_batch - one_epoch} to {one_epoch}):", mean_reward)
                print("Current agent epsilon:", round(agent.epsilon, 2))
                epoch_value.append(mean_reward)
                mean_reward = [0, 0]

        # Decay epsilon
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)

        # Update target network and plot stats
        if (episode + 1) % target_update_frequency == 0:
            save_max_network = False
            if epoch_value[-1] > max_reward:
                max_reward = epoch_value[-1]
                save_max_network = True
                print(f"New max_reward net with r={max_reward}")
            agent.update_target_network(save_max=save_max_network, max_reward=max_reward)

            plt.figure(figsize=(15, 10))
            plt.plot(np.arange(len(epoch_value)), np.array(epoch_value))
            plt.title("Average reward on breakout")
            plt.xlabel("Training epochs")
            plt.ylabel("Average reward per episode")
            plt.savefig(f"figures/reward_breakout_epoch_{len(epoch_value) - 1}.png")
            plt.close()


    env.close()

if __name__ == "__main__":
    main()
