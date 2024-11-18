import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from DuelingDQN_model import create_DQN_model
from torchrl.data import ListStorage, PrioritizedReplayBuffer

class DeepQLearningAgent():
    """Deep Q-Learning agent implementation with experience replay and target network.

    This agent implements Deep Q-Learning with several key features following
    the "Playing Atari with Deep Reinforcement Learning" paper:
    - Experience replay buffer to store and sample transitions
    - Separate target/policy network for stable training
    - Dueling DQN architecture for improved Q-value estimation.
    """
    def __init__(
        self,
        learning_rate: float,
        gamma: float,
        n_actions: int,
        batch_size: int = 32,
        memory_size: int = 100000,
        replay_start_size: int = 80000
    ):
        """Initialize the Deep Q-Learning agent.

        Args:
            learning_rate (float): Learning rate for the Adam optimizer.
            gamma (float): Discount factor for future rewards.
            n_actions (int): Number of possible actions in the environment.
            batch_size (int, optional): Size of training batches. Defaults to 32.
            memory_size (int, optional): Size of the replay memory buffer. Defaults to 100000.
            replay_start_size (int, optional): Minimum number of transitions in memory before training starts. Defaults to 80000.
        """
        # Agent parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size

        # Replay memory
        self.memory = PrioritizedReplayBuffer(
            alpha=0.6, beta=0.4, storage=ListStorage(memory_size)
        )

        # Initialize networks and torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = create_DQN_model(n_actions).to(self.device)
        self.target_net = create_DQN_model(n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate, eps=1.5e-4)
        self.loss_fn = nn.MSELoss()

        # Print params
        self._print_init_info(memory_size)


    def get_action(self, state: np.ndarray) -> int:
        """Select an action using epsilon-greedy policy.

        Args:
            state (np.ndarray): Current state observation from the environment

        Returns:
            int: Selected action index
        """

        # Exploitation: best action according to policy network
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)

            # Action that maximize the approximation of Q*(s, a, theta)
            return q_values.argmax().item()

    def update(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool
        ) -> None:
        """Update the agent by storing the transition and performing a learning step.

        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received after taking the action.
            next_state (np.ndarray): Observed next state after taking the action.
            done (bool): Whether the episode has terminated.
        """
        # Store transition in replay memory
        self._save_to_memory(state, action, reward, next_state, done)

        if len(self.memory) < self.replay_start_size:
            return
        elif len(self.memory) == self.replay_start_size:
            print(f"Learning phase started, memory length: {self.replay_start_size}")

        # Sample and process batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self._process_batch(batch)

        # Reset NoisyLinear layers
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        # Compute target Q-values using target network
        next_q_values = self.target_net(next_states).max(1).values.detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute the current Q-values and loss
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        loss = self.loss_fn(current_q_values, target_q_values.unsqueeze(1))

        # Optimize and stabilize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self, epoch: int, save_max: bool = False, max_reward: int = 0):
        """Update the target network by copying weights from the policy network.

        Args:
            epoch (int): Current training epoch for saving checkpoints.
            save_max (bool, optional): Whether to save the model as the best model. Defaults to False.
            max_reward (float, optional): Current maximum reward for naming the checkpoint file. Defaults to 0.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.save_model(f"current_{epoch}_dueling_dqn")

        # Save the best model
        if save_max:
            self.policy_net.save_model(f"max_dueling_dqn_{max_reward}")

    def load_saved_model(self, filename: str) -> None:
        """Load a saved model state dict.

        Args:
            filename (str): filename of saved model state dict
        """
        self.policy_net.load_model(filename)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _save_to_memory(self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool
        ):
        """Save a transition tuple into the replay memory.

        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received after taking the action.
            next_state (np.ndarray): Observed next state after taking the action.
            done (bool): Whether the episode has terminated.
        """
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)

        action = torch.tensor(action, dtype=torch.uint8)
        done = torch.tensor(done, dtype=torch.uint8)
        reward = torch.tensor(reward, dtype=torch.float32)

        self.memory.add((state, action, reward, next_state, done))

    def _process_batch(self, batch: list) -> tuple:
        """Process a batch of transitions into tensors for training.

        Args:
            batch (List[Tuple]): Batch of (state, action, reward, next_state, done) tuples

        Returns:
            tuple: Processed tensors for each object.
        """
        states = batch[0].to(self.device)
        actions = batch[1].to(torch.int64).to(self.device)
        rewards = batch[2].to(self.device)
        next_states = batch[3].to(self.device)
        dones = batch[4].to(self.device)

        return states, actions, rewards, next_states, dones

    def _print_init_info(self, memory_size: int) -> None:
        """Print initialization information about the agent.

        Args:
            memory_size (int): Size of replay memory
        """
        print("\n======== DeepQLearningAgent ========")
        print("Learning rate:", self.learning_rate)
        print("Gamma:", self.gamma)
        print("Batch size:", self.batch_size)
        print("Max memory size:", memory_size)
        print("Number of actions:", self.n_actions, end="\n\n")
        print("Current device used:", self.device)
        print("Optimizer:", "Adam()")
        print("Loss:", self.loss_fn)
        print("====================================", end="\n\n")
