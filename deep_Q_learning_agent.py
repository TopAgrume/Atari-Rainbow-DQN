import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from DQN_model import create_DQN_model

class DeepQLearningAgent():
    """Deep Q-Learning agent implementation with experience replay and target network.

    This agent implements Deep Q-Learning with several key features following
    the "Playing Atari with Deep Reinforcement Learning" paper:
    - Experience replay buffer to store and sample transitions
    - Separate target network for stable training
    - Epsilon-greedy exploration strategy
    - Dueling DQN architecture
    """
    def __init__(
        self,
        learning_rate: float,
        epsilon: float,
        gamma: float,
        n_actions: int,
        batch_size: int = 32,
        memory_size: int = 100000
    ):
        """Initialize the Deep Q-Learning agent.

        Args:
            learning_rate (float): Learning rate for the Adam optimizer
            epsilon (float): Initial exploration rate for epsilon-greedy policy
            gamma (float): Discount factor for future rewards
            n_actions (int): Number of possible actions in the environment
            batch_size (int, optional): Size of training batches. Defaults to 32.
            memory_size (int, optional): Size of replay memory. Defaults to 100000.
        """
        # Agent parameters
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_actions = n_actions
        self.batch_size = batch_size

        # Replay memory
        self.memory = deque(maxlen=memory_size)

        # Initialize networks and torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = create_DQN_model(n_actions).to(self.device)
        self.target_net = create_DQN_model(n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

        print("======== DeepQLearningAgent ========")
        print("Learning rate:", self.learning_rate)
        print("Epsilon:", self.epsilon)
        print("Gamma:", self.gamma)
        print("Batch size:", self.batch_size)
        print("Max memory size:", memory_size)
        print("Number of actions:", self.n_actions, end="\n\n")
        print("Current device used:", self.device)
        print("Optimizer:", "Adam()")
        print("Loss:", self.loss_fn)
        print("====================================", end="\n\n")


    def get_action(self, state: np.ndarray) -> int:
        """Select an action using epsilon-greedy policy.

        Args:
            state (np.ndarray): Current state observation from the environment

        Returns:
            int: Selected action index
        """
        # Exploration: random action
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        # Exploitation: best action according to policy network
        else:
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
        """Update the agent.

        Stores the transition in replay memory and performs a learning step.

        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether the episode ended
        """
        # Store transition in replay memory
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) < self.batch_size:
            return

        # Sample and process batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = self._process_batch(batch)

        # Compute target Q-values using target network
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute the current Q-values and loss
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        loss = self.loss_fn(current_q_values, target_q_values.unsqueeze(1))

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self, save_max=False, max_reward=0):
        """Update the target network by copying weights from policy network.
        Also saves the current model state.

        Args:
            save_max (bool, optional): Whether to save as best model. Defaults to False.
            max_reward (float, optional): Current max reward (for filename). Defaults to 0.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.save_model("dueling_dqn")

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

    def _process_batch(self, batch):
        """Process a batch of transitions into tensors for training.

        Args:
            batch (List[Tuple]): Batch of (state, action, reward, next_state, done) tuples

        Returns:
            tuple: Processed tensors for each object.
        """
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        return states, actions, rewards, next_states, dones

    def _print_init_info(self, memory_size: int) -> None:
        """Print initialization information about the agent.

        Args:
            memory_size (int): Size of replay memory
        """
        print("======== DeepQLearningAgent ========")
        print("Learning rate:", self.learning_rate)
        print("Epsilon:", self.epsilon)
        print("Gamma:", self.gamma)
        print("Batch size:", self.batch_size)
        print("Max memory size:", memory_size)
        print("Number of actions:", self.n_actions, end="\n\n")
        print("Current device used:", self.device)
        print("Optimizer:", "Adam()")
        print("Loss:", self.loss_fn)
        print("====================================", end="\n\n")