import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from DQN import create_DQN_model

Action = int
State = int

class DeepQLearningAgent():
    def __init__(
        self,
        learning_rate: float,
        epsilon: float,
        gamma: float,
        n_actions: int,
        batch_size: int = 32,
        memory_size: int = 100000
    ):
        """_summary_

        Args:
            learning_rate (float): _description_
            epsilon (float): _description_
            gamma (float): _description_
            n_actions (int): _description_
            batch_size (int, optional): _description_. Defaults to 32.
            memory_size (int, optional): _description_. Defaults to 100000.
        """
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = create_DQN_model(n_actions).to(self.device)
        self.target_net = create_DQN_model(n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        print("========DeepQLearningAgent========")
        print("Learning rate:", self.learning_rate)
        print("Epsilon:", self.epsilon)
        print("Gamma:", self.gamma)
        print("Batch size:", self.batch_size)
        print("Max memory size:", memory_size)
        print("Number of actions:", self.n_actions, end="\n\n")
        print("Current device used:", self.device)
        print("Optimizer:", "Adam()")
        print("Loss:", self.loss_fn)
        print("==================================")


    def get_action(self, state: State) -> Action:
        """_summary_

        Args:
            state (State): _description_

        Returns:
            Action: _description_
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)

                # Retrieve the action that maximize the approximation of Q*(s, a, theta)
                return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done, verbose):
        """_summary_

        Args:
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            next_state (_type_): _description_
            done (function): _description_
            verbose (_type_): _description_
        """
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) < self.batch_size:
            return

        if verbose:
            print("Length of the current replays", len(self.memory))

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        # Set the label y
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        # -> The goal is to delete the second part of equation in cases done=True
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # The so called prediction
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        loss = self.loss_fn(current_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self, save_max=False):
        """_summary_

        Args:
            save_max (bool, optional): _description_. Defaults to False.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
        torch.save(self.policy_net.state_dict(), "model_saves/model_load_state_dict.pt")
        if save_max:
            torch.save(self.policy_net.state_dict(), "model_saves/max_model_load_state_dict.pt")
