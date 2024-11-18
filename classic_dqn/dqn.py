import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) implementation using PyTorch.

    This model is designed for reinforcement learning tasks, specifically to approximate the Q-value function
    for an agent's policy. The architecture consists of convolutional layers followed by fully connected layers,
    which makes it suitable for processing visual input.
    """
    def __init__(self, n_actions: int):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    # Save a model
    def save_model(self):
        torch.save(self.state_dict(), './models/' + self.filename + '.pth')

    # Loads a model
    def load_model(self):
        self.load_state_dict(torch.load('./models/' + self.filename + '.pth'))

def create_DQN_model(n_actions: int) -> DQN:
    return DQN(n_actions)
