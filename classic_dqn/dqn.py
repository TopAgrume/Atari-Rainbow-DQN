import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self, n_actions: int):
        """_summary_

        Args:
            n_actions (_type_): _description_
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    # Save a model
    def save_model(self):
        """_summary_
        """
        torch.save(self.state_dict(), './models/' + self.filename + '.pth')

    # Loads a model
    def load_model(self):
        """_summary_
        """
        self.load_state_dict(torch.load('./models/' + self.filename + '.pth'))

def create_DQN_model(n_actions: int) -> DQN:
    """_summary_

    Args:
        n_actions (int): _description_

    Returns:
        DuelingDQN: _description_
    """
    return DQN(n_actions)
