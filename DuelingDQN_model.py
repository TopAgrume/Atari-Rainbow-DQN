import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.modules import NoisyLinear

class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network (DQN) architecture implementation.

    This network uses a dueling architecture that separates state value and action
    advantage estimations. This separation helps in better policy evaluation by
    explicitly decomposing the Q-value into the value of the state and the advantage
    of each action.

    See implementation here: https://lzzmm.github.io/2021/11/05/breakout/

    The network processes 4 stacked frames as input through 3 convolutional layers,
    followed by separate streams for value and advantage estimation.
    """
    def __init__(self, n_actions: int, std_init = 0.5):
        """Initialization of the Dueling DQN architecture.

        Args:
            n_actions (int): Number of possible actions in the action space
            std_init (float, optional): Initial standard deviation for NoisyLinear layers
        """
        super(DuelingDQN, self).__init__()

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Value stream layers
        self.fc1_a = NoisyLinear(64 * 7 * 7, 512, std_init=std_init)     # Advantage stream first dense layer
        self.fc2_a = NoisyLinear(512, n_actions, std_init=std_init)      # Advantage stream output layer

        # Advantage stream layers
        self.fc1_b = NoisyLinear(64 * 7 * 7, 512, std_init=std_init)     # State-value stream first dense layer
        self.fc2_b = NoisyLinear(512, 1, std_init=std_init)              # State-value stream output layer

        self.n_actions = n_actions

    def forward(self, x: torch.Tensor):
        """
        Processes the input through convolutional layers, then splits into value and
        advantage streams. Combines them using the dueling architecture formula:
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 4, height, width)
                              containing 4 stacked frames

        Returns:
            torch.Tensor: Q-values for each action, shape (batch_size, n_actions)
        """
        # Shared convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        xv = x.view(x.size(0), -1)

        # Value stream
        vs = F.relu(self.fc1_b(xv))
        vs = self.fc2_b(vs).expand(x.size(0), self.n_actions)

        # Advantage stream
        asa = F.relu(self.fc1_a(xv))
        asa = self.fc2_a(asa)

        # Combine value and advantage streams
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return vs + asa - asa.mean(1).unsqueeze(1).expand(x.size(0), self.n_actions)

    def reset_noise(self):
        """Resets the noise in NoisyLinear layers to encourage exploration."""
        self.fc1_a.reset_noise()
        self.fc2_a.reset_noise()
        self.fc1_b.reset_noise()
        self.fc2_b.reset_noise()

    def save_model(self, filename: str):
        """Save the model's state dict to a file.

        The model is saved in the './models/' directory
        """
        torch.save(self.state_dict(), './models/' + filename + '.pth')

    def load_model(self, filename: str):
        """Load the model's state dict from a file."""
        self.load_state_dict(torch.load(filename))

def create_DQN_model(n_actions: int) -> DuelingDQN:
    """Create a new Dueling DQN model.

    Args:
        n_actions (int): Number of possible actions in the action space

    Returns:
        DuelingDQN: Initialized Dueling DQN model
    """
    return DuelingDQN(n_actions)
