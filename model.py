import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, mode, fc1_units=127, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            mode (string) : dqn improvement methods (vanilla, double, prioritized, dueling, rainbow)
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.mode = mode
        
        self.action_value_fc1 = nn.Linear(state_size, fc1_units)
        self.action_value_fc2 = nn.Linear(fc1_units, fc2_units)
        self.action_value_fc3 = nn.Linear(fc2_units, action_size)
        
        if((mode == "dueling") or (mode == "rainbow")):
            self.state_value_fc1 = nn.Linear(state_size, fc1_units)
            self.state_value_fc2 = nn.Linear(fc1_units, fc2_units)
            self.state_value_fc3 = nn.Linear(fc2_units, 1)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        a = F.relu(self.action_value_fc1(state))
        a = F.relu(self.action_value_fc2(a))
        
        if((self.mode == "dueling") or (self.mode == "rainbow")):
            s = F.relu(self.state_value_fc1(state))
            s = F.relu(self.state_value_fc2(s))
            q = self.action_value_fc3(a).add(self.state_value_fc3(s))
            return q
        return self.action_value_fc3(a)

class ConvQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, mode, fc1_units=127, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            mode (string) : dqn improvement methods (vanilla, double, prioritized, dueling, rainbow)
        """
        super(ConvQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.mode = mode

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.action_value_fc1 = nn.Linear(7*7*32, 128)
        self.action_value_fc2 = nn.Linear(128, action_size)
        
        if((mode == "dueling") or (mode == "rainbow")):
            self.state_value_fc1 = nn.Linear(7*7*32, 128)
            self.state_value_fc2 = nn.Linear(128, 1)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        state = state.permute(0, 3, 1, 2)
        feature = F.relu(self.bn1(self.conv1(state)))
        feature = F.relu(self.bn2(self.conv2(feature)))
        feature = F.relu(self.bn3(self.conv3(feature)))
        feature = feature.view(feature.size()[0], -1)
        
        a = F.relu(self.action_value_fc1(feature))
        if((self.mode == "dueling") or (self.mode == "rainbow")):
            s = F.relu(self.state_value_fc1(state))
            q = self.action_value_fc2(feature).add(self.state_value_fc2(s))
            return q
        return self.action_value_fc2(a)
