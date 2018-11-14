import numpy as np
import random
from model import QNetwork,ConvQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

from openai.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
PRIORITIZED_ALPHA = 0.6
PRIORITIZED_BETA = 0.4
PRIORITIZED_EPS = 1e-6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, mode="vanilla", visual=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            mode (string) : dqn improvement methods (vanilla, double, prioritized, dueling, rainbow)
        """
        self.mode = mode
        if(self.mode not in ["vanilla", "double", "prioritized", "dueling", "rainbow"]):
            raise NameError('Wrong mode input')
        if(self.mode in ["prioritized","rainbow"]):
            self.prioritized_replay = True
        else:
            self.prioritized_replay = False
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.visual = visual
        # Q-Network
        if(self.visual):
            self.qnetwork_local = ConvQNetwork(state_size, action_size, seed, mode).to(device)
            self.qnetwork_target = ConvQNetwork(state_size, action_size, seed, mode).to(device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed, mode).to(device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed, mode).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        if(self.prioritized_replay):
            self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, PRIORITIZED_ALPHA, PRIORITIZED_BETA)
        else:
            self.memory = ReplayBuffer(BUFFER_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        if(self.visual):
            state = np.squeeze(state, axis=0)
            next_state = np.squeeze(next_state, axis=0)
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample(BATCH_SIZE)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        if(self.visual):
            state = torch.from_numpy(state).float().to(device)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if(self.prioritized_replay):
            states, actions, rewards, next_states, dones, weights, batch_idxes = experiences
        else:
            states, actions, rewards, next_states, dones = experiences
            weights, batch_idxes = torch.ones_like(rewards), None
        # Get max predicted Q values (for next states) from target model
        if(self.mode in ["double","rainbow"]):
            next_actions = torch.argmax(self.qnetwork_local(states),dim=1).unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).gather(1,next_actions)
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        td_error = torch.abs(Q_expected - Q_targets)
        # Update Prioritized replay buffer
        if(self.prioritized_replay):
            new_priorities = td_error.cpu().data.numpy() + PRIORITIZED_EPS
            self.memory.update_priorities(batch_idxes, new_priorities)
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def save(self):
        """Save model weights.
        """
        if(self.visual):
            torch.save(self.qnetwork_local.state_dict(), "saved_weights/{}_visual_local.pt".format(self.mode))
            torch.save(self.qnetwork_target.state_dict(), "saved_weights/{}_visual_target.pt".format(self.mode))
        else:  
            torch.save(self.qnetwork_local.state_dict(), "saved_weights/{}_local.pt".format(self.mode))
            torch.save(self.qnetwork_target.state_dict(), "saved_weights/{}_target.pt".format(self.mode))

    def load(self):
        """Load saved model weights.
        """
        if(self.visual):
            self.qnetwork_local.load_state_dict(torch.load("saved_weights/{}_visual_local.pt".format(self.mode)))
            self.qnetwork_target.load_state_dict(torch.load("saved_weights/{}_visual_target.pt".format(self.mode)))
        else:
            self.qnetwork_local.load_state_dict(torch.load("saved_weights/{}_local.pt".format(self.mode)))
            self.qnetwork_target.load_state_dict(torch.load("saved_weights/{}_target.pt".format(self.mode)))
        