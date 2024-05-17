import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical


class Agent(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super().__init__()

        # Network architecture
        self.network = nn.Sequential( # Feature extractor
            self._layer_init(nn.Linear(obs_dim, 128)),  # First hidden layer
            nn.ReLU(),
            self._layer_init(nn.Linear(128, 256)),     # Second hidden layer
            nn.ReLU(),
            self._layer_init(nn.Linear(256, 128)),     # Third hidden layer
            nn.ReLU(),
        )

        # Actor
        self.actor = self._layer_init(nn.Linear(128, num_actions), std=0.01)
        
        # Critic
        self.critic = self._layer_init(nn.Linear(128,1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def get_value(self, x):
        return self.critic(self.network(x))
    
    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)