import numpy as np
from random import sample

import torch
from vizdoom import *

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resolution = (30, 45)
config_path = "basic.cfg"

class ReplayMemory:
    def __init__(self, replay_capacity):
        channels = 1
        self.replay_capacity = replay_capacity
        self.position = 0
        self.size = 0
        shape = (replay_capacity, channels, *resolution)
        self.state1 = torch.zeros(shape, dtype=torch.float32).to(torch_device)
        self.state2 = torch.zeros(shape, dtype=torch.float32).to(torch_device)
        self.act = torch.zeros(replay_capacity, dtype=torch.long).to(torch_device)
        self.rew = torch.zeros(replay_capacity, dtype=torch.float32).to(torch_device)
        self.isfinal = torch.zeros(replay_capacity, dtype=torch.float32).to(torch_device)
        
    def transition(self, state1, action, state2, isfinal, reward):
        ind = self.position
        self.state1[ind,0,:,:] = state1
        self.act[ind] = action
        if not isfinal:
            self.state2[ind,0,:,:] = state2
        self.isfinal[ind] = isfinal
        self.rew[ind] = reward

        self.position = (self.position + 1) % self.replay_capacity
        self.size = min(self.size + 1, self.replay_capacity)

    def sample(self, size):
        ind = sample(range(0, self.size), size)
        return (self.state1[ind], self.act[ind], self.state2[ind], self.isfinal[ind], self.rew[ind])