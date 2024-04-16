from vizdoom import *

import numpy as np
from random import randint, random, sample

import torch
from torch import nn
import torch.nn.functional as F
from absl import flags
from skimage.transform import resize

config_path = "basic.cfg"
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Replay Memory

resolution = (30, 45)

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

# DNN Model
FLAGS = flags.FLAGS
class QNN(nn.Module):
    def __init__(self, available_actions_count):
        super(QNN, self).__init__()
        # Layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3) 
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2) 
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, available_actions_count)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), FLAGS.learning_rate)
        self.memory = ReplayMemory(capacity=FLAGS.replay_memory)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 192)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_best_action(self, state):
        q = self(state)
        _, index = torch.max(q, 1)
        return index

    def train_step(self, state1, target_q):
        output = self(state1)
        loss = self.criterion(output, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def learn_from_memory(self):
        if self.memory.size < FLAGS.batch_size: return
        state1, act, state2, isfinal, r = self.memory.sample(FLAGS.batch_size)
        q = self(state2).detach()
        q2, _ = torch.max(q, dim=1)
        target_q = self(state1).detach()
        idxs = (torch.arange(target_q.shape[0]), act)
        target_q[idxs] = r + FLAGS.discount * (1-isfinal) * q2
        self.train_step(state1, target_q)

# Agent performing actions

def preprocess(img):
    return torch.from_numpy(resize(img, resolution).astype(np.float32))

def game_state(game):
    return preprocess(game.get_state().screen_buffer)

