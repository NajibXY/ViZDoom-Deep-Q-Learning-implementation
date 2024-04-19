# Code adapted from the works of FARAMA-FOUNDATION https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python/learning_pytorch.py
# And Brandon Morris https://brandonmorris.dev/2018/10/09/dql-vizdoom/

from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution
from absl import app,flags
from skimage.transform import resize
from torch import nn
from random import randint, random, sample
from time import time, sleep
from tqdm import trange
import numpy as np
import torch
import torch.nn.functional as F
import itertools

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resolution = (30, 45)
frame_repeat = 12
basic_config = 'basic.cfg'
deadly_corridor_config = 'deadly_corridor.cfg'
defend_the_center_config = 'defend_the_center.cfg'

# Replay Memory

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
        self.memory = ReplayMemory(replay_capacity=FLAGS.replay_memory_size)

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
        target_q[idxs] = r + FLAGS.discount_factor * (1-isfinal) * q2
        self.train_step(state1, target_q)

# Processing game state

def preprocess(img):
    return torch.from_numpy(resize(img, resolution).astype(np.float32))

def game_state(game):
    return preprocess(game.get_state().screen_buffer)

def init_vizdoom(config):
    game = DoomGame()
    game.load_config(config)
    # Set to true to see window during training
    game.set_window_visible(True)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    return game

def watch_episodes(game, model, actions):
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()
    for episode in range(FLAGS.episodes_to_watch):
        game.new_episode(f'episode-{episode}')
        while not game.is_episode_finished():
            state = game_state(game)
            state = state.reshape([1, 1, resolution[0], resolution[1]])
            a_idx = model.get_best_action(state.to(torch_device))
            game.set_action(actions[a_idx])
            for _ in range(frame_repeat):
                game.advance_action()
        sleep(1.0)
        score = game.get_total_reward()
        print(f'Total score: {score}')

# Agent performing actions

def find_eps(epoch):
    start, end = 1.0, 0.1
    const_epochs, decay_epochs = .1*FLAGS.epochs, .6*FLAGS.epochs
    if epoch < const_epochs:
        return start
    elif epoch > decay_epochs:
        return end
    # Apply linear decay between const and decay epochs
    progress = (epoch-const_epochs)/(decay_epochs-const_epochs)
    return start - progress * (start - end)

def perform_action(epoch, game, model, actions):
    state1 = game_state(game)
    if random() <= find_eps(epoch):
        action = torch.tensor(randint(0, len(actions) - 1)).long()
    else:
        state1 = state1.reshape([1, 1, *resolution])
        action = model.get_best_action(state1.to(torch_device))
    reward = game.make_action(actions[action], frame_repeat)

    if game.is_episode_finished():
        isfinal, state2 = 1., None
    else:
        isfinal = 0.
        state2 = game_state(game)

    model.memory.transition(state1, action, state2, isfinal, reward)
    model.learn_from_memory()

# Agent training

def train(game, model, actions):
    time_start = time()
    print("Saving the network weigths to:", FLAGS.save_path)
    for epoch in range(FLAGS.epochs):
        print(f'------------- Epoch {epoch+1}')
        episodes_finished = 0
        scores = np.array([])
        game.new_episode()
        for _ in trange(FLAGS.iters_per_epoch, leave=False):
            perform_action(epoch, game, model, actions)
            if game.is_episode_finished():
                score = game.get_total_reward()
                scores = np.append(scores, score)
                game.new_episode()
                episodes_finished += 1
        print(f'--- Completed {episodes_finished} episodes')
        print(f'--- Mean: {scores.mean():.1f} +/- {scores.std():.1f}')
        print("------ Testing...")
        test(FLAGS.test_episodes, game, model, actions)
        torch.save(model, FLAGS.save_path)
    print(f'Total elapsed time: {(time()-time_start)/60:.2f} minutes')

def test(iters, game, model, actions):
    scores = np.array([])
    for _ in trange(FLAGS.test_episodes, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state = game_state(game)
            state = state.reshape([1, 1, resolution[0], resolution[1]])
            a_idx = model.get_best_action(state.to(torch_device))
            game.make_action(actions[a_idx], frame_repeat)
        r = game.get_total_reward()
        scores = np.append(scores, r)
    print(f'--- Results: mean: {scores.mean():.1f} +/- {scores.std():.1f}')


def main(_):
    print("Cuda available ? : ", torch.cuda.is_available())
    game = init_vizdoom(FLAGS.config)

    n = game.get_available_buttons_size()
    actions = [list(a) for a in itertools.product([0, 1], repeat=n)]

    if FLAGS.load_model:
        print(f'Model loaded from: {FLAGS.save_path}')
        model = torch.load(FLAGS.save_path).to(torch_device)
    else:
        model = QNN(len(actions)).to(torch_device)

    print("Training started!")
    if not FLAGS.skip_training:
        train(game, model, actions)

    game.close()
    print("Training finished!")
    print("======================================")
    watch_episodes(game, model, actions)

if __name__ == '__main__':
######################################### Model, config, save and watch parameters ########################################
    # Set both to True to skip training and use the lastly trained model
    # flags.DEFINE_boolean('skip_training', False, 'Set to skip training')
    # flags.DEFINE_boolean('load_model', False, 'Load the trained model')
    flags.DEFINE_integer('episodes_to_watch', 20, 'Trained episodes to watch')
    flags.DEFINE_boolean('skip_training', False, 'Set to skip training')
    flags.DEFINE_boolean('load_model', False, 'Load the trained model')
    flags.DEFINE_string('save_path', 'saved_model_doom.pth','Path for the trained model')
    flags.DEFINE_string('config', defend_the_center_config, 'Path to the doom config file')

######################################## Training parameters ########################################
# 16/04/2024
    flags.DEFINE_integer('batch_size', 64, 'Batch size')
    flags.DEFINE_integer('replay_memory_size', 10000, 'Replay memory capacity')
    flags.DEFINE_integer('iters_per_epoch', 2000, 'Iterations per epoch')
    flags.DEFINE_integer('epochs', 20, 'Number of epochs')
    flags.DEFINE_integer('test_episodes', 100, 'Episodes to test with')
    flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate')
    flags.DEFINE_float('discount_factor', 0.99, 'Discount factor')
# 17/04/2024
    # flags.DEFINE_integer('batch_size', 64, 'Batch size')
    # flags.DEFINE_integer('replay_memory_size', 10000, 'Replay memory capacity')
    # flags.DEFINE_integer('iters_per_epoch', 1000, 'Iterations per epoch')
    # flags.DEFINE_integer('epochs', 10, 'Number of epochs')
    # flags.DEFINE_integer('test_episodes', 50, 'Episodes to test with')
    # flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
    # flags.DEFINE_float('discount_factor', 0.99, 'Discount factor')
# 17/04/2024 2
    # flags.DEFINE_integer('batch_size', 64, 'Batch size')
    # flags.DEFINE_integer('replay_memory_size', 30000, 'Replay memory capacity')
    # flags.DEFINE_integer('iters_per_epoch', 3000, 'Iterations per epoch')
    # flags.DEFINE_integer('epochs', 30, 'Number of epochs')
    # flags.DEFINE_integer('test_episodes', 100, 'Episodes to test with')
    # flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate')
    # flags.DEFINE_float('discount_factor', 0.99, 'Discount factor')

# 17 & 18/04/2024 : Tests on deadly_corridor
    # flags.DEFINE_integer('batch_size', 64, 'Batch size')
    # flags.DEFINE_integer('replay_memory_size', 100000, 'Replay memory capacity')
    # flags.DEFINE_integer('iters_per_epoch', 10000, 'Iterations per epoch')
    # flags.DEFINE_integer('epochs', 30, 'Number of epochs')
    # flags.DEFINE_integer('test_episodes', 100, 'Episodes to test with')
    # flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
    # flags.DEFINE_float('discount_factor', 0.99, 'Discount factor')

    # flags.DEFINE_integer('batch_size', 64, 'Batch size')
    # flags.DEFINE_integer('replay_memory_size', 10000, 'Replay memory capacity')
    # flags.DEFINE_integer('iters_per_epoch', 2000, 'Iterations per epoch')
    # flags.DEFINE_integer('epochs', 20, 'Number of epochs')
    # flags.DEFINE_integer('test_episodes', 100, 'Episodes to test with')
    # flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate')
    # flags.DEFINE_float('discount_factor', 0.99, 'Discount factor')

# 18/04/2024 : Tests on defend_the_center
    # flags.DEFINE_integer('batch_size', 64, 'Batch size')
    # flags.DEFINE_integer('replay_memory_size', 30000, 'Replay memory capacity')
    # flags.DEFINE_integer('iters_per_epoch', 3000, 'Iterations per epoch')
    # flags.DEFINE_integer('epochs', 30, 'Number of epochs')
    # flags.DEFINE_integer('test_episodes', 100, 'Episodes to test with')
    # flags.DEFINE_float('learning_rate', 0.000125, 'Learning rate')
    # flags.DEFINE_float('discount_factor', 0.99, 'Discount factor')

    app.run(main)
