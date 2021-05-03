"""
This file will hold the agent class.
The agent class implements the deep Q-Network algorithm explained here:
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
"""
import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """ This class represents the reinforcement learning agent """

    def __init__(self, state_size: int, action_size: int, hidden_sizes: [int] = (64, 64),
                 gamma: float = 0.99, lr: float = 5e-3, tau: float = 1e-3,
                 eps_start: float = 1.0, eps_dec: float = .9995, eps_min: float = 0.01,
                 buffer_size: int = 1e5, batch_size: int = 64, update_rate: int = 5,
                 seed: int = int(random.random() * 100)):
        """
        Initializes the agent
        :param state_size: dimensions of a state
        :param action_size: dimension of a action
        :param hidden_sizes: array containing the size for each hidden layer of the deep Q network
        :param gamma: discount factor for learning
        :param lr: learning rate
        :param tau: #todo don't know what this does, something with soft update?
        :param eps_start: epsilon start value
        :param eps_dec: epsilon decay per episode
        :param eps_min: minimum value for epsilon (never stop exploring)
        :param buffer_size: size of the replay buffer (FIFO)
        :param batch_size: #todo don't know what this does..
        :param update_rate: # every nth step after which the networks will be updated
        :param seed: seed to get comparable model runs
        """
        random.seed(seed)
        # initialize the deep Q-Network
        self.local_network: QNetwork = QNetwork(state_size, hidden_sizes, action_size, seed).to(device)
        self.target_network: QNetwork = QNetwork(state_size, hidden_sizes, action_size, seed).to(device)

        # initialize optimizer
        self.optimizer = optim.Adam(self.local_network.parameters(), lr=lr)

        # initialize replay memory
        self.memory: ReplayBuffer = ReplayBuffer(action_size, buffer_size, batch_size)

        # counter increasing at each step
        self.iter_count = 0

    def reset(self):
        self.iter_count = 0
        self.memory.reset()

    def step(self, state: torch.Tensor, action, reward, next_state, done) -> None:
        """
        todo: add types, add docs
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        pass


class ReplayBuffer:
    """ FiFo buffer storing experience tuples of the agent """

    def __init__(self, action_size, buffer_size, batch_size):
        """
        Initialize Buffer
        :param action_size: dimension of each action
        :param buffer_size: maximum amount of experiences the buffer saves
        :param batch_size: size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def reset(self):
        self.memory.clear()

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
