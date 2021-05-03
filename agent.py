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


class Agent():
    """ This class represents the reinforcement learning agent """

    def __init__(self, state_size: int, action_size: int, hidden_sizes: [int] = (64, 64),
                 gamma: float = 0.99, lr: float = 5e-3, tau: float = 1e-3,
                 eps_start: float = 1.0, eps_dec: float = .9995, eps_min: float = 0.01,
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
        :param seed: seed to get comparable model runs
        """
        pass
