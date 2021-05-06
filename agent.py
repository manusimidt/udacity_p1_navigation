"""
This file will hold the agent class.
The agent class implements the deep Q-Network algorithm explained here:
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
"""
import numpy as np
import random
from collections import namedtuple, deque

from typing import Tuple

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

"""
For some reason the pytorch function .to(device) takes forever to execute.
Probably some issue between my cuda version and the outdated pytorch version (0.4.0) that unityagents requires
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 465.19.01    Driver Version: 465.19.01    CUDA Version: 11.3     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:01:00.0  On |                  N/A |
|  0%   59C    P0    28W / 130W |    679MiB /  5941MiB |      2%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
"""
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')


class Agent:
    """ This class represents the reinforcement learning agent """

    def __init__(self, state_size: int, action_size: int, hidden_sizes: [int] = (64,),
                 gamma: float = 0.99, lr: float = 0.001, tau: float = 0.001,
                 buffer_size: int = 100000, batch_size: int = 64, update_rate: int = 5,
                 seed: int = int(random.random() * 100)):
        """
        Initializes the agent
        :param state_size: dimensions of a state
        :param action_size: dimension of a action
        :param hidden_sizes: array containing the size for each hidden layer of the deep Q network
        :param gamma: discount factor for learning
        :param lr: learning rate
        :param tau: #todo don't know what this does, something with soft update?
        :param buffer_size: size of the replay buffer (FIFO)
        :param batch_size: #todo don't know what this does..
        :param update_rate: # every nth step after which the networks will be updated
        :param seed: seed to get comparable model runs
        """
        random.seed(seed)
        self.state_site:int = state_size
        self.action_size:int = action_size
        self.hidden_sizes:[int] = hidden_sizes

        self.gamma: float = gamma
        self.tau: float = tau

        self.update_rate: int = update_rate
        self.batch_size: int = batch_size

        # initialize the deep Q-Network
        self.local_network: QNetwork = QNetwork(state_size, hidden_sizes, action_size, seed).to(device)
        self.target_network: QNetwork = QNetwork(state_size, hidden_sizes, action_size, seed).to(device)

        # initialize optimizer
        self.optimizer = optim.Adam(self.local_network.parameters(), lr=lr)

        # initialize replay memory
        self.memory: ReplayBuffer = ReplayBuffer(action_size, buffer_size, batch_size)

        # Used to determine when the agent starts learning
        self.t_step = 0


    def step(self, state: np.ndarray, action, reward, next_state, done) -> None:
        """
        todo: add types, add docs
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        # save experience to buffer
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_rate

        # at every nth step (n=self.update_rate) take random experiences from
        # buffer and learn from them
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self._learn(experiences)

    def act(self, state: np.ndarray, epsilon: float) -> int:
        """
        uses to policy to decide on an action given the state
        :return:
        """
        state = torch.from_numpy(state).float().to(device)
        self.local_network.eval()
        with torch.no_grad():
            action_values = self.local_network.forward(state)
        self.local_network.train()

        if random.random() > epsilon:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return random.choice(np.arange(self.action_size))

    def _learn(self, experiences: Tuple[torch.Tensor]):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_network.forward(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.local_network(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # soft update target network
        self.soft_update(self.local_network, self.target_network, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update target network

        θ_target = tau*θ_local + (1 - tau)*θ_target

        :param local_model:
        :param target_model:
        :param tau:
        :return:
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """ FiFo buffer storing experience tuples of the agent """

    def __init__(self, action_size: int, buffer_size: int, batch_size: int):
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
