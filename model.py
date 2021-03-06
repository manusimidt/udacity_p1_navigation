"""
This file will hold the neural network responsible for estimating the action values for a given state
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """ This class represents the neural network the agent uses for estimating action values """

    def __init__(self, input_size: int, hidden_sizes: [int], output_size: int, seed: int) -> None:
        """
        Initializes the layers of the network
        :param input_size: size of the first layer
        :param hidden_sizes: array containing the size for each hidden layer
        :param output_size: size of the output layer
        :param seed: seed for the model
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)

        # self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        # self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        # self.fc3 = nn.Linear(hidden_sizes[1], output_size)

        self.layers = nn.ModuleList()
        # add input layer
        self.layers.append(nn.Linear(in_features=input_size, out_features=hidden_sizes[0]))
        # add hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(in_features=hidden_sizes[i], out_features=hidden_sizes[i + 1]))
        # add output layer
        self.layers.append(nn.Linear(in_features=hidden_sizes[-1], out_features=output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Do a forward pass through the network
        :param x: data to pass through the network
        :return:
        """
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # return self.fc3(x)
        x: torch.Tensor
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
