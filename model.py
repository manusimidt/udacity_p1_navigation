"""
This file will hold the neural network responsible for estimating the action values for a given state
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """ This class represents the neural network the agent uses for estimating action values """

    def __init__(self, input_size: int, hidden_sizes: [int], output_size: int) -> None:
        """
        Initializes the layers of the network
        :param input_size:
        :param hidden_sizes:
        :param output_size:
        """
