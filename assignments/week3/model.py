import torch
from typing import Callable

import torch
import torch.nn as nn


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU(),
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()

        self.activation = activation
        self.initializer = initializer

        self.layers = nn.ModuleList()
        for i in range(hidden_count):
            self.layers += [nn.Linear(input_size, hidden_size)]
            # self.layers += [self.activation]
            input_size = hidden_size
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        # flattening
        # x = x.view(x.shape[0], -1)

        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x
