import torch
from torch import nn

class FeedForwardNN(nn.Module):
    """
    Simple Feed-Forward Neural Network
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()

        # Define layers
        self.layer_stack = nn.Sequential(
                nn.Linear(input_size, hidden_size),  # Linear function - fully connected
                nn.ReLU(),  # Non-linearity
                nn.Linear(hidden_size, output_size),  # Linear function (readout) - fully connected
                nn.Sigmoid()  # Force the outputs between [0,1)
            )

    def forward(self, x):
        # Define forward pass
        out = layer_stack(x)

        return out
