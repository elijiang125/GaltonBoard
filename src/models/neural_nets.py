import torch

class FeedForwardNN(torch.nn.Module):
    """
    Simple Feed-Forward Neural Network
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNN, self).__init__()

        # Define layers
        self.fc1 = torch.nn.Linear(input_size, hidden_size) # Linear function - fully connected
        self.relu = torch.nn.ReLU()  # Non-linearity
        self.fc2 = torch.nn.Linear(hidden_size, output_size)  # Linear function (readout) - fully connected

    def forward(self, x):
        # Define forward pass
        out = self.fc1(x)  # Pass input through first layer
        out = self.relu(out)  # Apply ReLU activation
        out = self.fc2(out)  # Pass through second layer to get output

        return out
