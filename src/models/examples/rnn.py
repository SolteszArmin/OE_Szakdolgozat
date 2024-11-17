import torch
import torch.nn as nn

# Define parameters
input_size = 3     # Number of features per time step
hidden_size = 5    # Number of hidden units in the LSTM
num_layers = 1     # Number of LSTM layers
sequence_length = 5   # Length of each input sequence
batch_size = 1     # Number of sequences processed together

# Example input: Random data to simulate a batch of one sequence with 5 time steps and 3 features per time step
input_data = torch.randn(batch_size, sequence_length, input_size)

# Define an LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SimpleLSTM, self).__init__()
        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer (output layer)
        self.fc = nn.Linear(hidden_size, 1)  # Outputting one value per sequence

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(num_layers, x.size(0), hidden_size)  # Hidden state
        c0 = torch.zeros(num_layers, x.size(0), hidden_size)  # Cell state
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: [batch_size, sequence_length, hidden_size]
        
        # Apply fully connected layer to the output from the last time step of each sequence
        out = self.fc(out[:, -1, :])   # Only take the output from the last time step
        
        return out

# Instantiate the model
model = SimpleLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

# Perform a forward pass with example input data
output = model(input_data)

print("Input shape:", input_data.shape)      # [batch_size, sequence_length, input_size]
print("Output shape:", output.shape)         # [batch_size, 1]