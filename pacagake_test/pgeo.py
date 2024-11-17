import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv  # Graph Convolutional Network
from torch_geometric.data import Data  # PyTorch Geometric Data Object
import torch_geometric.transforms as T
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Artificial data generation for traffic scenarios
def generate_artificial_data(num_agents=5, num_timesteps=20):
    """ Generate artificial graph data for traffic agents over timesteps """
    # Randomly generate features for each agent: [x, y, vx, vy] (position and velocity)
    agent_features = [np.random.rand(num_agents, 4) for _ in range(num_timesteps)]
    
    # Adjacency matrix: fully connected graph (all agents interact with each other)
    adj_matrix = np.ones((num_agents, num_agents)) - np.eye(num_agents)
    
    # Edge indices for the fully connected graph (PyTorch Geometric format)
    edge_index = torch.tensor(np.nonzero(adj_matrix), dtype=torch.long)
    
    return agent_features, edge_index

# Define GNN-RNN Architecture (simplified version)
class GNNRNNModel(nn.Module):
    def __init__(self, in_feats=4, hidden_gnn_dim=16, hidden_rnn_dim=32, num_classes=8):
        super(GNNRNNModel, self).__init__()
        
        # GNN Layers (Graph Convolutional Network)
        self.gnn1 = GCNConv(in_feats, hidden_gnn_dim)
        self.gnn2 = GCNConv(hidden_gnn_dim, hidden_gnn_dim)
        
        # RNN Layer (LSTM to capture temporal dynamics)
        self.rnn = nn.LSTM(hidden_gnn_dim, hidden_rnn_dim, batch_first=True)
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_rnn_dim, num_classes)
    
    def forward(self, node_features_list, edge_index):
        """
        node_features_list: A list of node features over timesteps [T x num_agents x feat_dim]
        edge_index: Edge indices (connectivity) in the graph.
        """
        # Process each timestep independently through the GNN
        gnn_outputs = []
        for t_features in node_features_list:
            x = torch.tensor(t_features, dtype=torch.float32)
            x = self.gnn1(x, edge_index)
            x = torch.relu(x)
            x = self.gnn2(x, edge_index)
            gnn_outputs.append(x.unsqueeze(0))  # Add time dimension
        
        # Stack all GNN outputs as a sequence [batch_size=1, timesteps, num_agents, hidden_gnn_dim]
        gnn_sequence = torch.cat(gnn_outputs, dim=0).transpose(0, 1)  # [T x num_agents x hidden_gnn_dim]
        
        # Process the sequence through the RNN (LSTM expects input [batch_size, seq_len, input_size])
        rnn_out, _ = self.rnn(gnn_sequence)  # Output shape: [num_agents x T x hidden_rnn_dim]
        
        # We are interested in predicting the ego-vehicle's maneuver, so we take the first agent's output
        ego_output = rnn_out[0][-1]  # Taking the last timestep output for ego vehicle
        
        # Classify the maneuver at the last timestep
        out = self.classifier(ego_output)
        
        return out

# Simulating Data for a Traffic Scenario
num_agents = 6  # Including ego vehicle and surrounding agents
num_timesteps = 20  # Sequence length
agent_features_list, edge_index = generate_artificial_data(num_agents=num_agents, num_timesteps=num_timesteps)

# Model Initialization
num_classes = 8  # Number of maneuver classes (as per research paper)
model = GNNRNNModel(in_feats=4, hidden_gnn_dim=16, hidden_rnn_dim=32, num_classes=num_classes)

# Dummy labels for training (random maneuvers between 0 and 7)
labels = torch.randint(0, num_classes, (1,), dtype=torch.long)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop (for demonstration purposes only)
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    output = model(agent_features_list, edge_index)
    
    # Compute loss
    loss = criterion(output.unsqueeze(0), labels)
    
    # Backpropagation and optimization step
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')

print("Training Completed!")

# Inference example (after training):
model.eval()
with torch.no_grad():
    predicted_maneuver = model(agent_features_list, edge_index).argmax(dim=0).item()
    print(f"Predicted maneuver class: {predicted_maneuver}")