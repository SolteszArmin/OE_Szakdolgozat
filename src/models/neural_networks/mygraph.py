import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv  # Graph Convolutional Network Layer

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes the GNN model.
        
        Parameters:
        - input_dim: Dimension of input features (RNN output for each node).
        - hidden_dim: Dimension of hidden layers in the GNN.
        - output_dim: Dimension of output layer (number of classes for classification).
        """
        super(GNN, self).__init__()
        
        # First graph convolutional layer
        self.conv1 = GCNConv(input_dim, hidden_dim)
        
        # Second graph convolutional layer
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Linear layer for classification
        self.classifier = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, edge_index, ego_index):
        """
        Forward pass through the GNN.
        
        Parameters:
        - x: Node feature matrix of shape [num_nodes, input_dim].
             Each row is a feature vector for a node (output from RNN).
        - edge_index: Edge list of shape [2, num_edges].
                      Defines the connectivity (graph structure).
        - ego_index: Index of the ego-vehicle node (single integer).
        
        Returns:
        - logits: Output logits for classification of ego-vehicle maneuver.
        """
        # First graph convolution + activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Second graph convolution + activation
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Extract embedding for the ego-vehicle node
        ego_embedding = x[ego_index]
        
        # Classification layer
        logits = self.classifier(ego_embedding)
        
        return logits