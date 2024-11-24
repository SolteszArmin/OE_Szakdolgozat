import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, LayerNorm


class GNNLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        GNN layer for spatial feature extraction.

        Parameters:
        - input_dim: The dimensionality of the input node features.
        - hidden_dim: The dimensionality of the hidden GNN layers.
        """
        super(GNNLayer, self).__init__()
        
        # Define three GCNConv layers
        self.gnn1 = GCNConv(input_dim, hidden_dim)
        self.norm1 = LayerNorm(hidden_dim)
        
        self.gnn2 = GCNConv(hidden_dim, hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        
        self.gnn3 = GCNConv(hidden_dim, hidden_dim)
        self.norm3 = LayerNorm(hidden_dim)

    def forward(self, x, edge_index):
        """
        Forward pass through the GNN layer.

        Parameters:
        - x: Node feature matrix of shape [num_nodes, input_dim].
        - edge_index: Edge list defining graph connectivity (shape [2, num_edges]).

        Returns:
        - x: Updated node embeddings of shape [num_nodes, hidden_dim].
        """
        # First GNN layer
        x = self.gnn1(x, edge_index)
        x = self.norm1(x)
        x = F.leaky_relu(x, negative_slope=0.1)

        # Second GNN layer
        x = self.gnn2(x, edge_index)
        x = self.norm2(x)
        x = F.leaky_relu(x, negative_slope=0.1)

        # Third GNN layer
        x = self.gnn3(x, edge_index)
        x = self.norm3(x)
        x = F.leaky_relu(x, negative_slope=0.1)

        return x