
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# Node feature matrix (3 nodes, 4 features per node)
X = torch.tensor([[0.5, 1.0, 0.1, 0.2],  # Node 0 features: [x, y, v_x, v_y]
                  [1.0, 0.5, -0.1, -0.2], # Node 1 features: [x, y, v_x, v_y]
                  [1.5, 1.5, 0.2, -0.1]]) # Node 2 features: [x, y, v_x, v_y]
# Edge index matrix: 2 edges (directed graph)

edge_index = torch.tensor([[0, 1],   # Edge from Node 0 to Node 1
                           [1, 2]])  # Edge from Node 1 to Node 2

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # First GCN layer
        x = F.relu(self.conv1(x, edge_index))
        # Second GCN layer
        x = self.conv2(x, edge_index)
        return x

# Create GNN model
gnn_model = GNN(input_dim=4, hidden_dim=8, output_dim=8)

# Forward pass through the GNN
gnn_output = gnn_model(X, edge_index)
print(gnn_output)