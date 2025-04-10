import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np

def create_pytorch_graph(num_nodes, num_node_attrs):
    """
    Creates a PyTorch Geometric graph with random node features and edge connections.
    
    Args:
        num_nodes (int): Number of nodes in the graph
        num_node_attrs (int): Number of attributes/features per node
        
    Returns:
        torch_geometric.data.Data: A PyTorch Geometric graph object
    """
    # Create random node features
    x = torch.randn(num_nodes, num_node_attrs)
    
    # Create random edge connections (fully connected graph for simplicity)
    # For a more realistic graph, you might want to customize this part
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # Avoid self-loops
                edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Create random edge attributes (optional)
    edge_attr = torch.randn(edge_index.size(1), 1)
    
    # Create the graph
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return graph

class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder that processes individual graphs.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        # First GNN layer with ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Second GNN layer
        x = self.conv2(x, edge_index)
        
        return x

class TemporalGraphModel(nn.Module):
    """
    A model that processes a sequence of graphs and makes predictions for each node in the last graph.
    """
    def __init__(self, node_features, hidden_dim, lstm_hidden_dim, output_dim):
        super(TemporalGraphModel, self).__init__()
        
        # GNN for processing individual graphs
        self.graph_encoder = GNNEncoder(node_features, hidden_dim, hidden_dim)
        
        # LSTM for processing the sequence of graph embeddings
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Prediction head for node-level predictions
        self.predictor = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, graph_sequence):
        """
        Process a sequence of graphs and make predictions for nodes in the last graph.
        
        Args:
            graph_sequence (list): List of PyTorch Geometric Data objects
            
        Returns:
            torch.Tensor: Predictions for each node in the last graph
        """
        # Check if we have the expected number of graphs
        if len(graph_sequence) != 5:
            raise ValueError(f"Expected 5 graphs in sequence, but got {len(graph_sequence)}")
        
        # Process each graph to get node embeddings
        sequence_embeddings = []
        for graph in graph_sequence:
            # Process the graph with GNN
            node_embeddings = self.graph_encoder(graph.x, graph.edge_index)
            sequence_embeddings.append(node_embeddings)
        
        # Get the number of nodes in the last graph
        last_graph_nodes = graph_sequence[-1].x.size(0)
        
        # Prepare LSTM input for each node in the last graph
        lstm_inputs = []
        for node_idx in range(last_graph_nodes):
            # For each node in the last graph, collect its features across time
            node_sequence = []
            for t in range(len(sequence_embeddings)):
                # Handle the case where earlier graphs might have fewer nodes
                if node_idx < sequence_embeddings[t].size(0):
                    node_sequence.append(sequence_embeddings[t][node_idx])
                else:
                    # If the node doesn't exist in earlier graphs, use zero embedding
                    node_sequence.append(torch.zeros_like(sequence_embeddings[-1][0]))
            
            # Stack the node's embeddings across time
            node_sequence = torch.stack(node_sequence, dim=0).unsqueeze(0)  # [1, seq_len, hidden_dim]
            lstm_inputs.append(node_sequence)
        
        # Process each node's temporal sequence through the LSTM
        node_predictions = []
        for node_input in lstm_inputs:
            # Process through LSTM
            lstm_out, _ = self.lstm(node_input)
            
            # Take the last output and make predictions
            final_embedding = lstm_out[:, -1, :]  # [1, hidden_dim]
            prediction = self.predictor(final_embedding)
            node_predictions.append(prediction)
        
        # Stack all node predictions
        return torch.cat(node_predictions, dim=0)  # [num_nodes, output_dim]

def process_graph_sequence(graph_list):
    """
    Process a sequence of 5 graphs and make predictions for nodes in the last graph.
    
    Args:
        graph_list (list): A list of 5 PyTorch Geometric graphs
        
    Returns:
        torch.Tensor: Predictions for each node in the last graph
    """
    # Initialize the model
    model = TemporalGraphModel(
        node_features=13,  # Number of features per node
        hidden_dim=64,     # Hidden dimension for GNN
        lstm_hidden_dim=128, # Hidden dimension for LSTM
        output_dim=1       # Output dimension (e.g., 1 for scalar prediction)
    )
    
    # Forward pass
    predictions = model(graph_list)
    
    return predictions

if __name__ == "__main__":
    # Example usage
    graphs = [
        create_pytorch_graph(num_nodes=3, num_node_attrs=13),
        create_pytorch_graph(num_nodes=5, num_node_attrs=13),
        create_pytorch_graph(num_nodes=4, num_node_attrs=13),
        create_pytorch_graph(num_nodes=6, num_node_attrs=13),
        create_pytorch_graph(num_nodes=4, num_node_attrs=13)
    ]

    # Get predictions
    predictions = process_graph_sequence(graphs)
    print(f"Predictions shape: {predictions.shape}")  # Should be [num_nodes_last_graph, output_dim]
    print(predictions) 