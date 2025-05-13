import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np


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
            batch_first=True,
        )

        # Prediction heads for two different classes
        self.predictor1 = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.predictor2 = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, graph_sequence):
        """
        Process a sequence of graphs and make predictions for nodes in the last graph.

        Args:
            graph_sequence (list): List of PyTorch Geometric Data objects

        Returns:
            tuple: Two torch.Tensor predictions for each node in the last graph (excluding the first node)
        """
        # Check if we have the expected number of graphs
        if len(graph_sequence) != 5:
            raise ValueError(
                f"Expected 5 graphs in sequence, but got {len(graph_sequence)}"
            )

        # Process each graph to get node embeddings
        sequence_embeddings = []
        for graph in graph_sequence:
            # Process the graph with GNN
            node_embeddings = self.graph_encoder(graph.x, graph.edge_index)
            sequence_embeddings.append(node_embeddings)

        # Get the number of nodes in the last graph
        last_graph_nodes = graph_sequence[-1].x.size(0)

        # Prepare LSTM input for each node in the last graph, skipping the first node
        lstm_inputs = []
        for node_idx in range(
            1, last_graph_nodes
        ):  # Start from 1 to skip the first node
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
            node_sequence = torch.stack(node_sequence, dim=0).unsqueeze(
                0
            )  # [1, seq_len, hidden_dim]
            lstm_inputs.append(node_sequence)

        # Process each node's temporal sequence through the LSTM
        node_predictions1 = []
        node_predictions2 = []
        for node_input in lstm_inputs:
            # Process through LSTM
            lstm_out, _ = self.lstm(node_input)

            # Take the last output and make predictions
            final_embedding = lstm_out[:, -1, :]  # [1, hidden_dim]
            prediction1 = self.predictor1(final_embedding)
            prediction2 = self.predictor2(final_embedding)
            node_predictions1.append(prediction1)
            node_predictions2.append(prediction2)

        # Stack all node predictions
        return torch.cat(node_predictions1, dim=0), torch.cat(
            node_predictions2, dim=0
        )  # [num_nodes-1, output_dim]