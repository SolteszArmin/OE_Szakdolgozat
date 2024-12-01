import torch
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, global_mean_pool,global_max_pool

class GATv2SequenceModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, out_channels, heads=1):
        super(GATv2SequenceModel, self).__init__()

        # GATv2 Layers
        self.gat1 = GATv2Conv(
            in_channels, hidden_channels, heads=heads, edge_dim=edge_dim
        )
        self.gat2 = GATv2Conv(
            hidden_channels * heads, hidden_channels, heads=heads, edge_dim=edge_dim
        )

        # GRU for Sequence Processing
        self.gru = torch.nn.GRU(
            hidden_channels * heads,
            hidden_channels,
            batch_first=True,
            bidirectional=True,
        )

        # Fully Connected Layer for Predictions
        self.fc = torch.nn.Linear(hidden_channels, out_channels)
        self.fc_bools=torch.nn.Linear(hidden_channels,2)

    def forward(self, data_list):
        graph_embeddings = []

        for data in data_list:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

            # GATv2 Layers
            x = F.leaky_relu(self.gat1(x, edge_index, edge_attr), 0.01)
            x = F.dropout(x, p=0.2, training=self.training)
            x = F.leaky_relu(self.gat2(x, edge_index, edge_attr), 0.01)

            # Aggregate node features into graph embeddings
            graph_embedding = global_mean_pool(
                x, data.batch
            )  # [batch_size, hidden_channels]
            graph_embeddings.append(graph_embedding)
        # Stack graph embeddings into a sequence tensor
        graph_sequence = torch.stack(
            graph_embeddings, dim=1
        )  # [batch_size, seq_len, hidden_channels]

        # Pass through GRU
        _, h_n = self.gru(graph_sequence)

        # Final output layer
        out = self.fc(h_n[-1])  # Use the final GRU state
        out_bool=self.fc_bools(h_n[-1])
        return out, out_bool