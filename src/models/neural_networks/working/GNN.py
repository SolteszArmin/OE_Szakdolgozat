import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool,global_max_pool,global_add_pool
from torch.nn import BatchNorm1d
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


class GATv2SequenceModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, out_channels, heads=1):
        super(GATv2SequenceModel, self).__init__()

        # GATv2 Layers
        self.gat1 = GATv2Conv(
            in_channels, hidden_channels, heads=heads, edge_dim=edge_dim
        )
        self.gat2 = GATv2Conv(
            hidden_channels * heads, 32, heads=heads, edge_dim=edge_dim
        )

        self.bn1 = BatchNorm1d(hidden_channels * heads)
        self.bn2 = BatchNorm1d(32*heads)

        # GRU for Sequence Processing
        # self.gru = torch.nn.GRU(
        #     hidden_channels,
        #     hidden_channels,
        #     batch_first=True,
        #     bidirectional=True,
        # )
        self.lstm = torch.nn.LSTM(
            32*heads, hidden_channels, batch_first=True, bidirectional=False
        )

        # Fully Connected Layers for Predictions
        self.fc = torch.nn.Linear(
            hidden_channels, out_channels
        )  # Bidirectional GRU doubles hidden size
        self.fc_bools = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index, edge_attr, batch, sequence_lengths):
        # GATv2 Layers
        x = self.gat1(x, edge_index, edge_attr)
        x = self.bn1(x)  # Apply BatchNorm
        x = F.leaky_relu(x, 0.01)
        x = self.gat2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.leaky_relu(x, 0.01)

        # Aggregate node features into graph embeddings
        graph_embeddings = global_add_pool(
            x, batch
        )  # [num_graphs_in_batch, hidden_channels]

        sequence_lengths
        # Reshape into sequences (batch_size x max_seq_len x hidden_channels)
        graph_sequence = pad_sequence(
            graph_embeddings.split(sequence_lengths), batch_first=True
        )

        # Pass through GRU
        packed_sequences = pack_padded_sequence(
            graph_sequence, sequence_lengths, batch_first=True, enforce_sorted=False
        )
        # packed_output, h_n = self.gru(packed_sequences)
        packed_output, (h_n, c_n) = self.lstm(packed_sequences)

        # Unpack if you want the full GRU output (optional)
        # unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Final output layer (use last GRU state for prediction)
        out = self.fc(
            h_n[-1]
        )  # Use the final GRU state (hidden state of last time step)
        out_bool = self.fc_bools(h_n[-1])
        return out, out_bool
