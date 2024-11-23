import torch
import torch.nn as nn

# Encoder and decoder LSTM Parameter
input_size = 5     # Number of features
hidden_size = 16    # Number of hidden units in the LSTM: kimenet számossága(sequence_length*hidden_size)
num_layers = 2     # Number of LSTM layers: tehát hogy hány LSTM réteg van egymás után
sequence_length = 20   # Length of each input sequence 
batch_size = 1     # Number of sequences processed together

class EncLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(EncLSTM, self).__init__()
        self.enc_lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(1), hidden_size)  # Hidden state
        c0 = torch.zeros(num_layers, x.size(1), hidden_size)  # Cell state

        out, (h0, c0) =self.enc_lstm(x, (h0, c0))

        node_features = out[-1, :, :]

        return node_features

class DecLSTM(nn.Module):
    def __init__(self, gnn_output_size, hidden_size, num_layers, dropout=0.2):
        super(DecLSTM, self).__init__()
        self.gnn_output_size = gnn_output_size
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.dec_lstm = nn.LSTM(gnn_output_size, hidden_size, num_layers)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, ):
        h0 = torch.zeros(num_layers, x.size(1), hidden_size)  # Hidden state
        c0 = torch.zeros(num_layers, x.size(1), hidden_size)  # Cell state

        out, (h0, c0) =self.dec_lstm(x, (h0, c0))

        out_flatten = out.view(x.size(1), -1)

        return out_flatten

