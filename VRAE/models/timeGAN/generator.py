import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Generator(nn.Module):
    def __init__(self, hidden_dim, num_layers, max_seq = 50, cell_type = 'GRU'):
        super(Generator,self).__init__()
        self.max_seq = max_seq
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        elif cell_type == 'RNN':
            self.rnn = nn.RNN(hidden_dim, hidden_dim, num_layers, batch_first=True)
        else:
            print('Invalid cell type, using LSTM')
            self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid())
        
    def forward(self, x, T):
        T = T.cpu()
        packed = pack_padded_sequence(x, T, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True,total_length=self.max_seq)
        E = self.fc(output)

        return E # return the generated data embeddings