import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Discriminator(nn.Module):
    def __init__(self, num_hidden, num_layers, max_seq, cell_type = 'GRU'):
        super(Discriminator,self).__init__()
        self.max_seq = max_seq
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(num_hidden, num_hidden, num_layers, batch_first=True, bidirectional=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(num_hidden, num_hidden, num_layers, batch_first=True, bidirectional=True)
        elif cell_type == 'RNN':
            self.rnn = nn.RNN(num_hidden, num_hidden, num_layers, batch_first=True, bidirectional=True)
        else:
            print('Invalid cell type, using default GRU')
            self.rnn = nn.GRU(num_hidden, num_hidden, num_layers, batch_first=True, bidirectional=True)
        
        # Output size is doubled due to bidirectional RNN
        
        self.fc = nn.Sequential(
            nn.Linear(num_hidden*2, 1),
            nn.Sigmoid())
    
    def forward(self, x, T):
        T = T.cpu()
        packed = pack_padded_sequence(x, T, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length= self.max_seq)
        Y_hat = self.fc(output)

        return Y_hat