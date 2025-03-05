import torch
import torch.nn as nn

class Recover(nn.Module):
    def __init__(self, latent_size, output_size, hidden_size, num_layers=1):
        super(Recover, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Definisci la rete ricorrente (RNN)
        self.rnn = nn.RNN(latent_size, hidden_size, num_layers, batch_first=True)
        
        # Definisci un livello fully connected per riportare allo spazio originale
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Inizializza lo stato nascosto
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Passa l'input attraverso la RNN
        out, hn = self.rnn(x, h0)
        
        # Passa l'output della RNN attraverso il livello fully connected
        out = self.fc(out)