import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils import spectral_norm

class Embedder(nn.Module):
    """
    Embedding network that maps original time-series features into a latent space.
    
    Args:
      - input_dim: dimensione degli input (feature per ogni timestep)
      - hidden_dim: dimensione dello spazio latente
      - num_layers: numero di strati dell'LSTM
    """
    def __init__(self, input_dim, hidden_dim, num_layers):
        
        super(Embedder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = spectral_norm(nn.Linear(hidden_dim, hidden_dim))  # Spectral Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, X, T=None):
        """
        Args:
          - X: tensor di input shape [batch_size, seq_len, input_dim]
          - T: lunghezze effettive delle sequenze (lista o tensor di interi), opzionale.
               Le sequenze sono già padded con 0.0 dove necessario.
        
        Returns:
          - H: embeddings, tensor shape [batch_size, seq_len, hidden_dim]
        """
        if T is not None:
            # Se T è fornito, usiamo il packing per gestire sequenze di lunghezza variabile.
            if isinstance(T, torch.Tensor):
                T = T.cpu().numpy().tolist()
            packed_X = rnn_utils.pack_padded_sequence(X, T, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.rnn(packed_X)
            # pad_packed_sequence ricrea il batch con padding_value=0.0 (default) esplicito
            outputs, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True, padding_value=0.0)
        else:
            outputs, _ = self.rnn(X)
        # Applichiamo la fully connected layer con attivazione sigmoide ad ogni timestep
        H = torch.sigmoid(self.layer_norm(self.fc(outputs)))
        return H

class Recovery(nn.Module):
    """
    Recovery network that maps the latent representation back to the original space.
    
    Args:
      - hidden_dim: dimensione dello spazio latente
      - output_dim: dimensione dello spazio originale (feature per timestep)
      - num_layers: numero di strati dell'LSTM
    """
    def __init__(self, hidden_dim, output_dim, num_layers):
        super(Recovery, self).__init__()
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = spectral_norm(nn.Linear(hidden_dim, output_dim))  # Spectral Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)  # Layer normalization aggiunta
        
    def forward(self, H, T=None):
        """
        Args:
          - H: rappresentazione latente, tensor shape [batch_size, seq_len, hidden_dim]
          - T: lunghezze effettive delle sequenze, opzionale.
               Le sequenze sono già padded con 0.0 dove necessario.
        
        Returns:
          - X_tilde: dati recuperati, tensor shape [batch_size, seq_len, output_dim]
        """
        if T is not None:
            if isinstance(T, torch.Tensor):
                T = T.cpu().numpy().tolist()
            packed_H = rnn_utils.pack_padded_sequence(H, T, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.rnn(packed_H)
            outputs, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True, padding_value=0.0)
        else:
            outputs, _ = self.rnn(H)
        # Applichiamo layer norm prima della fully connected
        normalized_outputs = self.layer_norm(outputs)
        X_tilde = torch.sigmoid(self.fc(normalized_outputs))
        return X_tilde

# Classe Opzionale: RNN con Weight Normalization
class WeightNormLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        super(WeightNormLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Creare LSTM standard
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )
        
        # Applicare weight normalization ai pesi dell'LSTM
        # Nota: questa è una semplificazione, in pratica sarebbe necessario
        # implementare una versione custom di weight normalization per LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                # Non possiamo usare direttamente spectral_norm su parametri LSTM
                # Questo è un approccio semplificato (normalizzazione L2)
                param.data = param.data / torch.norm(param.data)
    
    def forward(self, x, hx=None):
        return self.lstm(x, hx)