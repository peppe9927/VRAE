import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import sys

sys.path.append('VRAE/VRAE')
from utils import to_onehot


class ConditionalVariationalEncoder(nn.Module):
    def __init__(self,
                 num_classes: int, 
                 sensor_dim: int, 
                 num_routes: int,
                 route_emb_dim: int,
                 num_hidden_lstm: int,
                 num_hidden: int,
                 num_latent: int,
                 device):
        """
        Args:
            num_classes (int): Number of classes for the condition.
            sensor_dim (int): Dimension of sensor data for each timestep.
            num_routes (int): Number of possible route categories.
            route_emb_dim (int): Dimension of the embedding for routes.
            num_hidden_lstm (int): Dimension of the hidden state in the LSTM.
            num_hidden (int): Dimension of the hidden state after merging branches.
            num_latent (int): Dimension of the latent space.
        """
        self.num_classes = num_classes

        super(ConditionalVariationalEncoder, self).__init__()
        
        # Embedding for routes, takes a tensor of shape (batch, seq_len) and returns a tensor of shape (batch, seq_len, route_emb_dim)
        self.route_embedding = nn.Embedding(num_routes, route_emb_dim, padding_idx=0) 

        # Branch for sensor data: using an LSTM
        self.sensor_lstm = nn.LSTM(sensor_dim + route_emb_dim, num_hidden_lstm, batch_first=True)
                
        # Merging the two branches: the concatenated vector will have dimension num_hidden
        self.linear_layer = nn.Linear(num_hidden_lstm + num_classes, num_hidden) # adding num_classes for the condition
        
        # Calculating the parameters of the latent space
        self.z_mean = nn.Linear(num_hidden, num_latent) 
        self.z_log_var = nn.Linear(num_hidden, num_latent)
    
    def reparameterize(self, z_mean, z_log_var):
        """
        Samples z in a differentiable way using the reparameterization trick.

        Args:
            z_mean (Tensor): Mean of the latent variable.
            z_log_var (Tensor): Log variance of the latent variable.

        Returns:
            Tensor: Sampled latent variable.
        """
        eps = torch.randn_like(z_mean)
        z = z_mean + eps * torch.exp(z_log_var / 2)
        return z

    def forward(self, routes, times, lengths, labels, device):
        """
        Forward pass through the encoder.

        Args:
            routes (Tensor): Tensor of shape (batch, seq_len) containing route indices.
            times (Tensor): Tensor of shape (batch, seq_len, sensor_dim) containing sensor data.
            lengths (Tensor): Tensor containing the lengths of each sequence in the batch.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: z_mean, z_log_var, z
        """
        # Process the route branch
        route_emb = self.route_embedding(routes)  # compute the embedding sequence
        
        combined = torch.cat([times, route_emb], dim=-1)  # concatenate the sequences
        
        packed_input = pack_padded_sequence(combined, lengths, batch_first=True)

        sensor_out, (h_sensor, _) = self.sensor_lstm(packed_input) # pass to an LSTM
        h_final = h_sensor[-1]  # shape: [batch, num_hidden_lstm]
        
        onehot_targets = to_onehot(labels, self.num_classes, device) # convert the labels to one-hot encoding
        
        h_final = torch.cat([h_final, onehot_targets], dim=-1) # adding the condition to theZ
        
        # Concatenate the representations
        output = F.leaky_relu(self.linear_layer(h_final))
        
        # Calculate the latent parameters
        z_mean = self.z_mean(output) # shape: [batch, num_latent]
        z_log_var = self.z_log_var(output) # shape: [batch, num_latent]
        z = self.reparameterize(z_mean, z_log_var) # compute a sample from the latent space
        
        return z_mean, z_log_var, z