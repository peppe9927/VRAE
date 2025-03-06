# python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoRegressiveDecoder(nn.Module):
    def __init__(self, sensor_dim, num_routes, num_hidden, num_latent):
        """
        Parametri:
          sensor_dim : dimensione dei dati sensor per ogni timestep.
          num_routes : numero di categorie per le route.
          num_hidden : dimensione dello stato nascosto della LSTM decoder.
          num_latent : dimensione dello spazio latente.
        """
        super(AutoRegressiveDecoder, self).__init__()
        
        # Decoder LSTM: prende in input un vettore latente (differenziato ad ogni step grazie all'autoregressione)
        self.decoder_lstm = nn.LSTM(num_latent, num_hidden, batch_first=True)
        
        # Output per ricostruire i dati sensor
        self.sensor_out = nn.Linear(num_hidden, sensor_dim)
        # Output per ricostruire le route (logits, da trasformare con softmax)
        self.route_out = nn.Linear(num_hidden, num_routes)
        
        # Token di partenza appreso (start token) per l'autoregressione
        self.start_input = nn.Parameter(torch.randn(num_latent))
        # Proiezione per mappare l'output sensor all'ingresso latente per il passo successivo
        self.out2latent = nn.Linear(sensor_dim, num_latent)

    def forward(self, z, seq_len):
        """
        Decodifica "parallela" (non autoregressiva) come implementato precedentemente.
        """
        # Espandi z lungo la dimensione temporale
        z_expanded = z.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, num_latent)
        decoder_out, _ = self.decoder_lstm(z_expanded)       # (batch, seq_len, num_hidden)
        
        sensor_recon = self.sensor_out(decoder_out)
        route_logits = self.route_out(decoder_out)
        route_recon = F.softmax(route_logits, dim=-1)
        return sensor_recon, route_recon

    def autoregressive_decode(self, z, seq_lens):
        """
        Autoregressive decoding:
          z      : vettore latente di forma (batch, num_latent)
          seq_lens: lista o tensore con la lunghezza reale per ogni esempio del batch (batch,)
        
        Restituisce:
          sensor_outputs: lista di tensori con output sensor validi per ogni esempio
          route_outputs : lista di tensori con output route validi per ogni esempio
        """
        batch_size = z.size(0)
        max_seq_len = max(seq_lens) if not torch.is_tensor(seq_lens) else int(seq_lens.max().item())
        device = z.device

        # Inizializza lo stato nascosto dell'LSTM
        h = torch.zeros(1, batch_size, self.decoder_lstm.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.decoder_lstm.hidden_size, device=device)
        
        # Token di partenza: estendi il parametro a batch size
        input_t = self.start_input.unsqueeze(0).expand(batch_size, 1, -1)  # (batch, 1, num_latent)
        
        all_sensor_outputs = []
        all_route_outputs = []
        
        for t in range(max_seq_len):
            # Combina l'input corrente con z per condizionare la generazione
            current_input = input_t + z.unsqueeze(1)  # (batch, 1, num_latent)
            
            # Esegue un passo LSTM
            out, (h, c) = self.decoder_lstm(current_input, (h, c))  # out: (batch, 1, num_hidden)
            
            # Genera gli output per questo timestep
            sensor_t = self.sensor_out(out)  # (batch, 1, sensor_dim)
            route_logits = self.route_out(out)
            route_t = F.softmax(route_logits, dim=-1)  # (batch, 1, num_routes)
            
            all_sensor_outputs.append(sensor_t)
            all_route_outputs.append(route_t)
            
            # Calcola il nuovo input trasformando l'output sensor corrente
            input_t = self.out2latent(sensor_t)  # (batch, 1, num_latent)
        
        # Concatena lungo la dimensione temporale
        sensor_outputs = torch.cat(all_sensor_outputs, dim=1)  # (batch, max_seq_len, sensor_dim)
        route_outputs = torch.cat(all_route_outputs, dim=1)      # (batch, max_seq_len, num_routes)
        
        # Raccogli gli output in base alle lunghezze reali
        sensor_results = []
        route_results = []
        for i, seq_len in enumerate(seq_lens):
            seq_len = int(seq_len)  # assicura tipo int
            sensor_results.append(sensor_outputs[i, :seq_len, :])
            route_results.append(route_outputs[i, :seq_len, :])
        
        return sensor_results, route_results