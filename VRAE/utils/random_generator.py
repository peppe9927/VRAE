import numpy as np
import torch

def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    Z_mb = []
    for i in range(batch_size):
        # Inizializza un array di zeri con dimensione [max_seq_len, z_dim]
        temp = np.zeros([max_seq_len, z_dim])
        # Genera rumore per la parte attiva della sequenza
        temp_Z = np.random.uniform(0., 1, [T_mb[i].item() if isinstance(T_mb[i], torch.Tensor) else T_mb[i], z_dim])
        # Inserisci il rumore nel posto giusto
        temp[:T_mb[i], :] = temp_Z
        Z_mb.append(temp)
    # Converte la lista in un array NumPy e poi in un tensore PyTorch
    Z_mb = torch.tensor(np.array(Z_mb), dtype=torch.float32)
    return Z_mb