import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from VRAE.utils import custom_collate_fn
from VRAE.models import VRAE, Encoder

dataset=torch.load('train_dataset_samples.pt')
torch.manual_seed(42)
train_loader = DataLoader(dataset, batch_size=40, shuffle=True, collate_fn=custom_collate_fn)


# 
BATCH_SIZE = 32 # dimensione del batch
EPOCHS = 10 # numero di epoch di allenamento
LEARNING_RATE = 1e-3 # lr dell'optimizer

SENSOR_DIM = 1       # dimensione dei dati temporali per ogni time-step 
NUM_ROUTES = 32        # numero di route possibili
ROUTE_EMB_DIM = 4     # dimensione dell'embedding per le route
NUM_HIDDEN = 64       # dimensione dello stato nascosto della LSTM
NUM_LATENT = 16       # dimensione dello spazio latente
SEQ_LEN = 20          # lunghezza delle sequenze

Enc = Encoder(
    sensor_dim=SENSOR_DIM,
    num_routes=NUM_ROUTES,
    route_emb_dim=ROUTE_EMB_DIM,
    num_hidden=NUM_HIDDEN,
    num_latent=NUM_LATENT
)

for batch in train_loader:
    z,z_mean,z_var = Enc(batch) 
    y = Enc(batch)
    print(z.shape)
    print(z_mean.shape)
    print(z_var.shape)
    break