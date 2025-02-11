import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import torch.nn.functional as F
from VRAE.utils import custom_collate_fn
from VRAE.utils import TrafficGenerativeDataset
import sys
import os

# Esempio di dizionari per il mapping degli edge e delle classi
edge_vocab = {
    '<unk>': -1,
    'E0': 1, 'E1': 2, 'E2': 3, 'E3': 4, 'E4': 5, 'E5': 6, 'E6': 7,
    '-E0': 8, '-E1': 9, '-E2': 10, '-E3': 11, '-E4': 12, '-E5': 13, '-E6': 14
}

one_hot_routes = {
    '-E3 E1': 1,
    '-E3 E2': 2,
    '-E3 E0 E6': 3,
    '-E3 E0 E5': 4,
    '-E3 E0 E4': 5,
    '-E1 E3': 6,
    '-E1 E2': 7,
    '-E1 E0 E6': 8,
    '-E1 E0 E5': 9,
    '-E1 E0 E4': 10,
    '-E2 E3': 11,
    '-E2 E1': 12,
    '-E2 E0 E6': 13,
    '-E2 E0 E5': 14,
    '-E2 E0 E4': 15,
    '-E6 E5': 16,
    '-E6 E4': 17,
    '-E6 -E0 E1': 18,
    '-E6 -E0 E2': 19,
    '-E6 -E0 E3': 20,
    '-E5 E6': 21,
    '-E5 E4': 22,
    '-E5 -E0 E1': 23,
    '-E5 -E0 E2': 24,
    '-E5 -E0 E3': 25,
    '-E4 E5': 26,
    '-E4 E6': 27,
    '-E4 -E0 E1': 28,
    '-E4 -E0 E2': 29,
    '-E4 -E0 E3': 30,
    'EOS': 31
}

class_vocab = {
    '<unk>': -1,
    'low': 0,
    'moderated': 1,
    'heavy': 2,
}

train_dir = './VRAE/datasets'

train_dataset = TrafficGenerativeDataset(
    root_dir=train_dir,
    route_vocab=one_hot_routes,
    pad_token=edge_vocab.get('<unk>', -1),
    class_vocab=class_vocab
)

print("Working directory:", os.getcwd())
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=custom_collate_fn)

torch.set_printoptions(threshold=torch.iinfo(torch.int64).max, precision=4)
print(next(iter(train_loader))['times'])
print("Numero di elementi nel train_dataset:", len(train_dataset))
print("---------------TEST-------------------")

for batch in train_loader:
    print("Batch times:", batch['times'].shape)
    print("Batch routes:", batch['routes'].shape)
    break

# Salva i sample per uso futuro (opzionale)
all_samples = [sample for sample in train_dataset] # salva tutti i tensori che costituiscono il dataset
torch.save(all_samples, 'train_dataset_samples.pt')