import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from multiprocessing import Pool, freeze_support
import time

class TrafficGenerativeDatasetTrajectory(Dataset):
    def __init__(self, root_dir, num_workers=4):
        print('Inizializzazione dataset...')
        self.samples = []
        
        # Raccolta percorsi file
        for root, _, files in os.walk(root_dir):
            self.samples.extend([os.path.join(root_dir, file) for file in files])
        
        # Trova lunghezza massima usando multiprocessing
        with Pool(num_workers) as pool:
            lengths = pool.map(self._get_sequence_length, self.samples)
        self.max_length = max(lengths)
        print(f'Lunghezza massima sequenza: {self.max_length}')
    
    @staticmethod
    def _get_sequence_length(file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        vehicles = defaultdict(int)
        
        for timestep in root.findall('timestep'):
            for vehicle in timestep.findall('vehicle'):
                vehicles[vehicle.get('id')] += 1
            
        return max(vehicles.values())
    
    @staticmethod
    def _process_file(args):
        file_path, max_length = args
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Pre-allocazione dizionario per veicoli
        vehicles_data = defaultdict(lambda: np.zeros((max_length, 2)))
        vehicle_lengths = defaultdict(int)
        
        # Estrazione dati
        for timestep in root.findall('timestep'):
            for vehicle in timestep.findall('vehicle'):
                vid = vehicle.get('id')
                idx = vehicle_lengths[vid]
                if idx < max_length:
                    vehicles_data[vid][idx] = [float(vehicle.get('x')), float(vehicle.get('y'))]
                    vehicle_lengths[vid] += 1
        
        # Converti in tensore
        sequences = [torch.tensor(data[:length], dtype=torch.float32) 
                    for data, length in zip(vehicles_data.values(), vehicle_lengths.values())]
        padded_tensor = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        
        return padded_tensor
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self._process_file((self.samples[idx], self.max_length))


if __name__ == '__main__':
    # Initialize multiprocessing support

    freeze_support()
    
    # Create dataset
    trj = TrafficGenerativeDatasetTrajectory(root_dir='./datasets/low/trj', num_workers=8)
    
    # Process samples
    print(trj.max_length)