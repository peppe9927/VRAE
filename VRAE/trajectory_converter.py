import os
from lxml import etree
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
        self.num_worker = num_workers

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
        
        vehicles = defaultdict(int)
        for event, timestep in etree.iterparse(file_path, tag="timestep"):
            
            seen_ids = {vehicle.get("id") for vehicle in timestep.findall("vehicle")}
            for vid in seen_ids:
                vehicles[vid] += 1
            
            timestep.clear()
            while timestep.getprevious() is not None:
             del timestep.getparent()[0]
        
        return max(vehicles.values(),default=0)
    
    @staticmethod
    def _process_file(args):
        file_path, max_length = args
        
        # Pre-allocazione dizionario per veicoli
        vehicles_data = defaultdict(lambda: np.zeros((max_length, 5)))
        vehicle_lengths = defaultdict(int)
        
        # Estrazione dati
        for event, timestep in etree.iterparse(file_path, tag="timestep"):
            for vehicle in timestep.findall('vehicle'):
                vid = vehicle.get('id')
                idx = vehicle_lengths[vid] # 
                if idx < max_length:
                    vehicles_data[vid][idx] = np.array([
                        float(vehicle.get('x')),
                        float(vehicle.get('y')),
                        float(vehicle.get('speed')),
                        float(vehicle.get('angle')),
                        float(vehicle.get('pos'))
                        ])
                    vehicle_lengths[vid] += 1
            
            timestep.clear()
            while timestep.getprevious() is not None:
                del timestep.getparent()[0]                    
        
        # Converti in tensore
        sequences = [torch.tensor(data[:length], dtype=torch.float32) 
                    for data, length in zip(vehicles_data.values(), vehicle_lengths.values())]

        return sequences
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self._process_file((self.samples[idx], self.max_length))


if __name__ == '__main__':
    # Initialize multiprocessing support
    start_time = time.time()
    freeze_support()
    
    # Create dataset
    trj = TrafficGenerativeDatasetTrajectory(root_dir='./datasets/low/trj', num_workers=8)
    
    with Pool(trj.num_worker) as pool:
        results = pool.map(trj.__getitem__, range(len(trj)))
    
    # Process samples
    
    flattened = [tensor for sublist in results for tensor in sublist]
    
    padded_tensor = pad_sequence(flattened, batch_first=True, padding_value=0.0)
    
    print('Dataset completed, dataset composed of:', padded_tensor.shape)
    
    end_time = time.time()

    print(f'time of execution {end_time-start_time}')