import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset

class TrafficGenerativeDataset(Dataset):
    def __init__(self, root_dir, transform=None, route_vocab=None, pad_token=0, class_vocab=None):
        """
        Args:
            root_dir (str): percorso alla cartella che contiene i file XML organizzati in sottocartelle per classi.
            transform (callable, optional): trasformazione da applicare a ogni sample.
            route_vocab (dict, optional): dizionario che mappa le stringhe delle route a liste di token (ad esempio one-hot).
            pad_token (int, optional): token da utilizzare per il padding degli edge (non usato qui perché
                                       la codifica one-hot prevede vettori di lunghezza fissa).
            class_vocab (dict, optional): dizionario che mappa le etichette delle classi a numeri interi.
                                          Se non viene passato, la classe rimane una stringa.
        """
        self.transform = transform
        self.samples = []  # Ogni elemento sarà una tupla (file_path, file_class)
        self.route_vocab = route_vocab
        self.pad_token = pad_token
        self.class_vocab = class_vocab
        
        # Scorriamo ricorsivamente la cartella root_dir
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.xml'):
                    file_path = os.path.join(root, file)
                    # Estraiamo la classe dal percorso: assumiamo che la prima cartella (relativa a root_dir)
                    # rappresenti la classe. Ad esempio: datasets/heavy/route1.xml -> "heavy"
                    rel_path = os.path.relpath(file_path, root_dir)
                    parts = rel_path.split(os.sep)
                    file_class = parts[0] if len(parts) > 1 else "unknown"
                    # Se è stato fornito un mapping per le classi, lo usiamo per convertire la stringa in un intero
                    if self.class_vocab is not None:
                        file_class = self.class_vocab.get(file_class, self.class_vocab.get('<unk>', 0))
                    self.samples.append((file_path, file_class))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Estraiamo il percorso del file e la sua classe (derivata dal nome della cartella)
        file_path, file_class = self.samples[idx]
        tree = ET.parse(file_path)
        root_xml = tree.getroot()
        
        depart_times = []
        routes = []
        
        # Itera su ogni veicolo presente nel file XML
        for vehicle in root_xml.findall('vehicle'):
            # Estrazione del tempo di partenza
            depart_str = vehicle.get('depart')
            try:
                depart = float(depart_str)
            except (TypeError, ValueError):
                depart = 0.0
            depart_times.append(depart)
            
            # Estrazione della route
            route_elem = vehicle.find('route')
            if route_elem is not None:
                edges_str = route_elem.get('edges', '')
                # Usando il dizionario di codifica one-hot; se non viene trovata, usa il token di padding
                tokens = self.route_vocab.get(edges_str, [self.pad_token])
                routes.append(tokens)
        
        

        # Conversione delle liste in tensori
        dt = torch.tensor(depart_times, dtype=torch.float)
        depart_times_tensor = (dt - dt.mean()) / dt.std()
        routes_tensor = torch.tensor(routes, dtype=torch.long)  # forma: [num_vehicles, route_vector_length]
        
        EOS_depart_value = 5.0
        EOS_tensor = torch.tensor([EOS_depart_value],
                          dtype=depart_times_tensor.dtype,
                          device=depart_times_tensor.device)
        
        depart_times_tensor = torch.cat((depart_times_tensor, EOS_tensor), dim=0) # Aggiungiamo il token EOS
        EOS_route_value = self.route_vocab.get('EOS', 0)
        EOS_tensor = torch.tensor([EOS_route_value],
                          dtype=routes_tensor.dtype,
                          device=routes_tensor.device)
        routes_tensor = torch.cat((routes_tensor, EOS_tensor), dim=0)
         
        sample = {
            'depart_times': depart_times_tensor,
            'routes': routes_tensor,
            'class': file_class
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
