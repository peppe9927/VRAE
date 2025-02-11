import torch
import torch.nn.functional as F

def custom_collate_fn(batch):
    """
    Questa funzione di collate gestisce il padding dei sample del batch, oltre a
    ordinare i sample in ordine decrescente in base alla lunghezza effettiva (numero di veicoli).
    - Per 'depart_times' padding con -1.0.
    - Per 'routes' padding con -1.
    - 'class' viene aggregata in un tensore se numerica.
    """
    # Ordina i sample in base alla lunghezza della sequenza (depart_times) in ordine decrescente
    batch = sorted(batch, key=lambda sample: sample['depart_times'].size(0), reverse=True)
    
    max_num_vehicles = batch[0]['depart_times'].size(0)
    
    batch_depart_times = []
    batch_routes = []
    batch_classes = []
    
    for sample in batch:
        depart_times = sample['depart_times']
        routes = sample['routes']
        file_class = sample['class']
        num_vehicles = depart_times.size(0)
        
        if num_vehicles < max_num_vehicles:
            pad_size = max_num_vehicles - num_vehicles
            depart_times = F.pad(depart_times, (0, pad_size), "constant", 0)
            routes = F.pad(routes, (0, pad_size), "constant", 0)
        
        batch_depart_times.append(depart_times)
        batch_routes.append(routes)
        batch_classes.append(file_class)
    
    batch_depart_times = torch.stack(batch_depart_times, dim=0).unsqueeze(2)  # [batch_size, max_num_vehicles]
    batch_routes = torch.stack(batch_routes, dim=0)             # [batch_size, max_num_vehicles, route_vector_length]
    
    if isinstance(batch_classes[0], int):
        batch_classes = torch.tensor(batch_classes, dtype=torch.long)
    
    # Aggiungiamo le lunghezze effettive (numero di veicoli) prima del padding come chiave "lengths"
    batch_lengths = [sample['depart_times'].size(0) for sample in batch]
    batch_lengths = torch.stack([torch.tensor(l) for l in batch_lengths], dim=0)
    
    return {
        'times': batch_depart_times,  # [batch_size, max_num_vehicles]
        'routes': batch_routes,       # [batch_size, max_num_vehicles, route_vector_length]
        'classes': batch_classes,     # [batch_size]
        'lengths': batch_lengths      # [batch_size]
    }
