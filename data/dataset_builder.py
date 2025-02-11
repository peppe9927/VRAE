import traci
import random
import numpy as np
from multiprocessing import Process, Semaphore
import logging
import os
import shutil
import glob  # Importa il modulo glob per la gestione dei pattern dei file

# Configura il logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definizione delle possibilità di rotte
possibilities = {
    1: [['-E3', 'E1'], '-E3'],
    2: [['-E3', 'E2'], '-E3'],
    3: [['-E3', 'E0', 'E6'], '-E3'],
    4: [['-E3', 'E0', 'E5'], '-E3'],
    5: [['-E3', 'E0', 'E4'], '-E3'],
    6: [['-E1', 'E3'], '-E1'],
    7: [['-E1', 'E2'], '-E1'],
    8: [['-E1', 'E0', 'E6'], '-E1'],
    9: [['-E1', 'E0', 'E5'], '-E1'],
    10: [['-E1', 'E0', 'E4'], '-E1'],
    11: [['-E2', 'E3'], '-E2'],
    12: [['-E2', 'E1'], '-E2'],
    13: [['-E2', 'E0', 'E6'], '-E2'],
    14: [['-E2', 'E0', 'E5'], '-E2'],
    15: [['-E2', 'E0', 'E4'], '-E2'],
    16: [['-E6', 'E5'], '-E6'],
    17: [['-E6', 'E4'], '-E6'],
    18: [['-E6', '-E0', 'E1'], '-E6'],
    19: [['-E6', '-E0', 'E2'], '-E6'],
    20: [['-E6', '-E0', 'E3'], '-E6'],
    21: [['-E5', 'E6'], '-E5'],
    22: [['-E5', 'E4'], '-E5'],
    23: [['-E5', '-E0', 'E1'], '-E5'],
    24: [['-E5', '-E0', 'E2'], '-E5'],
    25: [['-E5', '-E0', 'E3'], '-E5'],
    26: [['-E4', 'E5'], '-E4'],
    27: [['-E4', 'E6'], '-E4'],
    28: [['-E4', '-E0', 'E1'], '-E4'],
    29: [['-E4', '-E0', 'E2'], '-E4'],
    30: [['-E4', '-E0', 'E3'], '-E4']
}

MEANs = [0.2,0.3,0.4]
MAX_QUEUEs = [5,6]

# Template del file di configurazione
config_template = """<configuration>
    <input>
        <net-file value="net.net.xml"/>
        <route-files value="cars.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="2000"/>
        <step-length value="0.1"/>
    </time>
</configuration>
"""

# Directory per i file di configurazione temporanei
config_dir = "./VRAE/temp_files"

# Directory per i file di output
output_dir = "./VRAE/datasets"

def create_config_file(index):
    """
    Crea un file di configurazione con il template fornito.
    """
    os.makedirs(config_dir, exist_ok=True)
    config_filename = os.path.join(config_dir, f"config_file_{index}.sumocfg")
    with open(config_filename, 'w') as file:
        file.write(config_template)
    logging.info(f"File di configurazione creato: {config_filename}")
    return config_filename

def creation_dynamic_route(sumo_config, worker_label, iteration):
    """
    Gestisce la dinamica delle rotte e la generazione dei veicoli durante la simulazione.
    """
    try:
        mean = np.random.choice(MEANs)
        max_queue = np.random.choice(MAX_QUEUEs)
        step = 0
        vehicle_id = 0 
        vehicles = []
        # Rimosso traci.load() poiché la connessione è già stabilita in run_simulation
        possible_routes = list(possibilities.keys())
        
        # Aggiungi tutte le rotte definite
        for i in possible_routes:
            traci.route.add(f'route{i}', possibilities[i][0])

        while step < 20000:
            current_simulation_time = traci.simulation.getTime()
            if current_simulation_time < 1000:
                if step % 10 == 0:
                    num_vehicles = np.random.poisson(mean)
                    for _ in range(num_vehicles):
                        vehicle_id += 1
                        route = random.choice(possible_routes)
                        queue_length = traci.edge.getLastStepVehicleNumber(possibilities[route][1])
                        if queue_length < max_queue:
                            traci.vehicle.add(f"veh_{vehicle_id}", f"route{route}", typeID="DEFAULT_VEHTYPE")
                            vehicles.append({
                                'id': vehicle_id, 
                                'rou': possibilities[route][0], 
                                'depart': current_simulation_time
                            })
            traci.simulationStep()
            step += 1

        # Assicurati che la directory di output esista
        os.makedirs(output_dir, exist_ok=True)

        # Correzione della concatenazione delle stringhe: somma invece di concatenazione
        # worker_label è ora un numero intero
        output_file = os.path.join(
            output_dir, 
            f"Cars_{worker_label + iteration * 100}.rou.xml"
        )
        with open(output_file, "w") as fp:
            fp.write("<routes>\n")
            fp.write("  <vType accel='7.0' decel='5.0' id='CarA' maxSpeed='20.0' minGap='2.5'/>\n")
            for vehicle in vehicles:
                route_string = " ".join(vehicle['rou'])
                fp.write(f"  <vehicle depart='{vehicle['depart']}' id='veh_{vehicle['id']}' type='CarA'>\n")
                fp.write(f"    <route edges='{route_string}'/>\n")
                fp.write("  </vehicle>\n")
            fp.write("</routes>\n")

        logging.info(f"Simulazione completata per worker {worker_label}")

    except Exception as e:
        logging.error(f"Errore nella funzione di simulazione per worker {worker_label}: {e}")

def run_simulation(label, port, sumo_config, semaphore, iteration):
    """
    Esegue la simulazione SUMO per un determinato worker.
    """
    with semaphore:
        try:
            logging.info(f"Inizializzazione worker {label} sulla porta {port} con config {sumo_config}")
            sumoCmd = ["sumo", "-c", sumo_config]
            traci.start(sumoCmd, port=port, label=str(label))
            
            creation_dynamic_route(sumo_config, label, iteration)
            
        except Exception as e:
            logging.error(f"Errore nella simulazione {label}: {e}")
        finally:
            try:
                traci.switch(str(label))
                traci.close()
                logging.info(f"Chiusura connessione per worker {label}")
            except Exception as e:
                logging.error(f"Errore durante la chiusura di worker {label}: {e}")

def main():
    NUM_ITERATIONS = 200
    NUM_PROCESSES = 100      # Numero totale di processi
    MAX_CONCURRENT = 50     # Numero massimo di processi concorrenti

    config_files = [create_config_file(i) for i in range(1, NUM_PROCESSES + 1)]
    logging.info(f"{NUM_PROCESSES} file di configurazione creati nella directory '{config_dir}'.")

    for iteration in range(NUM_ITERATIONS // NUM_PROCESSES):
        # Generazione delle label e delle porte
        labels = [i for i in range(1, NUM_PROCESSES + 1)]  # Ora labels sono numeri interi
        base_port = 8813
        ports = [base_port + i for i in range(NUM_PROCESSES)]

        processes = []
        semaphore = Semaphore(MAX_CONCURRENT)  # Limita il numero di processi concorrenti

        for label, port, config in zip(labels, ports, config_files):
            p = Process(target=run_simulation, args=(label, port, config, semaphore, iteration))
            p.start()
            processes.append(p)
            logging.info(f"Processo avviato: worker {label} sulla porta {port}")

        # Attendi che tutti i processi terminino
        for p in processes:
            p.join()
        # Eliminazione solo dei file di configurazione creati
    logging.info("Tutte le simulazioni sono state completate.")

    try:
        # Trova tutti i file di configurazione creati
        config_pattern = os.path.join(config_dir, "config_file_*.sumocfg")
        config_files_created = glob.glob(config_pattern)

        for file_path in config_files_created:
            try:
                os.remove(file_path)
                logging.info(f"File eliminato: {file_path}")
            except Exception as e:
                logging.error(f"Errore nell'eliminazione del file {file_path}: {e}")

        logging.info(f"Tutti i file di configurazione nella directory '{config_dir}' sono stati eliminati.")
    except Exception as e:
        logging.error(f"Errore durante la ricerca o eliminazione dei file di configurazione: {e}")

if __name__ == "__main__":
    main()