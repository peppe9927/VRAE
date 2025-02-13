import traci
import random
import numpy as np
from multiprocessing import Process, Semaphore
import logging
import os
import glob  # Import the glob module for handling file patterns
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definition of route possibilities
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

MEANs = {'low': [0.2, 0.3, 0.4],
         'moderated': [1.0, 1.1, 1.2],
         'heavy': [7.0, 8.0, 9.0]}
MAX_QUEUEs = {'low': [3, 4, 5],
              'moderated': [10, 11],
              'heavy': [18, 19, 20]}

# Configuration file template
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

# Directory for temporary configuration files
config_dir = "./temp_files"

# Directory for output files
output_dir = "./datasets"


def create_config_file(index):
    """
    Create a configuration file using the provided template.
    """
    os.makedirs(config_dir, exist_ok=True)
    config_filename = os.path.join(config_dir, f"config_file_{index}.sumocfg")
    with open(config_filename, 'w') as file:
        file.write(config_template)
    logging.info(f"Configuration file created: {config_filename}")
    return config_filename


def creation_dynamic_route(sumo_config, worker_label, iteration, state):
    """
    Handle dynamic routing and vehicle generation during the simulation.
    """
    try:
        mean = np.random.choice(MEANs[state])
        max_queue = np.random.choice(MAX_QUEUEs[state])
        step = 0
        vehicle_id = 0
        vehicles = []
        # Removed traci.load() since the connection is already established in run_simulation
        possible_routes = list(possibilities.keys())

        # Add all defined routes
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

        state_dir = os.path.join(output_dir, state)  # Create a directory for the traffic state
        os.makedirs(state_dir, exist_ok=True)

        output_file = os.path.join(
            state_dir,
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

        logging.info(f"Simulation completed for worker {worker_label}")

    except Exception as e:
        logging.error(f"Error in simulation function for worker {worker_label}: {e}")


def run_simulation(label, port, sumo_config, semaphore, iteration, state):
    """
    Run the SUMO simulation for a specific worker.
    """
    with semaphore:
        try:
            logging.info(f"Initializing worker {label} on port {port} with config {sumo_config}")
            sumoCmd = ["sumo", "-c", sumo_config]
            traci.start(sumoCmd, port=port, label=str(label))
            
            creation_dynamic_route(sumo_config, label, iteration, state)
            
        except Exception as e:
            logging.error(f"Error in simulation {label}: {e}")
        finally:
            try:
                traci.switch(str(label))
                traci.close()
                logging.info(f"Connection closed for worker {label}")
            except Exception as e:
                logging.error(f"Error closing worker {label}: {e}")


def main(NUM_ITERATIONS, NUM_PROCESSES, MAX_CONCURRENT, state):
    config_files = [create_config_file(i) for i in range(1, NUM_PROCESSES + 1)]
    logging.info(f"{NUM_PROCESSES} configuration files created in directory '{config_dir}'.")

    # Calculate the number of iterations per batch
    iterations_per_batch = NUM_ITERATIONS // NUM_PROCESSES
    for iteration in range(iterations_per_batch):
        # Generate labels and ports
        labels = [i for i in range(1, NUM_PROCESSES + 1)]  # Labels are now integers
        base_port = 8813
        ports = [base_port + i for i in range(NUM_PROCESSES)]

        processes = []
        semaphore = Semaphore(MAX_CONCURRENT)  # Limit the number of concurrent processes

        for label, port, config in zip(labels, ports, config_files):
            p = Process(target=run_simulation, args=(label, port, config, semaphore, iteration, state))
            p.start()
            processes.append(p)
            logging.info(f"Worker {label} started on port {port}")

        # Wait for all processes to finish
        for p in processes:
            p.join()

    logging.info("All simulations have been completed.")

    try:
        # Find all created configuration files
        config_pattern = os.path.join(config_dir, "config_file_*.sumocfg")
        config_files_created = glob.glob(config_pattern)

        for file_path in config_files_created:
            try:
                os.remove(file_path)
                logging.info(f"File deleted: {file_path}")
            except Exception as e:
                logging.error(f"Error deleting file {file_path}: {e}")

        logging.info(f"All configuration files in directory '{config_dir}' have been deleted.")
    except Exception as e:
        logging.error(f"Error searching or deleting configuration files: {e}")


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="SUMO simulation with multiprocessing")
    parser.add_argument('--iterations', type=int, default=200, help='Total number of iterations')
    parser.add_argument('--processes', type=int, default=100, help='Total number of processes')
    parser.add_argument('--concurrent', type=int, default=50, help='Maximum number of concurrent processes')
    parser.add_argument('--state', type=str, choices=['low', 'moderated', 'heavy'], required=True,
                        help='Traffic state (low, moderated, heavy)')
    args = parser.parse_args()
    
    main(args.iterations, args.processes, args.concurrent, args.state)  # Run the simulation