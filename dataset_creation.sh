#!/bin/bash
set -euo pipefail

usage() {
    echo "Usage: $0 -i iterations -p processes -c concurrent -b batch_size" >&2
    exit 1
}

# Inizializza le variabili
iterations=""
processes=""
concurrent=""
batch_size=""

# Parsing degli argomenti
while getopts "i:p:c:b:a:" opt; do

    case $opt in
        i) iterations="$OPTARG" ;;
        p) processes="$OPTARG" ;;
        c) concurrent="$OPTARG" ;;
        b) batch_size="$OPTARG" ;;
        *) usage ;;
    esac
done
shift $((OPTIND-1))

# Verifica che tutti i parametri obbligatori siano stati forniti
if [ -z "$iterations" ] || [ -z "$processes" ] || [ -z "$concurrent" ] || [ -z "$batch_size" ]; then
    echo "Error: Missing required parameters." >&2
    usage
fi

# Verifica che la directory "VRAE" esista e spostati al suo interno
if [ -d "VRAE" ]; then
    cd VRAE
else
    echo "Error: Directory 'VRAE' not found." >&2
    exit 1
fi

# Calculate test processes as minimum between specified processes and iterations

# Esegui il dataset builder per gli stati "low" e "moderated"
for state in "low" "moderated" "heavy"; do
    echo "Building dataset for state: $state"
    python3 "./dataset_builder.py" --iterations "$iterations" --processes "$processes" --concurrent "$concurrent" --state "$state"
done

# (Facoltativo) Esegui il dataset builder per lo stato "heavy"
# echo "Building dataset for state: heavy"
# python3 "./dataset_builder.py" --iterations "$iterations" --processes "$processes" --concurrent "$concurrent" --state "heavy"

# Esegui lo script di conversione delle rotte
echo "Running route converter script..."
python3 "./route_converter.py" --save-path "./datasets/training_data.pt"

# Elimina i file nelle directory dei dataset
for dir in ./datasets/low ./datasets/moderated ./datasets/heavy; do
    if [ -d "$dir" ]; then
        echo "Cleaning directory: $dir"
        rm -f "$dir"/*
    else
        echo "Warning: directory $dir does not exist."
    fi
done

test_processes=$((processes < iterations/4 ? processes : iterations/4))
# Update processes for test dataset
processes=$test_processes

# Esegui il dataset builder per gli stati "low" e "moderated"
for state in "low" ; do
    echo "Building test dataset for state: $state"
    python3 "./dataset_builder.py" --iterations "$((iterations/4))" --processes "$processes" --concurrent "$concurrent" --state "$state"
done

# (Facoltativo) Esegui il dataset builder per lo stato "heavy"
# echo "Building dataset for state: heavy"
# python3 "./dataset_builder.py" --iterations "$iterations" --processes "$processes" --concurrent "$concurrent" --state "heavy"

# Esegui lo script di conversione delle rotte
echo "Running route converter script..."
python3 "./route_converter.py" --save-path "./datasets/test_dataset.pt"

# Elimina i file nelle directory dei dataset
for dir in ./datasets/low ./datasets/moderated ./datasets/heavy; do
    if [ -d "$dir" ]; then
        echo "Cleaning directory: $dir"
        rm -f "$dir"/*
    else
        echo "Warning: directory $dir does not exist."
    fi
done