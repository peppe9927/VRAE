import argparse
import torch
from torch.utils.data import DataLoader
import os
from utils import custom_collate_fn, TrafficGenerativeDataset

# Example dictionary for mapping edges
edge_vocab = {
    '<unk>': -1,
    'E0': 1, 'E1': 2, 'E2': 3, 'E3': 4, 'E4': 5, 'E5': 6, 'E6': 7,
    '-E0': 8, '-E1': 9, '-E2': 10, '-E3': 11, '-E4': 12, '-E5': 13, '-E6': 14
}

# Example dictionary for one-hot route mapping
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

# Example dictionary for mapping classes
class_vocab = {
    '<unk>': -1,
    'low': 0,
    'moderated': 1,
    'heavy': 2,
}

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Load dataset and create DataLoader")
    parser.add_argument(
        '--folder',
        type=str,
        default='./datasets',
        help='Reference folder for the dataset (default: ./datasets)'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        required=True,
        help='Path to save the processed dataset'
    )
    args = parser.parse_args()

    print(f"Loading dataset from: {args.folder}")

    # Initialize the dataset
    train_dataset = TrafficGenerativeDataset(
        root_dir=args.folder,
        route_vocab=one_hot_routes,
        pad_token=edge_vocab.get('<unk>', -1),
        class_vocab=class_vocab
    )

    print("Dataset successfully loaded.")
    print(f"Total dataset size: {len(train_dataset)} samples")

    # Convert the dataset to a list of samples
    all_samples = [sample for sample in train_dataset]

    # Save the processed dataset to the specified file path
    torch.save(all_samples, args.save_path)
    print("Dataset conversion completed.")
    print(f"Dataset saved to: {args.save_path}")

if __name__ == '__main__':
    main()