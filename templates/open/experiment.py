import argparse
import inspect
import json
import math
import os
import pickle
import time
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# --- BEGIN model.py ---

# TBD define model(s) here


# --- END model.py ---


def get_dataset(dataset_name, data_dir):
    """Retrieve dataset with appropriate preprocessing"""
    os.makedirs(data_dir, exist_ok=True)

    # Data transformations for image datasets
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Function to load character-level datasets
    def load_char_dataset(split):
        filename = f"{split}.bin"
        filepath = os.path.join(data_dir, dataset_name, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found.")
        return np.memmap(filepath, dtype=np.uint16, mode="r")

    # Dataset mappings
    datasets_map = {
        'mnist': lambda: {
            'train': datasets.MNIST(data_dir, train=True, download=True, transform=transform_mnist),
            'test': datasets.MNIST(data_dir, train=False, transform=transform_mnist),
            'input_size': (1, 28, 28),
            'num_classes': 10,
            'model_type': None,  # TBD
        },
        'shakespeare_char': lambda: {
            'train': load_char_dataset('train'),
            'test': load_char_dataset('val'),
            'input_size': None,  # Handled dynamically based on block size
            'num_classes': None,  # Vocabulary size
            'model_type': None,  # TBD
        },
        'enwik8': lambda: {
            'train': load_char_dataset('train'),
            'test': load_char_dataset('val'),
            'input_size': None,
            'num_classes': None,
            'model_type': None,  # TBD
        },
        'text8': lambda: {
            'train': load_char_dataset('train'),
            'test': load_char_dataset('val'),
            'input_size': None,
            'num_classes': None,
            'model_type': None,  # TBD
        }
    }
    # access and initialise only when needed, eg. datasets_map['text8']()

    # Add vocabulary size metadata for character-level datasets
    dataset = datasets_map.get(dataset_name, lambda: None)()
    if dataset_name in ['shakespeare_char', 'enwik8', 'text8']:
        meta_path = os.path.join(data_dir, dataset_name, 'meta.pkl')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"{meta_path} not found. Run preprocessing first.")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        dataset['num_classes'] = meta['vocab_size']

    return dataset


def train(
        dataset='mnist',
        out_dir='run_0',
        seed_offset=0,
        batch_size=64,
        block_size=256,  # Context size for character datasets
        max_iters=5000,
        learning_rate=1e-3,
        weight_decay=1e-2
):
    # Set random seed
    torch.manual_seed(1337 + seed_offset)
    np.random.seed(1337 + seed_offset)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data preparation
    dataset_info = get_dataset(dataset, data_dir=os.path.join("data", dataset))
    if dataset_info['train'] is None:
        raise ValueError(f"Dataset {dataset} not fully implemented yet.")

    # decide on block size based on whether it's an image or character dataset
    is_text_dataset = dataset in ['shakespeare_char', 'enwik8', 'text8']
    if not is_text_dataset:
        block_size = 1

    # Prepare data loaders
    if dataset_info['model_type'] == ...:
        model = ...  # TBD
        train_data = dataset_info['train']
        train_loader = ...  # TBD
        val_data = dataset_info['test']
        val_loader = ...  # TBD

    # Model selection and initialization
    if dataset_info['model_type'] == ...:
        model = ...  # TBD

    model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Training loop
    train_log_info, val_log_info = [], []
    best_val_loss = float('inf')

    for epoch in range(max_iters // len(train_loader) + 1):
        model.train()
        total_train_loss = 0

        # Train character-level or batched datasets
        if is_text_dataset:
            for _ in range(len(train_data) // (batch_size * block_size)):
                X, Y = train_loader()
                optimizer.zero_grad()
                logits, loss = model(X, Y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
        else:
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for _ in range(len(val_data) // (batch_size * block_size)):
                X, Y = val_loader()
                logits, loss = model(X, Y)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, 'best_model.pt'))

    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Machine Learning Training")
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'shakespeare_char', 'enwik8', 'text8'])
    parser.add_argument('--out_dir', type=str, default='run_0')
    args = parser.parse_args()

    num_seeds = {
        'mnist': 3,
        'shakespeare_char': 3,
        'enwik8': 1,
        'text8': 1
    }

    all_results = {}
    final_infos = {}

    for dataset in [args.dataset]:
        final_info_list = []
        for seed_offset in range(num_seeds[dataset]):
            final_info, train_info, val_info = train(dataset, args.out_dir, seed_offset)
            all_results[f"{dataset}_{seed_offset}_final_info"] = final_info
            all_results[f"{dataset}_{seed_offset}_train_info"] = train_info
            all_results[f"{dataset}_{seed_offset}_val_info"] = val_info
            final_info_list.append(final_info)

        # Compute statistics
        final_info_dict = {k: [d[k] for d in final_info_list] for k in final_info_list[0].keys()}
        means = {f"{k}_mean": np.mean(v) for k, v in final_info_dict.items()}
        stderrs = {f"{k}_stderr": np.std(v) / np.sqrt(len(v)) for k, v in final_info_dict.items()}

        final_infos[dataset] = {
            "means": means,
            "stderrs": stderrs,
            "final_info_dict": final_info_dict,
        }

    # Save comprehensive results
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)

    with open(os.path.join(args.out_dir, "all_results.npy"), "wb") as f:
        np.save(f, all_results)


if __name__ == "__main__":
    main()
