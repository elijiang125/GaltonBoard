import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class CircuitDataset(Dataset):
    def __init__(self, mode: str, data_dir: Path):
        # Read the files
        features_path = data_dir.joinpath(f"features_{mode}.npz")
        targets_path = data_dir.joinpath(f"targets_{mode}.npz")

        self.features = np.load(features_path)["arr_0"]
        self.targets = np.load(targets_path)["arr_0"]

    def __len__(self):
        # Return the total number of samples
        return self.features.shape[0]

    def __getitem__(self, idx):
        # Generate one sample of data
        features_row = self.features[idx, :]
        targets_row = self.targets[idx, :]

        # Return as torch tensors
        X = torch.tensor(features_row).float()
        y = torch.tensor(targets_row).float()

        return X, y

