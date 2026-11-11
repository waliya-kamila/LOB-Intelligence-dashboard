import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class FI2010TorchDataset(Dataset):

    def __init__(self, X, y, indices):
        self.X = X
        self.y = y
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]

        # Convert one sequence into a contiguous tensor
        x = torch.from_numpy(
            np.ascontiguousarray(self.X[sample_idx], dtype=np.float32)
        )

        # Original labels: 1=Up, 2=Stationary, 3=Down
        # PyTorch labels: 0=Up, 1=Stationary, 2=Down
        y = torch.tensor(
            self.y[sample_idx].astype(np.int64) - 1,
            dtype=torch.long
        )

        return x, y


def create_dataloaders(
    X_seq,
    y_seq,
    train_indices,
    val_indices,
    batch_size=8
):

    train_dataset = FI2010TorchDataset(
        X_seq, y_seq, train_indices
    )

    val_dataset = FI2010TorchDataset(
        X_seq, y_seq, val_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader