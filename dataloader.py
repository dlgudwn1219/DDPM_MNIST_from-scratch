import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MNISTTransform:
    def __call__(self, x):
        x = x / 255.0
        x = x.view(1, 28, 28) # channel, h, w
        return x

class MNISTDataset(Dataset):
    def __init__(self, X, y, transform = None):
        self.X = torch.as_tensor(X, dtype = torch.long)
        self.y = torch.as_tensor(y, dtype = torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

class MNISTDataLoader:
    def __init__(self, data_path="data/mnist_cached.npz", batch_size=64):
        self.data_path = data_path
        self.batch_size = batch_size
        self.transform = MNISTTransform()
        self.setup()

    def setup(self):
        data = np.load(self.data_path)
        X, y = data["feature"], data["label"]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=10000, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        self.train_dataset = MNISTDataset(X_train, y_train, transform=self.transform)
        self.val_dataset   = MNISTDataset(X_val, y_val, transform=self.transform)
        self.test_dataset  = MNISTDataset(X_test, y_test, transform=self.transform)
    
    def dataloaders(self):
        return (DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True),
                DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False),
                DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False),
                )
