import torch
from torchvision import datasets, transforms
from utils.plot_mnist import plot_mnist
import openml
import os
import numpy as np

mnist = openml.datasets.get_dataset(554)

X, y, _, _ = mnist.get_data(target = mnist.default_target_attribute)
X, y = np.array(X, dtype = np.int64), np.array(y, dtype = np.int64)

print(f"X: {X.shape}, y: {y.shape}")

# save in MNIST file
os.makedirs("data", exist_ok = True)
np.savez("data/mnist_cached.npz", feature=X, label=y)

# Load & Plot data
mnist_data = np.load("data/mnist_cached.npz")
X, y = mnist_data["feature"], mnist_data["label"]

for i in range(3):
    plot_mnist(np.array(X[i]), y[i], "images", f"image_{i}")

print("[Info] Download, Load, Plot all done!!")