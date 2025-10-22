import torch
from torchvision import datasets, transforms
from utils.plot_mnist import plot_mnist
import openml
import os
import numpy as np

# List of Datasets in openml
# df = openml.datasets.list_datasets(output_format="dataframe")
# print(df.head())
# os.environ["OPENML_CACHE_DIR"] = os.path.abspath("./openml")
print("Current working directory: ", os.getcwd())

# Openml: Platform for sharing datasets, algorithms
# openml.config.cache_directory = './openml'
# print(f"Cache dir: {os.path.abspath(openml.config.cache_directory)}")
# print("ENV OPENML_CACHE_DIR: ", os.environ.get("OPENML_CACHE_DIR"))

mnist = openml.datasets.get_dataset(554)

X, y, _, _ = mnist.get_data(target = mnist.default_target_attribute)
print(f"X: {X.shape}, y: {y.shape}")

for i in range(10):
    plot_mnist(np.array(X.iloc[i]), y[i], "images", f"image_{i}")