import matplotlib.pyplot as plt
import numpy as np
import os

def plot_mnist(x: np.ndarray, y: int, folder: str, file_name: str):

    assert isinstance(x, np.ndarray), "x must be "
    assert x.shape == (784,), f"Expected shape (784,), got {x.shape}"

    if not file_name.endswith(".png"):
        file_name += ".png"

    folder = os.path.join(os.getcwd(), folder)
    os.makedirs(folder, exist_ok = True)

    img = x.reshape(28, 28)
    plt.imshow(img, cmap = "gray")
    plt.axis("off")
    plt.title(f"Label: {y}")

    save_path = os.path.join(folder, file_name)
    plt.savefig(save_path, bbox_inches = "tight", pad_inches = 0.1)
    plt.close()

    print(f"[INFO] Mnist sample saved to {save_path}!")

    return