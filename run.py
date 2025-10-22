import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataloader import MNISTDataLoader
from models.DNN import DNNClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load dataloaders
MNIST_data_module = MNISTDataLoader(batch_size=64)
train_loader, val_loader, test_loader = MNIST_data_module.dataloaders()

# Model, Loss, Optimizer are needed for ML!
model = DNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train, Validation, Test
for epoch in range(5):

    # Validate first: Is it initially randomized correctly?
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"[INFO] Epoch {epoch} | Validation Accuracy: {acc:.4f}")

    # Train Second
    model.train()
    total_loss = 0
    progress = tqdm(train_loader, desc = f"Epoch {epoch + 1}")

    for x, y in progress:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix({"Train loss": f"{round(loss.item(), 2)}"})

    avg_loss = total_loss / len(train_loader)
    print(f"[INFO] Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"Final Test Accuracy: {correct/total:.4f}")
