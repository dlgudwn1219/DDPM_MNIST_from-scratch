import torch.nn as nn

class DNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                 # (1,28,28) → (784,)
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)            # 출력: 10 classes
        )

    def forward(self, x):
        return self.net(x)