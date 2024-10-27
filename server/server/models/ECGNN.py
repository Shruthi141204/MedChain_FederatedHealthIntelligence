import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGNN(nn.Module):
    def __init__(self):
        super(ECGNN, self).__init__()
        self.fc1 = nn.Linear(140, 256)  # First hidden layer
        self.fc2 = nn.Linear(256, 128)   # Second hidden layer
        self.fc3 = nn.Linear(128, 64)    # Third hidden layer
        self.fc4 = nn.Linear(64, 1)       # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Sigmoid for binary classification
        return x
