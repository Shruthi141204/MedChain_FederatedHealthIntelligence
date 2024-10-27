import torch
import torch.nn as nn

class ECGClassifier(nn.Module):
    def __init__(self):  # Pass input_dim and output_dim as parameters
        super(ECGClassifier, self).__init__()
        self.fc1 = nn.Linear(32, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 5)  # Use output_dim for the final layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
