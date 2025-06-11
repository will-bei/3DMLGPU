import torch.nn as nn
import torch.nn.functional as F

class SimpleNeRFMLP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)  # 1 for density + 3 for color

    def forward(self, x):
        # x shape: [batch_size, 3] (3D points)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        out = self.fc_out(h)  # [batch_size, 4]
        density = F.relu(out[:, 0])  # ensure density >= 0
        color = torch.sigmoid(out[:, 1:4])  # RGB in [0,1]
        return density, color