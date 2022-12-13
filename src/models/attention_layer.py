import torch
from torch import nn

class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        out = self.linear(x)
        weight = torch.softmax(out.squeeze(dim=-1), dim=-1)
        return weight