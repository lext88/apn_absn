import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        hdim = channel  # Ensure hdim is consistent with channel
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hdim // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(hdim // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, seq_len = x.size()
        y = self.avg_pool(x).view(b, c)  # Shape: (b, c)
        y = self.fc(y).view(b, c, 1)     # Shape: (b, c, 1)
        x = x * y.expand_as(x)          # Broadcasting y to match x's shape
        return x
