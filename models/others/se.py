# se.py code
import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    def __init__(self, channel):
        super(SqueezeExcitation, self).__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 16, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y
