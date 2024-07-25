import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets):
        # Use PyTorch's built-in CrossEntropyLoss for numerical stability and better GPU support
        return F.cross_entropy(inputs, targets)
