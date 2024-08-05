# nlsa.py code
import torch
from torch import nn
from torch.nn import functional as F

class NonLocalSelfAttentionWithSEAndCCA(nn.Module):
    def __init__(self, in_channels=256, inter_channels=None, num_heads=8):
        super(NonLocalSelfAttentionWithSEAndCCA, self).__init__()

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv1d
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.se_block = SqueezeExcitation(in_channels)
        self.cca_block = CCA(channel=in_channels, kernel_sizes=[3, 3], planes=[in_channels // 2, in_channels])

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(2)

        g_x = self.g(x).view(batch_size, self.num_heads, self.head_dim, seq_len)
        g_x = g_x.permute(0, 1, 3, 2)

        theta_x = self.theta(x).view(batch_size, self.num_heads, self.head_dim, seq_len)
        theta_x = theta_x.permute(0, 1, 3, 2)
        phi_x = self.phi(x).view(batch_size, self.num_heads, self.head_dim, seq_len)

        attn_scores = torch.matmul(theta_x, phi_x) / (self.head_dim ** 0.5)
        attn_scores = F.softmax(attn_scores, dim=-1)

        y = torch.matmul(attn_scores, g_x)
        y = y.permute(0, 1, 3, 2).contiguous()
        y = y.view(batch_size, self.inter_channels, seq_len)

        W_y = self.W(y)

        W_y = self.se_block(W_y)
        W_y = self.cca_block(W_y)

        return W_y
