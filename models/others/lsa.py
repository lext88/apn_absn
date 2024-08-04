import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class LocalSelfAttentionWithSEAndCCA(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, kernel_size=1, stride=1, padding=0, bias=False):
        super(LocalSelfAttentionWithSEAndCCA, self).__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        assert self.head_dim * num_heads == out_channels, "out_channels must be divisible by num_heads"

        self.rel = nn.Parameter(torch.randn(self.head_dim, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.agg = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        self.se_block = SqueezeExcitation(out_channels)
        self.cca_block = CCA(channel=out_channels, kernel_sizes=[3, 3], planes=[out_channels // 2, out_channels])

        self.reset_parameters()

    def forward(self, x):
        batch_size, channels, seq_len = x.size()

        padded_x = F.pad(x, [self.padding, self.padding])
        q_out = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, seq_len)
        k_out = self.key_conv(padded_x).unfold(2, self.kernel_size, self.stride)
        v_out = self.value_conv(padded_x).unfold(2, self.kernel_size, self.stride)

        k_out = k_out + self.rel.unsqueeze(0).unsqueeze(0)
        k_out = k_out.contiguous().view(batch_size, self.num_heads, self.head_dim, seq_len, -1)
        v_out = v_out.contiguous().view(batch_size, self.num_heads, self.head_dim, seq_len, -1)

        q_out = q_out.view(batch_size, self.num_heads, self.head_dim, seq_len, 1)

        attn_scores = (q_out * k_out).sum(dim=2) / (self.head_dim ** 0.5)
        attn_scores = F.softmax(attn_scores, dim=-1)
        out = torch.einsum('bnhql,bnhl -> bnhq', attn_scores, v_out).view(batch_size, -1, seq_len)

        out = self.agg(out)
        out = self.se_block(out)
        out = self.cca_block(out)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel, 0, 1)
