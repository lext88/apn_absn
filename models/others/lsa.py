
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        hdim = channel // reduction
        self.conv1x1_in = nn.Sequential(
            nn.Conv2d(channel, hdim, kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(hdim),
            nn.ReLU(inplace=False)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hdim, hdim // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(hdim // reduction, hdim, bias=False),
            nn.Sigmoid()
        )
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(hdim, channel, kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.conv1x1_in(x)
        y = self.avg_pool(y).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        y = self.conv1x1_out(y)
        return y

class LocalSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, kernel_size=1, stride=1, padding=0, bias=False):
        super(LocalSelfAttention, self).__init__()
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        assert self.head_dim * num_heads == out_channels, "out_channels must be divisible by num_heads"

        self.rel_h = nn.Parameter(torch.randn(self.head_dim, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(self.head_dim, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.agg = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.se = SqueezeExcitation(out_channels)  # Add SqueezeExcitation block

        self.reset_parameters()

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x).view(batch_size, self.num_heads, self.head_dim, height, width)
        k_out = self.key_conv(padded_x).unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = self.value_conv(padded_x).unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.head_dim, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch_size, self.num_heads, self.head_dim, height, width, -1)
        v_out = v_out.contiguous().view(batch_size, self.num_heads, self.head_dim, height, width, -1)

        q_out = q_out.view(batch_size, self.num_heads, self.head_dim, height, width, 1)

        attn_scores = (q_out * k_out).sum(dim=2) / (self.head_dim ** 0.5)
        attn_scores = F.softmax(attn_scores, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', attn_scores, v_out).view(batch_size, -1, height, width)
        
        out = self.agg(out)
        out = self.se(out)  # Apply SqueezeExcitation

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)
