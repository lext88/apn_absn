import torch
from torch import nn
from torch.nn import functional as F

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

class CCA(nn.Module):
    def __init__(self, kernel_sizes=[3, 3], planes=[16, 1]):
        super(CCA, self).__init__()
        num_layers = len(kernel_sizes)
        nn_modules = list()

        for i in range(num_layers):
            ch_in = 1 if i == 0 else planes[i - 1]
            ch_out = planes[i]
            k_size = kernel_sizes[i]
            nn_modules.append(SepConv4d(in_planes=ch_in, out_planes=ch_out, ksize=k_size, do_padding=True))
            if i != num_layers - 1:
                nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)

    def forward(self, x):
        x = self.conv(x) + self.conv(x.permute(0, 1, 4, 5, 2, 3)).permute(0, 1, 4, 5, 2, 3)
        return x

class SepConv4d(nn.Module):
    def __init__(self, in_planes, out_planes, stride=(1, 1, 1), ksize=3, do_padding=True, bias=False):
        super(SepConv4d, self).__init__()
        self.isproj = False
        padding1 = (0, ksize // 2, ksize // 2) if do_padding else (0, 0, 0)
        padding2 = (ksize // 2, ksize // 2, 0) if do_padding else (0, 0, 0)

        if in_planes != out_planes:
            self.isproj = True
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=bias, padding=0),
                nn.BatchNorm2d(out_planes))

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_planes, out_channels=in_planes, kernel_size=(1, ksize, ksize),
                      stride=stride, bias=bias, padding=padding1),
            nn.BatchNorm3d(in_planes))
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=in_planes, out_channels=in_planes, kernel_size=(ksize, ksize, 1),
                      stride=stride, bias=bias, padding=padding2),
            nn.BatchNorm3d(in_planes))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, u, v, h, w = x.shape
        x = self.conv2(x.view(b, c, u, v, -1))
        b, c, u, v, _ = x.shape
        x = self.relu(x)
        x = self.conv1(x.view(b, c, -1, h, w))
        b, c, _, h, w = x.shape

        if self.isproj:
            x = self.proj(x.view(b, c, -1, w))
        x = x.view(b, -1, u, v, h, w)
        return x

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, num_heads=8, dimension=3, sub_sample=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d if dimension == 2 else nn.Conv1d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2)) if dimension == 2 else nn.MaxPool1d(kernel_size=2)
        bn = nn.BatchNorm2d if dimension == 2 else nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
        
        # Integrate CCA block
        self.cca = CCA(kernel_sizes=[3, 3], planes=[self.inter_channels, self.inter_channels])

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        g_x = self.g(x).view(batch_size, self.num_heads, self.head_dim, -1)
        g_x = g_x.permute(0, 1, 3, 2)

        theta_x = self.theta(x).view(batch_size, self.num_heads, self.head_dim, -1)
        theta_x = theta_x.permute(0, 1, 3, 2)
        phi_x = self.phi(x).view(batch_size, self.num_heads, self.head_dim, -1)

        attn_scores = torch.matmul(theta_x, phi_x) / (self.head_dim ** 0.5)
        attn_scores = F.softmax(attn_scores, dim=-1)

        y = torch.matmul(attn_scores, g_x)
        y = y.permute(0, 1, 3, 2).contiguous()
        y = y.view(batch_size, self.inter_channels, height, width)

        # Apply CCA block
        y = self.cca(y.unsqueeze(1)).squeeze(1)

        W_y = self.W(y)

        return W_y

class NonLocalSelfAttention(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, num_heads=8, sub_sample=True):
        super(NonLocalSelfAttention, self).__init__(in_channels, inter_channels=inter_channels, num_heads=num_heads, dimension=2, sub_sample=sub_sample)
