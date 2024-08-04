import torch
import torch.nn as nn

class CCA(nn.Module):
    def __init__(self, channel, kernel_sizes=[3, 3], planes=[640, 640]):
        super(CCA, self).__init__()
        num_layers = len(kernel_sizes)
        nn_modules = []

        for i in range(num_layers):
            ch_in = channel if i == 0 else planes[i - 1]
            ch_out = planes[i]
            k_size = kernel_sizes[i]
            nn_modules.append(SepConv1d(in_planes=ch_in, out_planes=ch_out, ksize=k_size))
            if i != num_layers - 1:
                nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)

    def forward(self, x):
        # Apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
        # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
        x = self.conv(x) + self.conv(x.permute(0, 1, 2).flip(dims=[2]))
        return x

class SepConv1d(nn.Module):
    """ Approximates 3 x 3 kernels via two subsequent 3 x 1 and 1 x 3 """
    def __init__(self, in_planes, out_planes, ksize=3, do_padding=True, bias=False):
        super(SepConv1d, self).__init__()
        self.isproj = False
        padding = ksize // 2 if do_padding else 0

        if in_planes != out_planes:
            self.isproj = True
            self.proj = nn.Sequential(
                nn.Conv1d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=bias),
                nn.BatchNorm1d(out_planes)
            )

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_planes, out_channels=in_planes, kernel_size=ksize, padding=padding, bias=bias),
            nn.BatchNorm1d(in_planes)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=in_planes, out_channels=in_planes, kernel_size=ksize, padding=padding, bias=bias),
            nn.BatchNorm1d(in_planes)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, seq_len = x.shape
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv1(x)
        if self.isproj:
            x = self.proj(x)
        return x
