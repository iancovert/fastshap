import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiConv(nn.Module):
    '''(convolution => [BN] => ReLU) * n'''

    def __init__(self, in_channels, out_channels, mid_channels=None,
                 num_convs=2, batchnorm=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if batchnorm:
            # Input conv.
            module_list = [
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)]

            # Middle convs.
            for _ in range(num_convs - 2):
                module_list = module_list + [
                    nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                              padding=1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True)]

            # Output conv.
            module_list = module_list + [
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)]

        else:
            # Input conv.
            module_list = [
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)]

            # Middle convs.
            for _ in range(num_convs - 2):
                module_list = module_list + [
                    nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                              padding=1),
                    nn.ReLU(inplace=True)]

            # Output conv.
            module_list = module_list + [
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)]

        # Set up sequential.
        self.multi_conv = nn.Sequential(*module_list)

    def forward(self, x):
        return self.multi_conv(x)


class Down(nn.Module):
    '''
    Downscaling with maxpool then multiconv.
    Adapted from https://github.com/milesial/Pytorch-UNet
    '''

    def __init__(self, in_channels, out_channels, num_convs=2, batchnorm=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            MultiConv(in_channels, out_channels, num_convs=num_convs,
                      batchnorm=batchnorm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    '''
    Upscaling then multiconv.
    Adapted from https://github.com/milesial/Pytorch-UNet
    '''

    def __init__(self, in_channels, out_channels, num_convs=2, bilinear=True,
                 batchnorm=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                                  align_corners=True)
            self.conv = MultiConv(in_channels, out_channels, in_channels // 2,
                                  num_convs, batchnorm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                         kernel_size=2, stride=2)
            self.conv = MultiConv(in_channels, out_channels,
                                  num_convs=num_convs, batchnorm=batchnorm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self,
                 n_classes,
                 num_down,
                 num_up,
                 in_channels=3,
                 base_channels=64,
                 num_convs=2,
                 batchnorm=True,
                 bilinear=True):
        super().__init__()
        assert num_down >= num_up

        # Input conv.
        self.inc = MultiConv(in_channels, base_channels, num_convs=num_convs,
                             batchnorm=batchnorm)

        # Downsampling layers.
        down_layers = []
        channels = base_channels
        out_channels = 2 * channels
        for _ in range(num_down - 1):
            down_layers.append(
                Down(channels, out_channels, num_convs, batchnorm))
            channels = out_channels
            out_channels *= 2

        # Last down layer.
        factor = 2 if bilinear else 1
        down_layers.append(
            Down(channels, out_channels // factor, num_convs, batchnorm))
        self.down_layers = nn.ModuleList(down_layers)

        # Upsampling layers.
        up_layers = []
        channels *= 2
        out_channels = channels // 2
        for _ in range(num_up - 1):
            up_layers.append(
                Up(channels, out_channels // factor, num_convs, bilinear,
                   batchnorm))
            channels = out_channels
            out_channels = channels // 2

        # Last up layer.
        up_layers.append(
            Up(channels, out_channels, num_convs, bilinear, batchnorm))
        self.up_layers = nn.ModuleList(up_layers)

        # Output layer.
        self.outc = OutConv(out_channels, n_classes)

    def forward(self, x):
        # Input conv.
        x = self.inc(x)

        # Apply downsampling layers.
        x_list = []
        for down in self.down_layers:
            x = down(x)
            x_list.append(x)

        # Apply upsampling layers.
        for i, up in enumerate(self.up_layers):
            residual_x = x_list[-(i + 2)]
            x = up(x, residual_x)

        # Output.
        logits = self.outc(x)
        return logits
