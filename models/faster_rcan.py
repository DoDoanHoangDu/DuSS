import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_rate=16):
        super(ChannelAttention, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels // reduction_rate, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels // reduction_rate, out_channels=channels, kernel_size=1, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_feat = self.squeeze(x)
        avg_out = self.excitation(avg_feat)
        attention = self.sigmoid(avg_out)
        return attention * x

class ResidualBlockAttention(nn.Module):
    def __init__(self, num_filters=256):
        super(ResidualBlockAttention, self).__init__()
        self.scale = 1
        self.block = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.PReLU(num_filters),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
        )
        self.attention = ChannelAttention(num_filters)
    def forward(self,x):
        res = self.block(x)
        res = self.attention(res)
        return x + self.scale * res

class ResidualGroupAttention(nn.Module):
    def __init__(self, num_filters, num_blocks = 4):
        super(ResidualGroupAttention, self).__init__()
        layers = [ResidualBlockAttention(num_filters) for _ in range(num_blocks)]
        layers.append(nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=True))
        self.body = nn.Sequential(*layers)
    def forward(self, x):
        return x + self.body(x)

class Upsampler(nn.Sequential):
    def __init__(self, num_filters = 128, ratio = 4):
        layers = []
        if ratio == 2 or ratio == 4:
            layers.append(nn.Conv2d(num_filters, num_filters * 4, kernel_size=3, stride=1, padding=1, bias=True))
            layers.append(nn.PixelShuffle(2))
            if ratio == 4:
                layers.append(nn.Conv2d(num_filters, num_filters * 4, kernel_size=3, stride=1, padding=1, bias=True))
                layers.append(nn.PixelShuffle(2))
        elif ratio == 3:
            layers.append(nn.Conv2d(num_filters, num_filters * 9, kernel_size=3, stride=1, padding=1, bias=True))
            layers.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*layers)

class FasterRCAN(nn.Module):
    def __init__(self,big = 128, small = 64, num_groups = 2, ratio = 4):
        super(FasterRCAN, self).__init__()
        self.scale = 1
        self.first_part = nn.Sequential(
            nn.Conv2d(3, big, kernel_size=3, padding=1),
            nn.PReLU(big),
        )
        #Shrinking
        self.shrink = nn.Sequential(
            nn.Conv2d(big, small, kernel_size=1),
            nn.PReLU(small)
        )
        #Mapping
        mid_part_layers = [ResidualGroupAttention(small) for _ in range(num_groups)]
        mid_part_layers.append(nn.Conv2d(small, small, kernel_size=3, padding=1, bias=True))
        self.mid_part = nn.Sequential(*mid_part_layers)

        #Expanding
        self.expand = nn.Sequential(
            nn.Conv2d(small, big, kernel_size=1),
            nn.PReLU(big),
        )
        self.last_part = nn.Sequential(
            Upsampler(big,ratio),
            nn.Conv2d(big, 3, kernel_size=3, stride=1, padding=1, bias=True)
        )
    def forward(self, x):
        x = self.first_part(x)
        x1 = self.shrink(x)
        x1 = self.mid_part(x1)
        x1 = self.expand(x1)
        x = x + x1 * self.scale
        x = self.last_part(x)
        return x