import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, num_filters=256):
        super(ResidualBlock, self).__init__()
        self.scale = 0.1
        self.block = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.PReLU(num_filters),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
        )
    def forward(self,x):
        res = self.block(x)
        return x + self.scale * res

class ResidualGroup(nn.Module):
    def __init__(self, num_filters, num_blocks = 4):
        super(ResidualGroup, self).__init__()
        layers = [ResidualBlock(num_filters) for _ in range(num_blocks)]
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

class FasterRCN(nn.Module):
    def __init__(self,big = 128, small = 64, num_groups = 2, ratio = 4):
        super(FasterRCN, self).__init__()
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
        mid_part_layers = [ResidualGroup(small) for _ in range(num_groups)]
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