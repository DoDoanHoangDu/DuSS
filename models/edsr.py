import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, num_filters=256, scale = None):
        super(ResidualBlock,self).__init__()
        if scale is not None:
            self.scale = nn.Parameter(torch.tensor(scale,dtype=torch.float32))
            self.scale.requires_grad = False

        else:
            self.scale = nn.Parameter(torch.tensor(0.1,dtype=torch.float32))
        self.block = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
        )
    def forward(self,x):
        return self.block(x) * self.scale + x

class Upsampler(nn.Sequential):
    def __init__(self, num_filters = 256, ratio = 2):
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

class EDSR(nn.Module):
    def __init__(self, num_filters = 128, num_blocks = 8, ratio = 4, scale = 1):
        super(EDSR, self).__init__()
        if scale:
            self.scale = nn.Parameter(torch.tensor(scale,dtype=torch.float32))
            self.scale.requires_grad = False
            self.block_scale = 0.1
        else:
            self.scale = nn.Parameter(torch.tensor(1,dtype=torch.float32))
            self.block_scale = None
        self.first_part = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
        )

        mid_part_layers = [ResidualBlock(num_filters,self.block_scale) for _ in range(num_blocks)]
        #Mapping
        self.mid_part = nn.Sequential(*mid_part_layers)
        self.body_tail = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1,bias = True)

        self.last_part = nn.Sequential(
            Upsampler(num_filters,ratio),
            nn.Conv2d(num_filters, 3, kernel_size=3, stride=1, padding=1, bias=True)
        )
    def forward(self, x):
        x = self.first_part(x)
        res = self.mid_part(x)
        res = self.body_tail(res)
        x = x + res * self.scale
        x = self.last_part(x)
        return x