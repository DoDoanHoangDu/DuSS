import torch
import torch.nn as nn
import math


class FSRCNN(nn.Module):
    def __init__(self,big = 56, small = 12, num_mid_layers = 4, ratio = 4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(3, big, kernel_size=5, padding=2),
            nn.PReLU(big),
        )
        self.shrink = nn.Sequential(
            #Shrink
            nn.Conv2d(big, small, kernel_size=1),
            nn.PReLU(small)
        )

        self.mid_part = []
        for _ in range(num_mid_layers):
            self.mid_part.extend([nn.Conv2d(small, small, kernel_size=3, padding=1), nn.PReLU(small)])
        #Mapping
        self.mid_part = nn.Sequential(*self.mid_part)

        
        self.expand = nn.Sequential(
            #Expanding
            nn.Conv2d(small, big, kernel_size=1),
            nn.PReLU(big),
        )
        self.last_part = nn.Sequential(
            nn.ConvTranspose2d(big, 3, kernel_size=9, stride=ratio, padding=4,
                                            output_padding=ratio - 1),
        )
    def forward(self, x):
        x = self.first_part(x)
        x = self.shrink(x)
        x = self.mid_part(x)
        x = self.expand(x)
        x = self.last_part(x)
        return x