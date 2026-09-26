import math
import torch
import torch.nn as nn
from torch.func import rearrange

TLC_KERNEL=48

# ---------------------------------------------------------------------------------------------------------------------
# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# ---------------------------------------------------------------------------------------------------------------------
# FFN
class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor = 1, bias = False):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, padding=1, groups=hidden_features*2, bias=bias)
        self.GELU = nn.GELU()
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.GELU(x1) * x2
        x = self.project_out(x)
        return x

# ---------------------------------------------------------------------------------------------------------------------
# IDynamicDWConvBlock
"""class DynamicConv(nn.Module):
    def __init__(self, channels, kernel_size, group_channels, bias=True):
        super().__init__()
        assert channels % group_channels == 0
        self.k = kernel_size
        self.groups = channels // group_channels
        hidden = channels // 4

        self.hypernet = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=bias),
            nn.Conv2d(hidden, hidden, kernel_size, padding=kernel_size // 2, groups=hidden, bias=bias),
            nn.Conv2d(hidden, self.groups * kernel_size**2, 1, bias=bias),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        G, K = self.groups, self.k
        weight = self.hypernet(x).view(B, G, K*K, H, W)
        patches = nn.functional.unfold(x, K, padding=K//2)
        patches = patches.view(B, G, C//G, K*K, H, W)
        return torch.einsum('bgkhw,bgckhw->bgchw', weight, patches).reshape(B, C, H, W)"""
from models.dynamic_conv import DynamicConv

class DynamicConvBlock(nn.Module):
    def __init__(self, dim, kernel_size, group_channels, squeeze=2):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim//squeeze, 1, bias=False)
        self.conv = DynamicConv(dim//squeeze, kernel_size, group_channels)
        self.conv1 = nn.Conv2d(dim//squeeze, dim, 1, bias=False)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv(x)
        x = self.conv1(x)
        return x

# ---------------------------------------------------------------------------------------------------------------------
# Multi-DConv Head Transposed Self-Attention (MDTA)
class SparseAttention(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(SparseAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU()

    def attention_forward(self, qkv):
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.act(attn)
        out = (attn @ v)
        return out

    def forward(self, x, tlc_flag = True):
        qkv = self.qkv_dwconv(self.qkv(x))
        if tlc_flag: 
            qkv, idxes, original_size = self.grids(qkv)  # convert to local windows
        _, _, h, w = qkv.shape
        out = self.attention_forward(qkv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        if tlc_flag:
            out = self.grids_inverse(out, idxes, original_size)  # reverse
        out = self.project_out(out)
        return out

    def grids(self, x):
        b, c, h, w = x.shape
        original_size = (b, c // 3, h, w)
        assert b == 1
        k = min(h, w, TLC_KERNEL)
        self.num_row = (h - 1) // k + 1
        self.num_col = (w - 1) // k + 1
        step_j = k if self.num_col == 1 else math.ceil((w - k) / (self.num_col - 1) - 1e-8)
        step_i = k if self.num_row == 1 else math.ceil((h - k) / (self.num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k >= h:
                i = h - k
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k >= w:
                    j = w - k
                    last_j = True
                parts.append(x[:, :, i:i + k, j:j + k])
                idxes.append((i, j))
                j = j + step_j
            i = i + step_i
        parts = torch.cat(parts, dim=0)
        return parts, idxes, original_size

    def grids_inverse(self, outs, idxes, original_size):
        b, c, h, w = original_size
        preds = torch.zeros(
            original_size,
            device=outs.device,
            dtype=outs.dtype
        )
        count_mt = torch.zeros(
            (b, 1, h, w),
            device=outs.device,
            dtype=outs.dtype
        )
        k = min(h, w, TLC_KERNEL)

        for cnt, each_idx in enumerate(idxes):
            i, j = each_idx
            preds[0, :, i:i + k, j:j + k] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k, j:j + k] += 1.

        #del outs
        #torch.cuda.empty_cache()
        return preds / count_mt

# ---------------------------------------------------------------------------------------------------------------------
class MHDLSA(nn.Module):
    def __init__(self, dim, kernel_size=7, group_channels=8):
        super(MHDLSA, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.IDynamicDWConv = DynamicConvBlock(dim, kernel_size, group_channels)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x):
        x = self.IDynamicDWConv(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x

class SparseGSA(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(SparseGSA, self).__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = SparseAttention(dim, num_heads)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x, tlc_flag = True):
        x = self.attn(self.norm1(x), tlc_flag) + x
        x = self.ffn(self.norm2(x)) + x
        return x

# ---------------------------------------------------------------------------------------------------------------------
# BuildBlocks
class RHDTG(nn.Module):
    def __init__(self, dim, blocks=5):
        super(RHDTG, self).__init__()
        body = nn.ModuleList()
        for _ in range(blocks):
            body.append(MHDLSA(dim))
            body.append(SparseGSA(dim))
        body.append(nn.Conv2d(dim, dim, 3, padding=1))
        self.body = body

    def res_forward(self, x, tlc_flag = True):
        for block in self.body:
            if isinstance(block, SparseGSA):
                x = block(x, tlc_flag)
            else:
                x = block(x)
        return x

    def forward(self, x, tlc_flag = True):
        return x + self.res_forward(x, tlc_flag)

# ---------------------------------------------------------------------------------------------------------------------
class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_filters):
        m = []
        m.append(nn.Conv2d(num_filters, 3 * (scale**2), kernel_size=3, padding=1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

# Traditional Upsample from SwinIR EDSR RCAN
class Upsample(nn.Sequential):
    def __init__(self, scale, num_filters):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_filters, 4 * num_filters, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_filters, 9 * num_filters, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported.')
        super(Upsample, self).__init__(*m)

# ---------------------------------------------------------------------------------------------------------------------
# Network
class DLGSANet(nn.Module):
    def __init__(self, dim=64, groups=6, scale=4, upsampler = "pixelshuffledirect"):
        super(DLGSANet, self).__init__()
        self.register_buffer('mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.first_part = nn.Conv2d(3, dim, kernel_size=3, padding=1, bias=False)

        # ------------------------- deep feature extraction ------------------------- #
        self.body = nn.ModuleList()
        for _ in range(groups):
            self.body.append(RHDTG(dim))
        self.body.append(nn.Conv2d(dim, dim, 3, padding=1))

        # ------------------------- upsampling ------------------------- #

        if upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(scale, dim)
        elif upsampler == 'pixelshuffle':
            self.upsample = nn.Sequential(
                Upsample(scale, dim),
                nn.Conv2d(dim, 3, kernel_size=3, padding=1)
            )
        else:
            self.upsample = nn.Conv2d(dim, 3, kernel_size=3, padding=1)

    def deep_feature_extraction(self, x, tlc_flag = True):
        for block in self.body:
            if isinstance(block, RHDTG):
                x = block(x, tlc_flag)
            else:
                x = block(x)
        return x

    def forward(self, x, tlc_flag = True):
        x = x - self.mean
        x = self.first_part(x)
        x = self.deep_feature_extraction(x, tlc_flag) + x
        x = self.upsample(x)
        x = x + self.mean
        return x