from torch.autograd import Function
import torch
import torch.nn as nn
from collections import namedtuple
from string import Template
import cupy

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CUDA_NUM_THREADS = 256
Stream = namedtuple("Stream", ["ptr"])

# -----------------------------------------------------------------------------
# Dtype helpers
# -----------------------------------------------------------------------------

def Dtype(t):
    """Map torch dtype -> CUDA C++ scalar type."""
    if t.dtype == torch.float16:
        return "half"
    elif t.dtype == torch.bfloat16:
        return "nv_bfloat16"
    elif t.dtype == torch.float32:
        return "float"
    elif t.dtype == torch.float64:
        return "double"
    else:
        raise TypeError(f"Unsupported dtype: {t.dtype}")

def AccumDtype(t):
    """
    Accumulate low-precision inputs in FP32.
    Keep FP64 kernels in FP64.
    """
    if t.dtype == torch.float64:
        return "double"
    elif t.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return "float"
    else:
        raise TypeError(f"Unsupported dtype: {t.dtype}")

# -----------------------------------------------------------------------------
# CuPy kernel loading
# -----------------------------------------------------------------------------

@cupy._util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    module = cupy.RawModule(code=code, options=("--std=c++11",),name_expressions=(kernel_name,))
    return module.get_function(kernel_name)

# -----------------------------------------------------------------------------
# CUDA utilities
# -----------------------------------------------------------------------------

kernel_loop = r"""
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
"""

def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS

# -----------------------------------------------------------------------------
# Forward kernel
#   - half/bfloat16 are supported
#   - arithmetic accumulates in FP32
#   - output is stored back in the original low-precision dtype
# -----------------------------------------------------------------------------

_idynamic_kernel = kernel_loop + r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C"
__global__ void idynamic_forward_kernel(const ${Dtype}* __restrict__ bottom_data, const ${Dtype}* __restrict__ weight_data, ${Dtype}* __restrict__ top_data) {
    CUDA_KERNEL_LOOP(index, ${nthreads}) {
        const int HW = ${height} * ${width};
        const int CHW = ${channels} * HW;
        const int KHW = ${kernel} * HW;

        const int n = index / CHW;
        const int c = (index / HW) % ${channels};
        const int h = (index % HW) / ${width};
        const int w = index % ${width};
        const int g = c / ${channels_per_group};
        ${AccumType} value = (${AccumType})0;

        const int h_base = -${pad} + h * ${stride};
        const int w_base = -${pad} + w * ${stride};
        const int input_base = (n * ${channels} + c) * HW;
        const int weight_base = (n * ${groups} + g) * ${kernel} * KHW + h * ${width} + w;
        #pragma unroll
        for (int kh = 0; kh < ${kernel}; ++kh) {
            const int h_in = h_base + kh * ${dilation};
            if (h_in < 0 || h_in >= ${height})
                continue;

            const int input_row_base = input_base + h_in * ${width};
            const int weight_row_base = weight_base + kh * KHW;

            #pragma unroll
            for (int kw = 0; kw < ${kernel}; ++kw) {
                const int w_in = w_base + kw * ${dilation};
                if (w_in < 0 || w_in >= ${width})
                    continue;

                const int offset = input_row_base + w_in;
                const int offset_weight = weight_row_base + kw * HW;
                value += (${AccumType})weight_data[offset_weight] * (${AccumType})bottom_data[offset];
            }
        }
        top_data[index] = (${Dtype})value;
    }
}
"""

# -----------------------------------------------------------------------------
# Backward: grad input
# -----------------------------------------------------------------------------

_idynamic_kernel_backward_grad_input = kernel_loop + r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C"
__global__ void idynamic_backward_grad_input_kernel(const ${Dtype}* __restrict__ top_diff, const ${Dtype}* __restrict__ weight_data, ${Dtype}* __restrict__ bottom_diff) {
    CUDA_KERNEL_LOOP(index, ${nthreads}) {
        const int HW = ${height} * ${width};
        const int CHW = ${channels} * HW;

        const int n = index / CHW;
        const int c = (index / HW) % ${channels};
        const int h = (index % HW) / ${width};
        const int w = index % ${width};
        const int g = c / ${channels_per_group};

        ${AccumType} value = (${AccumType})0;
        const int top_base = (n * ${channels} + c) * HW;
        const int weight_base = (n * ${groups} + g) * ${kernel} * ${kernel} * HW;
        #pragma unroll
        for (int kh = 0; kh < ${kernel}; ++kh) {
            const int h_out_s = h + ${pad} - kh * ${dilation};
            if (h_out_s < 0)
                continue;
            if ((h_out_s % ${stride}) != 0)
                continue;

            const int h_out = h_out_s / ${stride};
            if (h_out < 0 || h_out >= ${height})
                continue;

            const int top_row = top_base + h_out * ${width};
            const int weight_row = weight_base + kh * ${kernel} * HW + h_out * ${width};

            #pragma unroll
            for (int kw = 0; kw < ${kernel}; ++kw) {
                const int w_out_s = w + ${pad} - kw * ${dilation};
                if (w_out_s < 0)
                    continue;
                if ((w_out_s % ${stride}) != 0)
                    continue;

                const int w_out = w_out_s / ${stride};
                if (w_out < 0 || w_out >= ${width})
                    continue;

                const int top_offset = top_row + w_out;
                const int weight_offset = weight_row + kw * HW + w_out;
                value += (${AccumType})weight_data[weight_offset] * (${AccumType})top_diff[top_offset];
            }
        }

        bottom_diff[index] = (${Dtype})value;
    }
}
"""

# -----------------------------------------------------------------------------
# Backward: grad weight
# -----------------------------------------------------------------------------

_idynamic_kernel_backward_grad_weight = kernel_loop + r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C"
__global__ void idynamic_backward_grad_weight_kernel(const ${Dtype}* __restrict__ top_diff, const ${Dtype}* __restrict__ bottom_data, ${Dtype}* __restrict__ buffer_data) {
    CUDA_KERNEL_LOOP(index, ${nthreads}) {
        const int w = index % ${width};
        const int h = (index / ${width}) % ${height};
        const int kw = (index / (${height} * ${width})) % ${kernel};
        const int kh = (index / (${kernel} * ${height} * ${width})) % ${kernel};
        const int g = (index / (${kernel} * ${kernel} * ${height} * ${width})) % ${groups};
        const int n = (index / (${groups} * ${kernel} * ${kernel} * ${height} * ${width})) % ${num};
        const int h_in = -${pad} + h * ${stride} + kh * ${dilation};
        const int w_in = -${pad} + w * ${stride} + kw * ${dilation};
        if (h_in < 0 || h_in >= ${height} || w_in < 0 || w_in >= ${width}) {
            buffer_data[index] = (${Dtype})0;
            continue;
        }

        ${AccumType} value = (${AccumType})0;
        const int c_begin = g * ${channels_per_group};
        const int c_end = c_begin + ${channels_per_group};

        #pragma unroll
        for (int c = c_begin; c < c_end; ++c) {
            const int top_offset = ((n * ${channels} + c) * ${height} + h) * ${width} + w;
            const int bottom_offset =((n * ${channels} + c) * ${height} + h_in) * ${width} + w_in;
            value += (${AccumType})top_diff[top_offset] * (${AccumType})bottom_data[bottom_offset];
        }
        buffer_data[index] = (${Dtype})value;
    }
}
"""

# -----------------------------------------------------------------------------
# Autograd function
# -----------------------------------------------------------------------------

class _idynamic(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, input, weight, stride, padding, dilation):
        if not input.is_cuda or not weight.is_cuda:
            raise RuntimeError("idynamic requires CUDA tensors")
        if input.dim() != 4:
            raise ValueError(f"input must be 4D [N,C,H,W], got {tuple(input.shape)}")
        if weight.dim() != 6:
            raise ValueError(f"weight must be 6D [N,G,K,K,H,W], got {tuple(weight.shape)}")
        if not input.is_floating_point() or not weight.is_floating_point():
            raise TypeError("input and weight must be floating point tensors")

        # ------------------------------------------------------------------
        # Choose the compute dtype.
        # Inside autocast, if both tensors are still FP32, use the active
        # autocast dtype (FP16 or BF16). Otherwise use the weight dtype,
        # which is normally already autocast by the hypernetwork.
        # ------------------------------------------------------------------
        if torch.is_autocast_enabled("cuda"):
            autocast_dtype = torch.get_autocast_dtype("cuda")
            if input.dtype == torch.float32 and weight.dtype == torch.float32:
                compute_dtype = autocast_dtype
            else:
                compute_dtype = weight.dtype
        else:
            compute_dtype = weight.dtype
        if compute_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TypeError(f"Unsupported compute dtype: {compute_dtype}")

        # ------------------------------------------------------------------
        # Make the two CUDA inputs share one dtype.
        #
        # This is important for AMP:
        #   x may remain FP32
        #   hypernet(x) may be FP16/BF16
        #
        # The CUDA kernel uses one concrete pointer type for both.
        # ------------------------------------------------------------------
        input_compute = input if input.dtype == compute_dtype else input.to(dtype=compute_dtype)
        weight_compute = weight if weight.dtype == compute_dtype else weight.to(dtype=compute_dtype)

        # Kernel assumes contiguous NCHW / BCHW memory.
        if not input_compute.is_contiguous():
            input_compute = input_compute.contiguous()
        if not weight_compute.is_contiguous():
            weight_compute = weight_compute.contiguous()

        batch_size, channels, height, width = input_compute.shape
        kernel = weight_compute.shape[2]
        groups = weight_compute.size(1)
        channels_per_group = channels // groups
        output = torch.empty(batch_size, channels, height, width, device=input_compute.device, dtype=compute_dtype)
        nthreads = output.numel()
        opt = dict(
            Dtype=Dtype(output),
            AccumType=AccumDtype(output),
            nthreads=nthreads,
            num=batch_size,
            channels=channels,
            groups=groups,
            channels_per_group=channels_per_group,
            height=height,
            width=width,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            pad=padding,
        )
        with torch.cuda.device_of(input_compute):
            f = load_kernel("idynamic_forward_kernel", _idynamic_kernel, **opt)
            f(
                block=(CUDA_NUM_THREADS, 1, 1),
                grid=(GET_BLOCKS(nthreads), 1, 1),
                args=[input_compute.data_ptr(), weight_compute.data_ptr(), output.data_ptr()],
                stream=Stream(ptr=torch.cuda.current_stream().cuda_stream)
            )
        # Save the tensors actually consumed by the CUDA kernel.
        # They may be FP16/BF16 even when the original input was FP32.
        ctx.save_for_backward(input_compute, weight_compute)
        ctx.input_dtype = input.dtype
        ctx.weight_dtype = weight.dtype
        ctx.compute_dtype = compute_dtype
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        input_compute, weight_compute = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation

        # The CUDA kernels require one dtype for all operands.
        if grad_output.dtype != ctx.compute_dtype:
            grad_output = grad_output.to(dtype=ctx.compute_dtype)
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        batch_size, channels, height, width = input_compute.shape
        kernel = weight_compute.shape[2]
        groups = weight_compute.size(1)
        channels_per_group = channels // groups
        grad_input = None
        grad_weight = None
        opt = dict(
            Dtype=Dtype(grad_output),
            AccumType=AccumDtype(grad_output),
            num=batch_size,
            channels=channels,
            groups=groups,
            channels_per_group=channels_per_group,
            height=height,
            width=width,
            kernel=kernel,
            stride=stride,
            dilation=dilation,
            pad=padding,
        )
        stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        with torch.cuda.device_of(input_compute):
            # --------------------------------------------------------------
            # dL/dInput
            # --------------------------------------------------------------
            if ctx.needs_input_grad[0]:
                grad_input_compute = torch.empty_like(input_compute)
                nthreads = grad_input_compute.numel()
                opt["nthreads"] = nthreads
                f = load_kernel("idynamic_backward_grad_input_kernel", _idynamic_kernel_backward_grad_input, **opt)
                f(
                    block=(CUDA_NUM_THREADS, 1, 1),
                    grid=(GET_BLOCKS(nthreads), 1, 1),
                    args=[grad_output.data_ptr(), weight_compute.data_ptr(), grad_input_compute.data_ptr()],
                    stream=stream
                )
                # Return gradient in the dtype of the original input.
                if grad_input_compute.dtype != ctx.input_dtype:
                    grad_input = grad_input_compute.to(dtype=ctx.input_dtype)
                else:
                    grad_input = grad_input_compute

            # --------------------------------------------------------------
            # dL/dWeight
            # --------------------------------------------------------------
            if ctx.needs_input_grad[1]:
                grad_weight_compute = torch.empty_like(weight_compute)
                nthreads = grad_weight_compute.numel()
                opt["nthreads"] = nthreads
                f = load_kernel("idynamic_backward_grad_weight_kernel", _idynamic_kernel_backward_grad_weight, **opt)
                f(
                    block=(CUDA_NUM_THREADS, 1, 1),
                    grid=(GET_BLOCKS(nthreads), 1, 1),
                    args=[grad_output.data_ptr(), input_compute.data_ptr(), grad_weight_compute.data_ptr()],
                    stream=stream
                )
                # Return gradient in the dtype of the original weight.
                if grad_weight_compute.dtype != ctx.weight_dtype:
                    grad_weight = grad_weight_compute.to(dtype=ctx.weight_dtype)
                else:
                    grad_weight = grad_weight_compute
        return grad_input, grad_weight, None, None, None


# -----------------------------------------------------------------------------
# Python wrapper
# -----------------------------------------------------------------------------

def _idynamic_cuda(input, weight, stride=1, padding=0, dilation=1):
    if not input.is_cuda:
        raise NotImplementedError("CPU implementation is not provided")
    if input.size(0) != weight.size(0):
        raise ValueError(f"Batch mismatch: input={input.size(0)}, weight={weight.size(0)}")

    expected_h = (input.size(-2) + 2 * padding - (dilation * (weight.size(2) - 1) + 1)) // stride + 1
    expected_w = (input.size(-1) + 2 * padding - (dilation * (weight.size(3) - 1) + 1)) // stride + 1
    assert expected_h == input.size(-2) and expected_w == input.size(-1)

    out = _idynamic.apply(input, weight, stride, padding, dilation)
    return out

# -----------------------------------------------------------------------------
# Dynamic Conv
# -----------------------------------------------------------------------------
class DynamicConv(nn.Module):
    def __init__(self, channels, kernel_size, group_channels, bias=True):
        super().__init__()
        if channels % group_channels != 0:
            raise ValueError(f"channels ({channels}) must be divisible by group_channels ({group_channels})")

        self.kernel_size = kernel_size
        self.groups = channels // group_channels
        hidden = channels // 4
        self.hypernet = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=bias),
            nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=kernel_size // 2, groups=hidden, bias=bias),
            nn.Conv2d(hidden, kernel_size ** 2 * self.groups, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        # Under autocast, this normally becomes FP16/BF16.
        weight = self.hypernet(x)

        B, C, H, W = weight.shape
        G = self.groups
        K = self.kernel_size

        weight = weight.view(B, G, K, K, H, W)
        return _idynamic_cuda(x, weight, stride=1, padding=(K - 1) // 2)