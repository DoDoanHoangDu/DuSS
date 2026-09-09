from torch.autograd import Function
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
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
    module = cupy.RawModule(code=code, options=("--std=c++11"),name_expressions=(kernel_name,))
    return module.get_function(kernel_name)


# -----------------------------------------------------------------------------
# CUDA utilities
# -----------------------------------------------------------------------------

kernel_loop = r"""
#define CUDA_KERNEL_LOOP(i, n)                                  \
  for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (long long)blockDim.x * gridDim.x)
"""


def GET_BLOCKS(N):
    return min((N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS, 2147483647)


# -----------------------------------------------------------------------------
# Forward kernel
#
# Important:
#   - half/bfloat16 are supported
#   - arithmetic accumulates in FP32
#   - output is stored back in the original low-precision dtype
# -----------------------------------------------------------------------------

_idynamic_kernel = kernel_loop + r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C"
__global__ void idynamic_forward_kernel(
    const ${Dtype}* __restrict__ bottom_data,
    const ${Dtype}* __restrict__ weight_data,
    ${Dtype}* __restrict__ top_data) {

    CUDA_KERNEL_LOOP(index, ${nthreads}) {
        const long long n = index / (${channels} * ${top_height} * ${top_width});
        const long long c = (index / (${top_height} * ${top_width})) % ${channels};
        const long long h = (index / ${top_width}) % ${top_height};
        const long long w = index % ${top_width};
        const int g = c / ${channels_per_group};

        ${AccumType} value = (${AccumType})0;
        #pragma unroll
        for (int kh = 0; kh < ${kernel_h}; ++kh) {
            const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
            if (h_in < 0 || h_in >= ${bottom_height})
                continue;

            #pragma unroll
            for (int kw = 0; kw < ${kernel_w}; ++kw) {
                const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
                if (w_in < 0 || w_in >= ${bottom_width})
                    continue;

                const long long offset = ((n * ${channels} + c) * ${bottom_height} + h_in) * ${bottom_width} + w_in;

                const long long offset_weight =
                    ((((n * ${groups} + g) 
                        * ${kernel_h} + kh)
                        * ${kernel_w} + kw)
                        * ${top_height} + h)
                        * ${top_width} + w;

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
__global__ void idynamic_backward_grad_input_kernel(
    const ${Dtype}* __restrict__ top_diff,
    const ${Dtype}* __restrict__ weight_data,
    ${Dtype}* __restrict__ bottom_diff) {

    CUDA_KERNEL_LOOP(index, ${nthreads}) {
        const long long n = index / (${channels} * ${bottom_height} * ${bottom_width});
        const long long c = (index / (${bottom_height} * ${bottom_width})) % ${channels};
        const long long h = (index / ${bottom_width}) % ${bottom_height};
        const long long w = index % ${bottom_width};
        const int g = c / ${channels_per_group};

        ${AccumType} value = (${AccumType})0;
        #pragma unroll
        for (int kh = 0; kh < ${kernel_h}; ++kh) {
            const int h_out_s = h + ${pad_h} - kh * ${dilation_h};
            if (h_out_s < 0)
                continue;

            if ((h_out_s % ${stride_h}) != 0)
                continue;

            const int h_out = h_out_s / ${stride_h};
            if (h_out < 0 || h_out >= ${top_height})
                continue;

            #pragma unroll
            for (int kw = 0; kw < ${kernel_w}; ++kw) {
                const int w_out_s = w + ${pad_w} - kw * ${dilation_w};
                if (w_out_s < 0)
                    continue;

                if ((w_out_s % ${stride_w}) != 0)
                    continue;

                const int w_out = w_out_s / ${stride_w};
                if (w_out < 0 || w_out >= ${top_width})
                    continue;

                const long long offset = 
                    ((n * ${channels} + c) 
                        * ${top_height} + h_out) 
                        * ${top_width} + w_out;

                const long long offset_weight =
                    ((((n * ${groups} + g) 
                        * ${kernel_h} + kh)
                        * ${kernel_w} + kw)
                        * ${top_height} + h_out)
                        * ${top_width} + w_out;

                value += (${AccumType})weight_data[offset_weight] * (${AccumType})top_diff[offset];
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
__global__ void idynamic_backward_grad_weight_kernel(
    const ${Dtype}* __restrict__ top_diff,
    const ${Dtype}* __restrict__ bottom_data,
    ${Dtype}* __restrict__ buffer_data) {

    CUDA_KERNEL_LOOP(index, ${nthreads}) {
        const int w = index % ${top_width};
        const int h = (index / ${top_width}) % ${top_height};
        const int kw = (index / (${top_height} * ${top_width})) % ${kernel_w};
        const int kh = (index / (${kernel_w} * ${top_height} * ${top_width})) % ${kernel_h};
        const int g = (index / (${kernel_h} * ${kernel_w} * ${top_height} * ${top_width})) % ${groups};
        const long long n = (index / (${groups} * ${kernel_h} * ${kernel_w} * ${top_height} * ${top_width})) % ${num};
        const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
        const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
        if (h_in < 0 || h_in >= ${bottom_height} || w_in < 0 || w_in >= ${bottom_width}) {
            buffer_data[index] = (${Dtype})0;
            continue;
        }

        ${AccumType} value = (${AccumType})0;

        const int c_begin = g * ${channels_per_group};
        const int c_end = c_begin + ${channels_per_group};

        #pragma unroll
        for (int c = c_begin; c < c_end; ++c) {
            const long long top_offset =
                ((n * ${channels} + c) 
                    * ${top_height} + h)
                    * ${top_width} + w;

            const long long bottom_offset =
                ((n * ${channels} + c) 
                    * ${bottom_height} + h_in)
                    * ${bottom_width} + w_in;

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
        #
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
        weight_compute = (weight if weight.dtype == compute_dtype else weight.to(dtype=compute_dtype))

        # Kernel assumes contiguous NCHW / BCHW memory.
        if not input_compute.is_contiguous():
            input_compute = input_compute.contiguous()
        if not weight_compute.is_contiguous():
            weight_compute = weight_compute.contiguous()

        batch_size, channels, height, width = input_compute.shape
        kernel_h, kernel_w = weight_compute.shape[2:4]
        stride_h, stride_w = stride
        pad_h, pad_w = padding
        dilation_h, dilation_w = dilation
        groups = weight_compute.size(1)
        if channels % groups != 0:
            raise ValueError(f"channels ({channels}) must be divisible by groups ({groups})")

        channels_per_group = channels // groups
        output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
        output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
        if output_h <= 0 or output_w <= 0:
            raise ValueError(f"Invalid output size: {(output_h, output_w)}")

        output = torch.empty(batch_size, channels, output_h, output_w, device=input_compute.device, dtype=compute_dtype)
        nthreads = output.numel()
        opt = dict(
            Dtype=Dtype(output),
            AccumType=AccumDtype(output),
            nthreads=nthreads,
            num=batch_size,
            channels=channels,
            groups=groups,
            channels_per_group=channels_per_group,
            bottom_height=height,
            bottom_width=width,
            top_height=output_h,
            top_width=output_w,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_h=stride_h,
            stride_w=stride_w,
            dilation_h=dilation_h,
            dilation_w=dilation_w,
            pad_h=pad_h,
            pad_w=pad_w,
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
        kernel_h, kernel_w = weight_compute.shape[2:4]
        output_h, output_w = grad_output.shape[2:]
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
            bottom_height=height,
            bottom_width=width,
            top_height=output_h,
            top_width=output_w,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_h=stride[0],
            stride_w=stride[1],
            dilation_h=dilation[0],
            dilation_w=dilation[1],
            pad_h=padding[0],
            pad_w=padding[1],
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

def _idynamic_cuda(input, weight, bias=None, stride=1, padding=0, dilation=1):
    if not input.is_cuda:
        raise NotImplementedError("CPU implementation is not provided")
    if input.size(0) != weight.size(0):
        raise ValueError(f"Batch mismatch: input={input.size(0)}, weight={weight.size(0)}")

    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    # The original assertion was stricter than necessary.
    # Validate the generated-weight spatial layout instead.
    expected_h = (input.size(-2) + 2 * padding[0] - (dilation[0] * (weight.size(2) - 1) + 1)) // stride[0] + 1
    expected_w = (input.size(-1) + 2 * padding[1] - (dilation[1] * (weight.size(3) - 1) + 1)) // stride[1] + 1
    if expected_h <= 0 or expected_w <= 0:
        raise ValueError(f"Invalid output shape: {(expected_h, expected_w)}")

    out = _idynamic.apply(input, weight, stride, padding, dilation)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
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
        if hidden < 1:
            raise ValueError(f"channels={channels} produces hidden={hidden}; channels must be >= 4")

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