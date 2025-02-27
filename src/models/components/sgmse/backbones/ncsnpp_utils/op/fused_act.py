import os

import torch
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F
from torch.utils.cpp_extension import load


# Function to check if CUDA is available and the GPU is NVIDIA
def is_nvidia_gpu():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        return "NVIDIA" in device_name
    return False


# Conditionally load the fused extension
if is_nvidia_gpu():
    module_path = os.path.dirname(__file__)
    fused = load(
        "fused",
        sources=[
            os.path.join(module_path, "fused_bias_act.cpp"),
            os.path.join(module_path, "fused_bias_act_kernel.cu"),
        ],
    )


class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        if is_nvidia_gpu():
            ctx.save_for_backward(out)
            ctx.negative_slope = negative_slope
            ctx.scale = scale

            empty = grad_output.new_empty(0)

            grad_input = fused.fused_bias_act(grad_output, empty, out, 3, 1, negative_slope, scale)

            dim = [0]

            if grad_input.ndim > 2:
                dim += list(range(2, grad_input.ndim))

            grad_bias = grad_input.sum(dim).detach()

            return grad_input, grad_bias
        else:
            raise NotImplementedError(
                "FusedLeakyReLUFunctionBackward is not implemented for non-NVIDIA GPUs"
            )

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        if is_nvidia_gpu():
            (out,) = ctx.saved_tensors
            gradgrad_out = fused.fused_bias_act(
                gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
            )

            return gradgrad_out, None, None, None
        else:
            raise NotImplementedError(
                "FusedLeakyReLUFunctionBackward is not implemented for non-NVIDIA GPUs"
            )


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        if is_nvidia_gpu():
            empty = input.new_empty(0)
            out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
            ctx.save_for_backward(out)
            ctx.negative_slope = negative_slope
            ctx.scale = scale

            return out
        else:
            raise NotImplementedError(
                "FusedLeakyReLUFunction is not implemented for non-NVIDIA GPUs"
            )

    @staticmethod
    def backward(ctx, grad_output):
        if is_nvidia_gpu():
            (out,) = ctx.saved_tensors

            grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
                grad_output, out, ctx.negative_slope, ctx.scale
            )

            return grad_input, grad_bias, None, None
        else:
            raise NotImplementedError(
                "FusedLeakyReLUFunction is not implemented for non-NVIDIA GPUs"
            )


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2**0.5):
    if input.device.type == "cpu" or not is_nvidia_gpu():
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope
            )
            * scale
        )
    else:
        return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
