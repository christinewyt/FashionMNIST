import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from typing import Union

##https://github.com/cvlab-yonsei/EWGS

class EWGS_discretizer(torch.autograd.Function):
    """
    x_in: continuous inputs within the range of [0,1]
    num_levels: number of discrete levels
    scaling_factor: backward scaling factor
    x_out: discretized version of x_in within the range of [0,1]
    """
    @staticmethod
    def forward(ctx, x_in, num_levels, scaling_factor):
        x = x_in * (num_levels - 1)
        x = torch.round(x)
        x_out = x / (num_levels - 1)
        
        ctx._scaling_factor = scaling_factor
        ctx.save_for_backward(x_in-x_out)
        return x_out
    @staticmethod
    def backward(ctx, g):
        diff = ctx.saved_tensors[0]
        delta = ctx._scaling_factor
        scale = 1 + delta * torch.sign(g)*diff
        return g * scale, None, None

class STE_discretizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in, num_levels):
        x = x_in * (num_levels - 1)
        x = torch.round(x)
        x_out = x / (num_levels - 1)
        return x_out
    @staticmethod
    def backward(ctx, g):
        return g, None
    
class Floor_divide(torch.autograd.Function):
    """
        Override floor division
    """
    @staticmethod
    def forward(ctx, x_in, divider):
        tmp = 1.0*(x_in >= divider)
        ctx.save_for_backward(tmp)
        return tmp
    @staticmethod
    def backward(ctx, g):
        tmp = ctx.saved_tensors[0]
        return tmp * g, None