import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union
import numpy as np
import math
import copy
from quantization.quant_solver import EWGS_discretizer, STE_discretizer, Floor_divide
from hw_architecture.CB_param import gp

def rescale_weight(weight, w_min, w_max, target_min, target_max):
    """
        Rescale weights from [w_min, w_max] to target range [target_min, target_max]
        return: rescaled weight; scaling_factor = old_range/new_range 
    """
    old_range = w_max - w_min
    new_range = target_max - target_min
    scaling_factor = old_range / new_range
    w_rescale = (weight - w_min) / scaling_factor + target_min
    return w_rescale, scaling_factor

# ref. https://github.com/ricky40403/DSQ/blob/master/DSQConv.py#L18
class QIdentity(nn.Module):
    """
        Quant input to act_bits
        if signed, quant between [-1, 1]; o.w. quant between [0, 1]
    """
    def __init__(self, args, act_bits = 8, signed = True, **kwargs):
        super(QIdentity, self).__init__()
        self.signed = signed
        if self.signed:
            self.act_bits = act_bits + 1
            self.act_levels = 2**self.act_bits - 1
        else:
            self.act_bits = act_bits
            self.act_levels = 2**self.act_bits
        self.quan_inp = args.QInpFlag 
        self.baseline = args.baseline
        self.STE_discretizer = STE_discretizer.apply
        self.EWGS_discretizer = EWGS_discretizer.apply
        
        self.set_act_scale = args.set_act_scale
        self.act_scaling = nn.Parameter(data = torch.tensor(1.0), requires_grad = True)
        
        if self.quan_inp:
            self.register_buffer('bkwd_scaling_factorI', torch.tensor(args.bkwd_scaling_factorI).float())

        self.register_buffer('init', torch.tensor([0]))
       
        self.hook_Qvalues = False
        self.buff_inp = None
        
    def initialize(self, x):
        Qinp = x
        if self.quan_inp:
            Qinp = self.forward(x)
    
    def forward(self, x):
        if self.init == 1:
            self.initialize(x)
        
        if self.signed:
            x = (x + 1.0)/2.0
        x = x.clamp(0, 1)
        
        if self.quan_inp:
            if not self.baseline:
                x = self.EWGS_discretizer(x, self.act_levels, self.bkwd_scaling_factorI)
            else:
                x = self.STE_discretizer(x, self.act_levels)
        
        if self.hook_Qvalues:
            self.buff_inp = x
            self.buff_inp.retain_grad()
            
        if self.signed:
            x = (x - 0.5) * 2  
        
        if self.set_act_scale:
            x = x * self.act_scaling
            
        return x
    
class QInput(nn.Module):
    """
        Quant input to inp_bits
        if signed, clamp between [-self.uI, self.uI]; o.w. clamp between [0, self.uI]
    """
    def __init__(self, args, inp_bits = 8, signed = True, **kwargs):
        super(QInput, self).__init__()
        self.signed = signed
        if self.signed:
            self.inp_bits = inp_bits + 1
            self.inp_levels = 2**self.inp_bits-1
        else:
            self.inp_bits = inp_bits
            self.inp_levels = 2**self.inp_bits
        
        if gp.inp_quant_type == 'ideal':
            self.quan_inp = False
            self.inp_quant_type = gp.inp_quant_type
        else:
            self.quan_inp = args.QInpFlag
            self.inp_quant_type = gp.inp_quant_type
        
        if 'relu' in self.inp_quant_type:
            idx = (self.inp_quant_type).index('_')
            self.uI = int(self.inp_quant_type[idx+1:])
        elif self.inp_quant_type == 'ADC_dynamic':
            self.uI = nn.Parameter(data = torch.tensor(1.0), requires_grad = True)
        
        self.baseline = args.baseline
        self.STE_discretizer = STE_discretizer.apply
        self.EWGS_discretizer = EWGS_discretizer.apply
        
        # self.set_inp_scale = args.set_inp_scale
        # self.inp_scaling = nn.Parameter(data = torch.tensor(1.0), requires_grad = True)
        
        if self.quan_inp:
            self.register_buffer('bkwd_scaling_factorI', torch.tensor(args.bkwd_scaling_factorI).float())

        self.register_buffer('init', torch.tensor([0]))
       
        self.hook_Qvalues = False
        self.buff_inp = None
        
        self.act_max = torch.tensor(0.0) # maximum of activation; used for act_scaling
        
    def inp_quantization(self, x):
        ## Act quantization for x, x range between [0, 1]
        ## Output: quantize x to act_bits, output between [0, 1]
        if not self.baseline:
            x = self.EWGS_discretizer(x, self.inp_levels, self.bkwd_scaling_factorI)
        else:
            x = self.STE_discretizer(x, self.inp_levels)
            
        if self.hook_Qvalues:
            self.buff_inp = x
            self.buff_inp.retain_grad()
        
        return x
        
   
    def forward(self, x):
        self.act_max = x.max().data
            
        if 'relu' in self.inp_quant_type or self.inp_quant_type == 'dynamic_act' or self.inp_quant_type == 'ADC_dynamic':
            x = x/self.uI
            
        if self.quan_inp:
            if self.signed:
                x = ((x+1)/2).clamp(0, 1)
            else:
                x = x.clamp(0, 1)
            x = self.inp_quantization(x)
            if self.signed:
                x = (x - 0.5)*2
        else:
            if not self.signed:
                x = x.clamp(0)
            
        if self.hook_Qvalues:
            self.buff_inp = x
            self.buff_inp.retain_grad()
        
        Qinp = x
        if 'relu' in self.inp_quant_type or self.inp_quant_type == 'dynamic_act' or self.inp_quant_type == 'ADC_dynamic':
            Qinp = Qinp * self.uI
            
        return Qinp

class QAct(nn.Module):
    """
        Quant activation into act_bits
        if signed, clamp between [-self.uA, self.uA]; o.w. clamp between [0, self.uA]
    """
    def __init__(self, args, act_bits = 8, signed = False, **kwargs):
        super(QAct, self).__init__()
        self.signed = signed 
        if self.signed:
            self.act_bits = act_bits + 1
            self.act_levels = 2**act_bits - 1
        else:
            self.act_bits = act_bits
            self.act_levels = 2**act_bits

        if gp.act_quant_type == 'ideal':
            self.quan_act = False
            self.act_quant_type = gp.act_quant_type
        else:
            self.quan_act = args.QActFlag
            self.act_quant_type = gp.act_quant_type
        
        if self.act_quant_type == 'dynamic_act':
            self.uA = nn.Parameter(data = torch.tensor(1.0), requires_grad = True)
        elif self.act_quant_type == 'ideal':
            self.uA = nn.Parameter(data = torch.tensor(1.0), requires_grad = False)
        elif 'relu' in self.act_quant_type:
            idx = (self.act_quant_type).index('_')
            self.uA = float(self.act_quant_type[idx+1:])
            print("Act quant type: relu_%d" %(self.uA))
        elif self.act_quant_type == 'ADC_dynamic':
            self.uA = nn.Parameter(data = torch.tensor(1.0), requires_grad = False)
        
        ## Whether to model ADC noise effect
#         self.ADC_noise = args.ADC_noise
#         self.act_noise_type = args.ADC_noise_type
#         self.act_noise_sigma = args.ADC_noise_sigma

        self.baseline = args.baseline
        self.STE_discretizer = STE_discretizer.apply
        self.EWGS_discretizer = EWGS_discretizer.apply
        
        if self.quan_act:
            self.register_buffer('bkwd_scaling_factorA', torch.tensor(args.bkwd_scaling_factorA).float())
        self.register_buffer('init', torch.tensor([0]))
       
        self.hook_Qvalues = False
        self.buff_act = None

        
    def act_quantization(self, x):
        ## Act quantization for x, x range between [0, 1]
        ## Output: quantize x to act_bits, output between [0, 1]
        if not self.baseline:
            x = self.EWGS_discretizer(x, self.act_levels, self.bkwd_scaling_factorA)
        else:
            x = self.STE_discretizer(x, self.act_levels)    
        if self.hook_Qvalues:
            self.buff_act = x
            self.buff_act.retain_grad()
        return x
        
    def forward(self, x):
        """
            Act quantization for x
            If self.act_signed: 
                x clamped between [-self.uA, self.uA]; 
            else:
                x clamped between [0, self.uA]
                
            Output: 
            - x quantized to act_bits
        """
        if self.act_quant_type == 'ideal':
            if self.act_signed == False:
                x = x.clamp(0)
            self.uA.data = x.abs().max()
        else:
            if self.act_quant_type == 'dynamic_act' and x.abs().max()>self.uA.data:
                self.uA.data = x.abs().max()
            x = x/self.uA

            if self.quan_act:
                if self.signed:
                    x = ((x+1.0)/2.0).clamp(0, 1)
                else:
                    x = x.clamp(0, 1.0)
                x = self.act_quantization(x)
                if self.signed:
                    x = (x - 0.5)*2.0
            else:
                if not self.signed:
                    x = x.clamp(0)

        ###################################################################
        ## Model ADC noise & saturation effect here
#         if self.training_mode in ['fixed_w_range_CB_quant', 'fixed_w_full_range_CB_quant']:
#             ADC_noise = torch.distributions.Normal(torch.zeros(x.shape, device = x.device), gp.ADC_sigma).rsample()
#             x -= x*(x.abs()>0.75)*(ADC_noise.abs())
        #######################################################################

        if self.act_quant_type!='ideal':
            x = x*self.uA

        return x
      
class unpack_bits(nn.Module):
    """
        Unpack the input (FP value between [0, 1]) to a num_bit length bit sequence of 0, 1 from MSB->LSB
        Input:
            x, size [Batch, ...]
            x between [0, 1]
            inp_bit: num_bit into which the input is unpacked, default = 8, DAC bits
        Output:
            output, list with len(inp_bit), each with size of x
    """
    def __init__(self, inp_bit = 8, name = None):
        super(unpack_bits, self).__init__()
        self.name = name
        self.inp_bit = inp_bit
        self.inp_levels = 2**inp_bit - 1
        self.base = (2**torch.arange(inp_bit-1, -1, -1)).int() # from MSB to LSB
        self.floor_divide = Floor_divide.apply
    def forward(self, x):
        x = x.clamp(0, 1)
        x = x*self.inp_levels
        output = [None]*self.inp_bit
        for i in range(self.inp_bit):
            output[i] = self.floor_divide(x, self.base[i])
            x = x - output[i] * self.base[i]
        return output
    
class pack_bits(nn.Module):
    """
        Revert of unpack_bits
        Convert a num_bit length bit sequence to a value between [0, 1]
        Input:
            inp_bit: num_bit of the input length, default = 8, ADC bits, MSB->LSB
            input, list with len(inp_bit)
        Output: 
            output, FP value converted from the bit-serial input
    """
    def __init__(self, inp_bit = 8, name = None):
        super(pack_bits, self).__init__()
        self.name = name
        self.inp_bit = inp_bit
        self.inp_levels = 2**inp_bit - 1
        self.base = (2**torch.arange(inp_bit-1, -1, -1)) * (1/self.inp_levels)
    
    def forward(self, x):
        output = [None]*len(x)
        for i in range(len(x)):
            output[i] = x[i] * self.base[i]
        output = torch.stack(output, dim=0).sum(dim=0)
        return output

    
class Quant_Conv2d(nn.Conv2d):
    """
        Conv. module for quantization aware training
        Perform weight quantization & Act quantization; 
        No bias
        **Note: during inderence, non-quantized FP weights are used
        Bit-serial mode is supported;
        Bitcell limited onoff_ratio is considered;
        Support noise injection to weights during training;
        
        Weight quantization training modes:
            1) Fixed_w_range_CB_quant: [weight_min_train, weight_max_train], only pos_weight, quantized weights, gp.weight_bits
            2) Fixed_w_full_range_CB_quant: [-weight_max_train+weight_min_train, weight_max_train-weight_min_train], 
            both pos & neg weight, quantized weights, gp.weight_bits
            2) Fixed_w_range_non_CB: for dual array training, load weights from other trained model, gp.weight_bits
            3) Dynamic_w_range: set self.uW = max_weight, only pos_weight, gp.weight_bits
            4) Dynamic_w_full_range: set self.uW = max_abs_weight, both pos and neg weights, gp.weight_bits+1 
        
        No bias, No input/activation scaling
        
        input:
        - args: argument class, 
        - training mode: []
        - dual array: boolean, whether to use dual column mapping for pos and neg weights
        - mode: ['training', 'reference', 'bit_serial']
            Use 'bit-serial': inference accuracy in bit-serial mode
        - tile: boolean, whether to use tiled architecture. This affects weight initialization. 
        - fan_in: int, number of fan in. Only work when tile=True. 
        - kwargs: arguments to define nn.Conv2d 
        - gp: crossbar class
            gp.act_bits; gp.onoff_ratio_ideal; gp.onoff_ratio; gp.deltaw_threshold
        
        function:
        
    """
    def __init__(self, args, training_mode='dynamic_w_range', dual_array=True, mode='training', 
                 tile=False, fan_in=1, **kwargs):
        super(Quant_Conv2d, self).__init__(**kwargs)
        self.quan_weight = args.QWeightFlag
        kwargs['bias'] = None
        self.baseline = args.baseline
        self.STE_discretizer = STE_discretizer.apply
        self.EWGS_discretizer = EWGS_discretizer.apply
        self.training_mode = training_mode
        self.dual_array = dual_array
        self.tile = tile
        self.inp_signed = args.inp_signed
        print('Inp_signed:', args.inp_signed)
        self.fan_in = fan_in
        self.mode = mode
        if mode=='bit_serial':
            self.input_layer = unpack_bits(inp_bit = gp.act_bits)
            self.input_levels = 2**gp.act_bits - 1
            self.base = (2**torch.arange(gp.act_bits-1, -1, -1)) * (1/self.input_levels)
            self.base = self.base.to(self.weight.device)
            
        self.onoff_ratio_ideal = gp.onoff_ratio_ideal
        if not self.onoff_ratio_ideal:
            self.onoff_ratio = gp.onoff_ratio
            
        self.noise_type = None
        if args.noisy and args.noise_type is not None and self.quan_weight:
            self.noise_type = args.noise_type
            self.noise_sigma = args.noise_sigma
    
        ################################################
        ## Weight initialization
        if self.tile and self.training_mode in ['fixed_w_range_CB_quant', 'fixed_w_full_range_CB_quant']:
            gain = math.sqrt(2.0)
            std = gain / math.sqrt(self.fan_in)
            with torch.no_grad():
                self.weight.normal_(0, std)
        else:
            self.weight = nn.init.kaiming_normal_(self.weight, mode = 'fan_in', nonlinearity = 'relu')
        
        if self.mode == 'mixed_precision_training':
            self.prog_noise_sigma = args.prog_noise_sigma
            self.deltaw_threshold = gp.deltaw_threshold ## Weight update threshold
            self.delta_w = nn.Parameter(data = torch.zeros(self.weight.shape), requires_grad = False)
            self.weight_low = nn.Parameter(data = self.weight.data, requires_grad = True)
         
        self.Qweight = self.weight.data
        ## Weight quantization
        if self.training_mode == 'fixed_w_range_CB_quant':
            """
                For training in CB_quant scheme w. fixed ADC dynamic range,
                Weight are quantized during training,
                Positive only weights normalized w.r.t the ADC dynamic range. 
                [weight_min_train, weight_max_train]=>[1/gp.onoff_ratio, 1]
                wt_bit = gp.weight_bits 
            """
            self.weight_bits = gp.weight_bits
            self.weight_levels = 2**self.weight_bits
            if self.onoff_ratio_ideal:
                self.lW_static = 0.0
            else:
                self.lW_static = 1.0/self.onoff_ratio 
            self.uW_static = 1.0
            kernel = self.weight.data
            kernel = kernel/kernel.abs().max()
            kernel = kernel * (2**self.weight_bits - 1)
            kernel = torch.round(kernel)
            kernel = kernel / (2**self.weight_bits - 1)
            kernel = kernel*(self.uW_static-self.lW_static) + self.lW_static
            self.Qweight = kernel
            with torch.no_grad():
                self.weight.data.copy_(self.Qweight)
                if self.mode == 'mixed_precision_training':
                    self.weight_low.data.copy_(self.Qweight)
        elif self.training_mode == 'fixed_w_full_range_CB_quant':
            """
                For training in CB scheme w. fixed ADC dynamic range
                Both positive & negative weights, normalize w.r.t the ADC dynamic range. 
                [-1+1/onoff_ratio, 1-1/onoff_ratio] 
                wt_bit = gp.weight_bits+1
            """
            self.weight_bits = gp.weight_bits + 1
            self.weight_levels = 2**self.weight_bits - 1
            self.uW_static = 1.0 - 1/self.onoff_ratio
            self.lW_static = -self.uW_static
            
            ## fake quantization
            kernel = self.weight.data
            kernel = kernel/kernel.abs().max()
            kernel = kernel * (2**self.weight_bits - 1)
            kernel = torch.round(kernel)
            kernel = kernel / (2**self.weight_bits - 1)
            kernel = kernel*self.uW_static
            self.Qweight = kernel
            with torch.no_grad():
                self.weight.data.copy_(self.Qweight)
                if self.mode == 'mixed_precision_training':
                    self.weight_low.data.copy_(self.Qweight)
        elif self.training_mode == 'dynamic_w_range':
            """
                For quant dual array training, non_CB scheme
                Positive only weights [self.uW/onoff_ratio, self.uW]
                wt_bits = gp.weight_bits
            """
            self.weight_bits = gp.weight_bits
            self.weight_levels = 2**self.weight_bits
            if self.quan_weight:
                self.uW = nn.Parameter(data = self.weight.max().data, requires_grad = True)
            else:
                self.uW = nn.Parameter(data = self.weight.max().data, requires_grad = False)
            if gp.onoff_ratio_ideal:
                self.lW = torch.tensor(0.0)
                gp.weight_min_train = 0
            else:
                self.lW = self.uW / self.onoff_ratio
        elif self.training_mode == 'dynamic_w_full_range':
            """ 
                For quant-only training, symmetric weights
                Support positive and negative weights
                wt_bits = gp.weight_bits + 1
            """
            self.weight_bits = gp.weight_bits + 1
            self.weight_levels = 2**self.weight_bits - 1
            if self.quan_weight:
                self.uW = nn.Parameter(data = self.weight.abs().max().data, requires_grad = True) 
                self.lW = -self.uW
            else:
                self.uW = nn.Parameter(data = self.weight.abs().max().data, requires_grad = False) 
                self.lW = -self.uW 
            
        if self.quan_weight:
            self.register_buffer('bkwd_scaling_factorW', torch.tensor(args.bkwd_scaling_factorW).float())
            
        ######################################################################################################
        self.register_buffer('init', torch.tensor([0]))
        self.hook_Qvalues = False
        self.buff_weight = None
    
    def weight_quantization(self, weight, w_min, w_max):
        """
            Perform quantization of weight to self.weight_levels between [w_min, w_max]
            inputs: 
            - weight: weight matrix in FP32
            - w_min, w_max: min and max of weight
            outputs:
            - quantized weight
            
            Gradient of the weights will be computed with STE or EWGS discretizer solver 
            
        """
        weight = (weight - w_min) / (w_max - w_min)
        weight = weight.clamp(0, 1)
        
        if not self.baseline:
            weight = self.EWGS_discretizer(weight, self.weight_levels, self.bkwd_scaling_factorW)
        else:
            weight = self.STE_discretizer(weight, self.weight_levels)
            
        if self.hook_Qvalues:
            self.buff_weight = weight
            self.buff_weight.retain_grad()
            
        weight = w_min + (w_max - w_min) * weight

        return weight


    def initialize(self, x):
        Qweight = self.weight.data
        
        if self.training_mode == 'fixed_w_range_CB_quant' and self.quan_weight:
            self.Qweight = self.weight_quantization(Qweight, self.lW_static, self.uW_static)
        elif self.training_mode == 'fixed_w_full_range_CB_quant' and self.quan_weight:
            self.Qweight = self.weight_quantization(Qweight, self.lW_static, self.uW_static)
        elif self.training_mode == 'dynamic_w_range':
            Qweight = Qweight.clamp(0)
            self.uW.data.fill_(torch.max(Qweight.max(), Qweight.std()*3))
            if self.onoff_ratio_ideal:
                self.lW = torch.tensor(0.0)
            else:
                self.lW = self.uW / self.onoff_ratio
            if self.quan_weight:
                self.Qweight = self.weight_quantization(Qweight, self.lW, self.uW)
        elif self.training_mode == 'dynamic_w_full_range':
            self.uW.data.fill_(torch.max(Qweight.abs().max(), Qweight.abs().std()*3))
            self.lW = -self.uW
            if self.quan_weight:
                self.Qweight = self.weight_quantization(Qweight, self.lW, self.uW)
        
    def distribution_func(self, weight, noise_type, noise_sigma):
        """
            Noise injection 
            inputs:
            - noise type: ['']
            - noise sigma:
        """
        if noise_type is None:
            return noise_type
        
        assert noise_type in ['sigma_relative'], 'Noise-aware type is not available.'
        noise_type = noise_type.lower()
        
        with torch.no_grad():
            if noise_type == 'sigma_relative':    
                if self.training_mode in ['fixed_w_range_CB_quant', 'fixed_w_full_range_CB_quant', 'fixed_w_range_non_CB']:
                    kernel_range = self.uW_static - self.lW_static
                elif self.training_mode in ['dynamic_w_range', 'dynamic_w_full_range']:
                    kernel_range = self.uW - self.lW
            
            scaled_sigma = kernel_range / (self.weight_levels - 1) * noise_sigma

            return torch.distributions.Normal(torch.zeros(weight.shape, device = weight.device), scaled_sigma)
            
    def forward(self, x):
        if self.init == 1:
            self.initialize(x)
            
        #########################################################################################
        ########################  Weight quantization & Noise ###################################
        with torch.no_grad():
            if self.training_mode in ['fixed_w_range_CB_quant', 'fixed_w_full_range_CB_quant']:
                self.weight.clamp_(min = self.lW_static, max = self.uW_static)
            elif self.training_mode in ['dynamic_w_range']:
                self.weight.clamp_(0)
        
        if self.mode != "mixed_precision_training":
            Qweight = self.weight
            
        # fake quantization of the weights
        # If noise-aware, add noise to weights during training (not during inference)
        if self.quan_weight and self.mode!='mixed_precision_training' and self.mode!='inference':
            if self.training_mode in ['fixed_w_range_CB_quant', 'fixed_w_full_range_CB_quant']:
                Qweight = self.weight_quantization(Qweight, self.lW_static, self.uW_static)
            elif self.training_mode in ['dynamic_w_range', 'dynamic_w_full_range']:
                if self.training_mode == 'dynamic_w_range':
                    if self.onoff_ratio_ideal:
                        self.lW = torch.tensor(0.0)
                    else:
                        self.lW = self.uW / self.onoff_ratio
                elif self.training_mode == 'dynamic_w_full_range':
                    self.lW = -self.uW
                Qweight = self.weight_quantization(Qweight, self.lW, self.uW)
        elif self.quan_weight == False and self.training_mode == 'dynamic_w_full_range':
            self.uW.data.fill_(Qweight.abs().max()) 
            self.lW = -self.uW
        elif self.mode=='mixed_precision_training':
            if self.training_mode == 'fixed_w_full_range_CB_quant':
                # Apply threshold function & rounding function
                if self.training:
                    with torch.no_grad():
                        mask_deltaw = self.delta_w.data.abs()>self.deltaw_threshold
                        w_diff = self.delta_w.data * mask_deltaw
                        w_diff = torch.round(w_diff/self.deltaw_threshold) * self.deltaw_threshold 
                        if len(torch.where(mask_deltaw)[0])>0:
                            idx = torch.where(mask_deltaw)
#                             print('# of update weights:', len(idx[0]), '; # of total weights:', mask_deltaw.numel(), '; delta_w:', w_diff[idx].abs().max())
                        self.delta_w[mask_deltaw] = 0 
                        if self.noise_type is not None:
                            kernel_range = self.uW_static - self.lW_static
                            w_diff = w_diff * torch.normal(1.0, self.prog_noise_sigma*kernel_range/(self.weight_levels-1), w_diff.shape, device = w_diff.device)
                        self.weight_low.data += w_diff
                        self.weight_low.data = self.weight_low.data.clamp(self.lW_static, self.uW_static)
                Qweight = self.weight_low
        
        self.Qweight = Qweight.data
        # Inject noise to weights during training 
#         if self.training and (self.noise_type is not None):
        if (self.noise_type is not None):
            Qweight = Qweight + self.distribution_func(Qweight, self.noise_type, self.noise_sigma).rsample()
        
        if self.training_mode in ['fixed_w_range_CB_quant', 'dynamic_w_range']:
            Qweight = Qweight.clamp(0)
            self.Qweight = Qweight.data
        elif self.training_mode == 'fixed_w_full_range_CB_quant':
            Qweight_pos = Qweight * (Qweight > 0) + 1.0/self.onoff_ratio
            Qweight_neg = -Qweight * (Qweight < 0) + 1.0/self.onoff_ratio
            self.Qweight_pos = Qweight_pos.data
            self.Qweight_neg = Qweight_neg.data
        elif self.training_mode == 'dynamic_w_full_range':
            Qweight_pos = Qweight * (Qweight > 0) 
            Qweight_neg = -Qweight * (Qweight < 0)
            self.Qweight_pos = Qweight_pos.data
            self.Qweight_neg = Qweight_neg.data
        
        #########################################################################################
        ##############################    Forward prop      ##################################### 
        Qact = []
        if self.mode == 'bit_serial':
            if self.inp_signed:
                x_pos = F.relu(x)
                x_neg = F.relu(-x)
                x_unpack = [self.input_layer(x_pos), self.input_layer(x_neg)]
            else:
                x_unpack = [self.input_layer(x)]

            for idx_inp in range(len(x_unpack)):
                x_inp = x_unpack[idx_inp]
                if self.training_mode in ['fixed_w_range_CB_quant', 'dynamic_w_range']:
                    qact = [self.base[idx_bit]*F.conv2d(x_inp[idx_bit], Qweight, None, self.stride, self.padding, self.dilation, self.groups) for idx_bit in range(len(x_inp))]
                    qact = torch.stack(qact, dim = 0).sum(dim = 0)
                    Qact.append(qact)
                elif self.training_mode in ['fixed_w_full_range_CB_quant', 'dynamic_w_full_range']:
                    if self.dual_array:
                        qact_pos = [self.base[idx_bit]*F.conv2d(x_inp[idx_bit], Qweight_pos, None, self.stride, self.padding, self.dilation, self.groups) for idx_bit in range(len(x_inp))]
                        qact_neg = [self.base[idx_bit]*F.conv2d(x_inp[idx_bit], Qweight_neg, None, self.stride, self.padding, self.dilation, self.groups) for idx_bit in range(len(x_inp))]
                        qact_pos = torch.stack(qact_pos, dim = 0).sum(dim = 0)
                        qact_neg = torch.stack(qact_neg, dim = 0).sum(dim = 0)
                        Qact.append(qact_pos)
                        Qact.append(qact_neg)
                    else:
                        qact = [self.base[idx_bit]*F.conv2d(x_inp[idx_bit], Qweight, None, self.stride, self.padding, self.dilation, self.groups) for idx_bit in range(len(x_inp))]
                        qact = torch.stack(qact, dim = 0).sum(dim = 0)
                        Qact.append(qact)

        else:
            if self.inp_signed:
                x_pos = F.relu(x)
                x_neg = F.relu(-x)
                x_inp = [x_pos, x_neg]
            else:
                x_inp = [x]
                
            if self.training_mode in ['fixed_w_range_CB_quant', 'dynamic_w_range']:
                for idx_inp in range(len(x_inp)):
                    qact = F.conv2d(x_inp[idx_inp], Qweight, None, self.stride, self.padding, self.dilation, self.groups)
                    Qact.append(qact)
            elif self.training_mode in ['fixed_w_full_range_CB_quant', 'dynamic_w_full_range']:
                if self.dual_array:
                    for idx_inp in range(len(x_inp)):
                        qact_pos = F.conv2d(x_inp[idx_inp], Qweight_pos, None, self.stride, self.padding, self.dilation, self.groups) 
                        qact_neg = F.conv2d(x_inp[idx_inp], Qweight_neg, None, self.stride, self.padding, self.dilation, self.groups)
                        Qact.append(qact_pos)
                        Qact.append(qact_neg)
                else:
                    for idx_inp in range(len(x_inp)):
                        qact = F.conv2d(x_inp[idx_inp], Qweight, None, self.stride, self.padding, self.dilation, self.groups)
                        Qact.append(qact)

        self.output_float = F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
        
        return Qact
    
    def scale_weights(self, w_max, w_source = None):
        """
            Scale weights to Bitcell weights
            args: 
                1) w_max: maximum abs value of signed weights; same for positive & nagative array
                2) inp_act_scale: scaling factor for weights to adapt to the new inp & act range
            Set scaling factor for output 
        """
        kernel = self.weight.data
        if w_source == None:
            if self.training_mode == 'fixed_w_range_CB_quant':
                kernel = self.weight.clamp(0)
                kernel = self.weight_quantization(kernel, 0, w_max)
                kernel, scaling_factor = rescale_weight(kernel, 0, w_max, self.lW_static, self.uW_static)
            elif self.training_mode == 'fixed_w_full_range_CB_quant':
                kernel = self.weight_quantization(kernel, -w_max, w_max)
                kernel, scaling_factor = rescale_weight(kernel, -w_max, w_max, self.lW_static, self.uW_static)
            with torch.no_grad():
                self.weight.data = kernel 
            return kernel, scaling_factor
        elif w_source == 'fixed_w_full_range_CB_quant':
            with torch.no_grad():
                kernel += self.lW_static
                kernel = self.weight_quantization(kernel, self.lW_static, self.uW_static)
                self.weight.data = kernel
            return kernel

class Quant_Linear(nn.Linear):
    """
        Linear module for quantization aware training
        Perform weight quantization & Act quantization;
        No bias;
        **Note: during inderence, non-quantized FP weights are used
        **Note: Weight initialization needs to be taken care of for HW aware training; Currently, only "fixed_w_full_range_CB_quant" is supported. 
        Bit-serial mode is supported;
        Bitcell limited onoff_ratio is considered;
        Support noise injection to weights during training;
        
        Weight quantization training modes:
            1) Fixed_w_range_CB_quant: [weight_min_train, weight_max_train], only pos_weight, quantized weights, gp.weight_bits
            2) Fixed_w_range_non_CB: for dual array training, load weights fro other trained model, gp.weight_bits
            3) Dynamic_w_range: set self.uW = max_weight, only pos_weight, gp.weight_bits
            4) Dynamic_w_full_range: set self.uW = max_abs_weight, both pos and neg weights, gp.weight_bits+1 
            
        No bias, No input/activation scaling
    """
    def __init__(self, args, training_mode = 'dynamic_w_range', dual_array = True, mode = 'training', 
                 tile = False, fan_in = 1, **kwargs):
        super(Quant_Linear, self).__init__(**kwargs)
        self.quan_weight = args.QWeightFlag
        kwargs['bias'] = None
        self.baseline = args.baseline
        self.STE_discretizer = STE_discretizer.apply
        self.EWGS_discretizer = EWGS_discretizer.apply
        self.training_mode = training_mode
        self.dual_array = dual_array
        self.tile = tile
        self.fan_in = fan_in
        self.inp_signed = args.inp_signed
        print('Inp_signed:', args.inp_signed)
        self.mode = mode
        if mode=='bit_serial':
            self.input_layer = unpack_bits(inp_bit = gp.act_bits)
            self.input_levels = 2**gp.act_bits - 1
            self.base = (2**torch.arange(gp.act_bits-1, -1, -1)) * (1/self.input_levels)
            self.base = self.base.to(self.weight.device)
            
        self.onoff_ratio_ideal = gp.onoff_ratio_ideal
        if not self.onoff_ratio_ideal:
            self.onoff_ratio = gp.onoff_ratio
        
        self.noise_type = None
        if args.noisy and args.noise_type is not None and self.quan_weight:
            self.noise_type = args.noise_type
            self.noise_sigma = args.noise_sigma
            
        ################################################
        ## Weight initialization
        if self.tile and self.training_mode in ['fixed_w_range_CB_quant', 'fixed_w_full_range_CB_quant']:
            gain = math.sqrt(2.0)
            std = gain / math.sqrt(self.fan_in)
            with torch.no_grad():
                self.weight.normal_(0, std)
        else:
            self.weight = nn.init.kaiming_normal_(self.weight, mode = 'fan_in', nonlinearity = 'relu')
            
        if self.tile and self.training_mode in ['fixed_w_range_CB_quant', 'fixed_w_full_range_CB_quant']:
            gain = math.sqrt(2.0)
            std = gain / math.sqrt(self.fan_in)
            with torch.no_grad():
                self.weight.normal_(0, std)
        else:
            self.weight = nn.init.kaiming_normal_(self.weight, mode = 'fan_in', nonlinearity = 'relu')
        
        if self.mode == 'mixed_precision_training':
            self.prog_noise_sigma = args.prog_noise_sigma
            self.deltaw_threshold = gp.deltaw_threshold ## Weight update threshold
            self.delta_w = nn.Parameter(data = torch.zeros(self.weight.shape), requires_grad = False)
            self.weight_low = nn.Parameter(data = self.weight.data, requires_grad = True)
        
        self.Qweight = self.weight.data

        ## Weight quantization
        if self.training_mode == 'fixed_w_range_CB_quant':
            """
                For training in CB_quant scheme w. fixed ADC dynamic range,
                Weight are quantized during training,
                Positive only weights normalized w.r.t the ADC dynamic range. 
                [weight_min_train, weight_max_train]=>[1/gp.onoff_ratio, 1]
                wt_bit = gp.weight_bits 
            """
            self.weight_bits = gp.weight_bits
            self.weight_levels = 2**self.weight_bits
            if self.onoff_ratio_ideal:
                self.lW_static = 0.0
            else:
                self.lW_static = 1.0/self.onoff_ratio 
            self.uW_static = 1.0
            kernel = self.weight.data
            kernel = kernel/kernel.abs().max()
            kernel = kernel * (2**self.weight_bits - 1)
            kernel = torch.round(kernel)
            kernel = kernel / (2**self.weight_bits - 1)
            kernel = kernel*(self.uW_static-self.lW_static) + self.lW_static
            self.Qweight = kernel
            with torch.no_grad():
                self.weight.data.copy_(self.Qweight)
                if self.mode == 'mixed_precision_training':
                    self.weight_low.data.copy_(self.Qweight)
        elif self.training_mode == 'fixed_w_full_range_CB_quant':
            """
                For training in CB scheme w. fixed ADC dynamic range
                Both positive & negative weights, normalize w.r.t the ADC dynamic range. 
                [-1+1/onoff_ratio, 1-1/onoff_ratio] 
                wt_bit = gp.weight_bits+1
            """
            self.weight_bits = gp.weight_bits + 1
            self.weight_levels = 2**self.weight_bits - 1
            self.uW_static = 1.0 - 1/self.onoff_ratio
            self.lW_static = -self.uW_static
            
            ## fake quantization
            kernel = self.weight.data
            kernel = kernel/kernel.abs().max()
            kernel = kernel * (2**self.weight_bits - 1)
            kernel = torch.round(kernel)
            kernel = kernel / (2**self.weight_bits - 1)
            kernel = kernel*self.uW_static
            self.Qweight = kernel
            with torch.no_grad():
                self.weight.data.copy_(self.Qweight)
                if self.mode == 'mixed_precision_training':
                    self.weight_low.data.copy_(self.Qweight)
        elif self.training_mode == 'dynamic_w_range':
            """
                For quant dual array training, non_CB scheme
                Positive only weights [self.uW/onoff_ratio, self.uW]
                wt_bits = gp.weight_bits
            """
            self.weight_bits = gp.weight_bits
            self.weight_levels = 2**self.weight_bits
            self.uW = nn.Parameter(data = self.weight.max().data, requires_grad = True)
            if gp.onoff_ratio_ideal:
                self.lW = torch.tensor(0.0)
                gp.weight_min_train = 0
            else:
                self.lW = self.uW / self.onoff_ratio
        elif self.training_mode == 'dynamic_w_full_range':
            """ 
                For quant-only training, symmetric weights
                Support positive and negative weights
                wt_bits = gp.weight_bits + 1
            """
            self.weight_bits = gp.weight_bits + 1
            self.weight_levels = 2**self.weight_bits - 1
            if self.quan_weight:
                self.uW = nn.Parameter(data = self.weight.abs().max().data, requires_grad = True) 
                self.lW = -self.uW
            else:
                self.uW = nn.Parameter(data = self.weight.abs().max().data, requires_grad = False) 
                self.lW = -self.uW 
        
        if self.quan_weight:
            self.register_buffer('bkwd_scaling_factorW', torch.tensor(args.bkwd_scaling_factorW).float())

        ######################################################################################################
        self.register_buffer('init', torch.tensor([0]))
        self.hook_Qvalues = False
        self.buff_weight = None
    
    def weight_quantization(self, weight, w_min, w_max):
        
        weight = (weight - w_min) / (w_max - w_min)
        weight = weight.clamp(0, 1)
        
        if not self.baseline:
            weight = self.EWGS_discretizer(weight, self.weight_levels, self.bkwd_scaling_factorW)
        else:
            weight = self.STE_discretizer(weight, self.weight_levels)
            
        if self.hook_Qvalues:
            self.buff_weight = weight
            self.buff_weight.retain_grad()
            
        weight = w_min + (w_max - w_min) * weight

        return weight

    def initialize(self, x):
        """
            Quantize the weights if quan_weight is True and store in self.Qweight 
            Initialize uW and lW
        """
        Qweight = self.weight.data
        
        if self.training_mode == 'fixed_w_range_CB_quant' and self.quan_weight:
            self.Qweight = self.weight_quantization(Qweight, self.lW_static, self.uW_static)
        elif self.training_mode == 'fixed_w_full_range_CB_quant' and self.quan_weight:
            self.Qweight = self.weight_quantization(Qweight, self.lW_static, self.uW_static)
        elif self.training_mode == 'dynamic_w_range':
            Qweight = Qweight.clamp(0)
            self.uW.data.fill_(torch.max(Qweight.max(), Qweight.std()*3))
            if self.onoff_ratio_ideal:
                self.lW = torch.tensor(0.0)
            else:
                self.lW = self.uW / self.onoff_ratio
            if self.quan_weight:
                self.Qweight = self.weight_quantization(Qweight, self.lW, self.uW)
        elif self.training_mode == 'dynamic_w_full_range':
            self.uW.data.fill_(torch.max(Qweight.abs().max(), Qweight.abs().std()*3))
            self.lW = -self.uW
            if self.quan_weight:
                self.Qweight = self.weight_quantization(Qweight, self.lW, self.uW)
                        
    def distribution_func(self, weight, noise_type, noise_sigma):
        if noise_type is None:
            return noise_type
        
        assert noise_type in ['sigma_relative'], 'Noise-aware type is not available.'
        noise_type = noise_type.lower()
        
        with torch.no_grad():
            if noise_type == 'sigma_relative':    
                if self.training_mode in ['fixed_w_range_CB_quant', 'fixed_w_full_range_CB_quant', 'fixed_w_range_non_CB']:
                    kernel_range = self.uW_static - self.lW_static
                elif self.training_mode in ['dynamic_w_range', 'dynamic_w_full_range']:
                    kernel_range = self.uW - self.lW
            
            scaled_sigma = kernel_range / (self.weight_levels - 1) * noise_sigma

            return torch.distributions.Normal(torch.zeros(weight.shape, device = weight.device), scaled_sigma)
    
    def forward(self, x):
        if self.init == 1:
            self.initialize(x)
        
        #########################################################################################
        ########################  Weight quantization & Noise ###################################
        with torch.no_grad():
            if self.training_mode in ['fixed_w_range_CB_quant', 'fixed_w_full_range_CB_quant']:
                self.weight.clamp_(min = self.lW_static, max = self.uW_static)
            elif self.training_mode in ['dynamic_w_range']:
                self.weight.clamp_(0)
       
        if self.mode != "mixed_precision_training":
            Qweight = self.weight
        
        # fake quantization of the weights
        # If noise-aware, add noise to weights during training (not during inference)
        if self.quan_weight and self.mode!='mixed_precision_training' and self.mode!='inference':
            if self.training_mode in ['fixed_w_range_CB_quant', 'fixed_w_full_range_CB_quant']:
                Qweight = self.weight_quantization(Qweight, self.lW_static, self.uW_static)
            elif self.training_mode in ['dynamic_w_range', 'dynamic_w_full_range']:
                if self.training_mode == 'dynamic_w_range':
                    if self.onoff_ratio_ideal:
                        self.lW = torch.tensor(0.0)
                    else:
                        self.lW = self.uW / self.onoff_ratio
                elif self.training_mode == 'dynamic_w_full_range':
                    self.lW = -self.uW
                Qweight = self.weight_quantization(Qweight, self.lW, self.uW)
        elif self.quan_weight == False and self.training_mode == 'dynamic_w_full_range':
            self.uW.data.fill_(Qweight.abs().max()) 
            self.lW = -self.uW
        elif self.mode=='mixed_precision_training':
            if self.training_mode == 'fixed_w_full_range_CB_quant':
                # Apply threshold function & rounding function
                if self.training:
                    with torch.no_grad():
                        mask_deltaw = self.delta_w.data.abs()>self.deltaw_threshold
                        w_diff = self.delta_w.data * mask_deltaw
                        w_diff = torch.round(w_diff/self.deltaw_threshold) * self.deltaw_threshold 
                        if len(torch.where(mask_deltaw)[0])>0:
                            idx = torch.where(mask_deltaw)
    #                         print('Update weight:', len(idx[0]), '; # of total weights:', mask_deltaw.numel(), '; delta_w:', w_diff[idx])
                        self.delta_w[mask_deltaw] = 0
                        if self.noise_type is not None:
                            kernel_range = self.uW_static - self.lW_static
                            w_diff = w_diff * torch.normal(1.0, self.prog_noise_sigma*kernel_range/(self.weight_levels-1), w_diff.shape, device = w_diff.device)
                        self.weight_low.data += w_diff
                        self.weight_low.data = self.weight_low.data.clamp(self.lW_static, self.uW_static)
                    Qweight = self.weight_low
        
        self.Qweight = Qweight.data
        # Inject noise to weights during training 
#         if self.training and (self.noise_type is not None):
        if (self.noise_type is not None):
            Qweight = Qweight + self.distribution_func(Qweight, self.noise_type, self.noise_sigma).rsample()
        
        if self.training_mode in ['fixed_w_range_CB_quant', 'dynamic_w_range']:
            Qweight = Qweight.clamp(0)
            self.Qweight = Qweight.data
        elif self.training_mode == 'fixed_w_full_range_CB_quant':
            Qweight_pos = Qweight * (Qweight > 0) + 1.0/self.onoff_ratio
            Qweight_neg = -Qweight * (Qweight < 0) + 1.0/self.onoff_ratio
            self.Qweight_pos = Qweight_pos.data
            self.Qweight_neg = Qweight_neg.data
        elif self.training_mode == 'dynamic_w_full_range':
            Qweight_pos = Qweight * (Qweight > 0) 
            Qweight_neg = -Qweight * (Qweight < 0)
            self.Qweight_pos = Qweight_pos.data
            self.Qweight_neg = Qweight_neg.data

        #########################################################################################
        ##############################    Forward prop      ##################################### 
        Qact = []
        if self.mode == 'bit_serial':
            if self.inp_signed:
                x_pos = F.relu(x)
                x_neg = F.relu(-x)
                x_unpack = [self.input_layer(x_pos), self.input_layer(x_neg)]
            else:
                x_unpack = [self.input_layer(x)]

            for idx_inp in range(len(x_unpack)):
                x_inp = x_unpack[idx_inp]
                if self.training_mode in ['fixed_w_range_CB_quant', 'dynamic_w_range']:
                    qact = [self.base[idx_bit]*F.linear(x_inp[idx_bit], Qweight, None) for idx_bit in range(len(x_inp))]
                    qact = torch.stack(qact, dim = 0).sum(dim = 0)
                    Qact.append(qact)
                elif self.training_mode in ['fixed_w_full_range_CB_quant', 'dynamic_w_full_range']:
                    if self.dual_array:
                        qact_pos = [self.base[idx_bit]*F.linear(x_inp[idx_bit], Qweight_pos, None) for idx_bit in range(len(x_inp))]
                        qact_neg = [self.base[idx_bit]*F.linear(x_inp[idx_bit], Qweight_neg, None) for idx_bit in range(len(x_inp))]
                        qact_pos = torch.stack(qact_pos, dim = 0).sum(dim = 0)
                        qact_neg = torch.stack(qact_neg, dim = 0).sum(dim = 0)
                        Qact.append(qact_pos)
                        Qact.append(qact_neg)
                    else:
                        qact = [self.base[idx_bit]*F.linear(x_inp[idx_bit], Qweight, None) for idx_bit in range(len(x_inp))]
                        qact = torch.stack(qact, dim = 0).sum(dim = 0)
                        Qact.append(qact)
        else:
            if self.inp_signed:
                x_pos = F.relu(x)
                x_neg = F.relu(-x)
                x_inp = [x_pos, x_neg]
            else:
                x_inp = [x]
                
            if self.training_mode in ['fixed_w_range_CB_quant', 'dynamic_w_range']:
                for idx_inp in range(len(x_inp)):
                    qact = F.linear(x_inp[idx_inp], Qweight, None)
                    Qact.append(qact)
            elif self.training_mode in ['fixed_w_full_range_CB_quant', 'dynamic_w_full_range']:
                if self.dual_array:
                    for idx_inp in range(len(x_inp)):
                        qact_pos = F.linear(x_inp[idx_inp], Qweight_pos, None)
                        qact_neg = F.linear(x_inp[idx_inp], Qweight_neg, None)
                        Qact.append(qact_pos)
                        Qact.append(qact_neg)
                else:
                    for idx_inp in range(len(x_inp)):
                        qact = F.linear(x_inp[idx_inp], Qweight, None)
                        Qact.append(qact)

        self.output_float = F.linear(x, self.weight, None)
    
        return Qact
    

    def scale_weights(self, w_max, w_source = None):
        """
            Scale weights to Bitcell weights
            args: 
                1) w_max: maximum abs value of signed weights; same for positive & nagative array
                2) inp_act_scale: scaling factor for weights to adapt to the new inp & act range
            Set scaling factor for output 
        """
        kernel = self.weight.data
        if w_source == None:
            if self.training_mode == 'fixed_w_range_CB_quant':
                kernel = self.weight.clamp(0)
                kernel = self.weight_quantization(kernel, 0, w_max)
                kernel, scaling_factor = rescale_weight(kernel, 0, w_max, self.lW_static, self.uW_static)
            elif self.training_mode == 'fixed_w_full_range_CB_quant':
                kernel = self.weight_quantization(kernel, -w_max, w_max)
                kernel, scaling_factor = rescale_weight(kernel, -w_max, w_max, self.lW_static, self.uW_static)
            with torch.no_grad():
                self.weight.data = kernel 
            return kernel, scaling_factor
        elif w_source == 'fixed_w_full_range_CB_quant':
            with torch.no_grad():
                kernel += self.lW_static
                kernel = self.weight_quantization(kernel, self.lW_static, self.uW_static)
                self.weight.data = kernel
            return kernel
        
##########################################
### Dual array module 
##########################################
class Dual_Conv2d(nn.Module):
    """
        Dual Array module for Conv2d
        Inp, weight, act quantization are supported
        If dual_array==True: dual_array/dual_column mapping; else: dual_row mapping
        If in tile mode: 
            Input & Act quantization will not be performed in this module; 
            These will be carried out in the tile module;
        No bias, No input/activation scaling
    """
    def __init__(self, args, training_mode = 'dynamic_w_range', dual_array = True, mode = 'training', 
                 tile = False, fan_in = 1, **kwargs):
        super(Dual_Conv2d, self).__init__()
        self.quan_inp = args.QInpFlag
        self.quan_weight = args.QWeightFlag
        self.quan_act = args.QActFlag
        self.inp_signed = args.inp_signed
        
        self.training_mode = training_mode
        self.tile = tile
        self.fan_in = fan_in
        self.dual_array = dual_array # dual_array if true; else dual_row 
        self.mode = mode
        
        self.kwargs = kwargs
        self.in_channels = kwargs['in_channels']
        self.out_channels = kwargs['out_channels']
        self.kernel_size = kwargs['kernel_size']
        
        self.act_max = torch.tensor(0.0) # maximum of activation; used for act_scaling
  
        if tile:
            kwargs['bias'] = False
            self.Flag_bias = kwargs['bias']
            self.fan_in = fan_in
        else:
            self.Flag_bias = kwargs['bias']
            self.fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            stdv = 1. / math.sqrt(self.in_channels)
            if self.Flag_bias:
                self.bias = nn.Parameter(data = torch.empty(self.out_channels), requires_grad=True)
                self.bias.data.uniform_(-stdv, stdv)
                
        kwargs['bias'] = None
        if self.dual_array == True and (self.training_mode in ['fixed_w_range_CB_quant', 'dynamic_w_range']):
            self.dual_module = nn.ModuleList()
            quant_pos = Quant_Conv2d(args, training_mode = training_mode, dual_array = dual_array, mode = mode, tile = self.tile, fan_in = self.fan_in, **kwargs)
            quant_neg = Quant_Conv2d(args, training_mode = training_mode, dual_array = dual_array, mode = mode, tile = self.tile, fan_in = self.fan_in, **kwargs)
            quant_pos.to(gp.device)
            quant_neg.to(gp.device)
            self.dual_module.append(quant_pos)
            self.dual_module.append(quant_neg)
        else:
            self.dual_module = nn.ModuleList()
            quant = Quant_Conv2d(args, training_mode = training_mode, dual_array = dual_array, mode = mode, tile = self.tile, fan_in = self.fan_in, **kwargs)
            quant.to(gp.device)
            self.dual_module.append(quant)
        
        if self.tile==False:
            self.Inp_quant = QInput(args, inp_bits = gp.inp_bits, signed = args.inp_signed, **kwargs)
            self.Act_quant = QAct(args, act_bits = gp.act_bits, signed = args.act_signed, **kwargs)

        self.register_buffer('init', torch.tensor([0]))
    
    def initialize(self, x):
        """
            Initialize uA if not in tile mode 
        """
        if self.tile == False:
            with torch.no_grad():
                Qinp = x
                if self.quan_inp and self.tile==False:
                    Qinp = self.Inp_quant(Qinp)

                Qact_list = []
                if self.dual_array == True and (self.training_mode in ['fixed_w_range_CB_quant', 'dynamic_w_range']):
                    qact_1 = self.dual_module[0].forward(Qinp)
                    qact_2 = self.dual_module[1].forward(Qinp)
                    for i in range(len(qact_1)):
                        Qact_list.append(qact_1[i])
                        Qact_list.append(qact_2[i])
                else:
                    qact = self.dual_module[0].forward(Qinp)
                    Qact_list = qact

                Qact = [((-1)**i)*self.Act_quant(Qact_list[i]) for i in range(len(Qact_list))]

                self.act_max = Qact[0].max().data
                for i in range(1, len(Qact)):
                    if Qact[i].abs().max().data > self.act_max:
                        self.act_max = Qact[i].abs().max().data
                
                if 'relu' not in self.Act_quant.act_quant_type:
                    self.uA.data = self.act_max

    def forward(self, x):
        """
            output:
            - Qact: tensor if not in tile mode; list of tensors if in tile mode. 
        """
        if self.init == 1:
            self.initialize(x)

        Qinp = x
        if self.quan_inp and self.tile==False:
            Qinp = self.Inp_quant(Qinp)
        
        Qact_list = []
        if self.dual_array == True and (self.training_mode in ['fixed_w_range_CB_quant', 'dynamic_w_range']):
            qact_1 = self.dual_module[0].forward(Qinp)
            qact_2 = self.dual_module[1].forward(Qinp)
            for i in range(len(qact_1)):
                Qact_list.append(qact_1[i])
                Qact_list.append(qact_2[i])
        else:
            qact = self.dual_module[0].forward(Qinp)
            Qact_list = qact

        if self.tile == False:
            Qact = [((-1)**i)*self.Act_quant(Qact_list[i]) for i in range(len(Qact_list))]

            self.act_max = Qact[0].max().data
            for i in range(1, len(Qact)):
                if Qact[i].abs().max().data > self.act_max:
                    self.act_max = Qact[i].abs().max().data

            Qact = torch.stack(Qact, dim=0).sum(dim = 0)

            if self.Flag_bias == True:
                Qact += self.bias.reshape(1, Qact.shape[1], 1, 1)
            return Qact
        else:
            return Qact_list 

    def set_weights(self, weight, w_max):
        if self.training_mode=='fixed_w_range_CB_quant':
            weight_pos = weight.clamp(0)
            weight_neg = (-weight).clamp(0)
            with torch.no_grad():
                self.dual_module[0].weight.data = weight_pos
                self.dual_module[1].weight.data = weight_neg
            kernel, scaling_factor = self.dual_module[0].scale_weights(w_max)
            kernel, scaling_factor = self.dual_module[1].scale_weights(w_max)
        elif self.training_mode == 'fixed_w_full_range_CB_quant':
            with torch.no_grad():
                self.dual_module[0].weight.data = weight
            kernel, scaling_factor = self.dual_module[0].scale_weights(w_max)
        return scaling_factor

    
class Dual_Linear(nn.Module):
    """
        Dual Array module for Linear
        Inp, weight, act quantization are supported
        If dual_array==True: dual_array/dual_column mapping
        If in tile mode: 
            Input & Act quantization will not be performed in this module; 
            These will be carried out in the tile module;
        Support bias if not in tile mode
    """
    def __init__(self, args, training_mode = 'dynamic_w_range', dual_array = True, mode = 'training', 
                 tile = False, fan_in = 1, **kwargs):
        super(Dual_Linear, self).__init__()
        self.quan_inp = args.QInpFlag
        self.quan_weight = args.QWeightFlag
        self.quan_act = args.QActFlag
        
        self.training_mode = training_mode
        self.tile = tile
        self.fan_in = fan_in
        self.dual_array = dual_array # dual_array if true; else dual_row 
        self.mode = mode
        
        self.kwargs = kwargs
        self.in_features = kwargs['in_features']
        self.out_features = kwargs['out_features']

        self.act_max = torch.tensor(0.0) # maximum of activation; used for act_scaling
        
        if tile:
            kwargs['bias'] = False
            self.Flag_bias = kwargs['bias']
            self.fan_in = fan_in
        else:
            self.Flag_bias = kwargs['bias']
            self.fan_in = self.in_features
            stdv = 1. / math.sqrt(self.in_features)
            if self.Flag_bias:
                self.bias = nn.Parameter(data = torch.empty(self.out_features), requires_grad=True)
                self.bias.data.uniform_(-stdv, stdv)
        
        kwargs['bias'] = None
        if self.dual_array == True and (self.training_mode in ['fixed_w_range_CB_quant', 'dynamic_w_range']):
            self.dual_module = nn.ModuleList()
            quant_pos = Quant_Linear(args, training_mode = training_mode, dual_array = dual_array, mode = mode, tile = self.tile, fan_in = self.fan_in, **kwargs)
            quant_neg = Quant_Linear(args, training_mode = training_mode, dual_array = dual_array, mode = mode, tile = self.tile, fan_in = self.fan_in, **kwargs)
            quant_pos.to(gp.device)
            quant_neg.to(gp.device)
            self.dual_module.append(quant_pos)
            self.dual_module.append(quant_neg)
        else:
            self.dual_module = nn.ModuleList()
            quant = Quant_Linear(args, training_mode = training_mode, dual_array = dual_array, mode = mode, tile = self.tile, fan_in = self.fan_in, **kwargs)
            quant.to(gp.device)
            self.dual_module.append(quant)
            
        if self.tile==False:
            self.Inp_quant = QInput(args, inp_bits = gp.inp_bits, signed = args.inp_signed, **kwargs)
            self.Act_quant = QAct(args, act_bits = gp.act_bits, signed = args.act_signed, **kwargs)
        
        self.register_buffer('init', torch.tensor([0]))

    def initialize(self, x):
        """
            Initialize uA if not in tile mode 
        """
        if self.tile == False:
            with torch.no_grad():
                Qinp = x
                if self.tile==False:
                    Qinp = self.Inp_quant(Qinp)

                Qact_list = []
                if self.dual_array == True and (self.training_mode in ['fixed_w_range_CB_quant', 'dynamic_w_range']):
                    qact_1 = self.dual_module[0].forward(Qinp)
                    qact_2 = self.dual_module[1].forward(Qinp)
                    for i in range(len(qact_1)):
                        Qact_list.append(qact_1[i])
                        Qact_list.append(qact_2[i])
                else:
                    qact = self.dual_module[0].forward(Qinp)
                    Qact_list = qact

                Qact = [((-1)**i)*self.Act_quant(Qact_list[i]) for i in range(len(Qact_list))]

                self.act_max = Qact[0].max().data
                for i in range(1, len(Qact)):
                    if Qact[i].abs().max().data > self.act_max:
                        self.act_max = Qact[i].abs().max().data
                
                if 'relu' not in self.Act_quant.act_quant_type:
                    self.uA.data = self.act_max

    def forward(self, x):
        """
            output:
            - Qact: tensor if not in tile mode; list of tensors if in tile mode. 
        """
        if self.init == 1:
            self.initialize(x)

        Qinp = x
        if self.quan_inp and self.tile==False:
            Qinp = self.Inp_quant(Qinp)
        
        Qact_list = []
        if self.dual_array == True and (self.training_mode in ['fixed_w_range_CB_quant', 'dynamic_w_range']):
            qact_1 = self.dual_module[0].forward(Qinp)
            qact_2 = self.dual_module[1].forward(Qinp)
            for i in range(len(qact_1)):
                Qact_list.append(qact_1[i])
                Qact_list.append(qact_2[i])
        else:
            qact = self.dual_module[0].forward(Qinp)
            Qact_list = qact

        if self.tile == False:
            Qact = [((-1)**i)*self.Act_quant(Qact_list[i]) for i in range(len(Qact_list))]
            
            self.act_max = Qact[0].max().data
            for i in range(1, len(Qact)):
                if Qact[i].abs().max().data > self.act_max:
                    self.act_max = Qact[i].abs().max().data

            Qact = torch.stack(Qact, dim=0).sum(dim = 0)

            if self.Flag_bias == True:
                Qact += self.bias.reshape(1, Qact.shape[1], 1, 1)
            return Qact
        else:
            return Qact_list
    
    def set_weights(self, weight,w_max):
        if self.training_mode=='fixed_w_range_CB_quant':
            weight_pos = weight.clamp(0)
            weight_neg = (-weight).clamp(0)
            with torch.no_grad():
                self.dual_module[0].weight.data = weight_pos
                self.dual_module[1].weight.data = weight_neg
            kernel, scaling_factor = self.dual_module[0].scale_weights(w_max)
            kernel, scaling_factor = self.dual_module[1].scale_weights(w_max)
        elif self.training_mode == 'fixed_w_full_range_CB_quant':
            print("Mapping weights to CB conductance value")
            with torch.no_grad():
                self.dual_module[0].weight.data = weight
            kernel, scaling_factor = self.dual_module[0].scale_weights(w_max)
        return scaling_factor

#################################################################
#### Tile Module 
#################################################################
class Tile_Conv2d(nn.Module):
    """
        Tile module for Conv2d
        Split weights & inputs to tiles
        Support bias & input/activation scaling
    """
    ###
    """
        To do list: add depth-wise convolution
        Add weight & bias & scaling factor setting function
    """
    ###
    def __init__(self, args, training_mode = 'dynamic_w_full_range', dual_array = True, mode = 'training', 
                 tile = True, fan_in = 1, **kwargs):
        super(Tile_Conv2d, self).__init__()
        self.kwargs = kwargs
        
        self.training_mode = training_mode
        self.dual_array = dual_array # dual_array if true; else dual_row 
        self.mode = mode
        self.quan_inp = args.QInpFlag
        self.inp_signed = args.inp_signed
        self.set_inp_scale = args.set_inp_scale
        self.inp_quant_type = gp.inp_quant_type
        # self.inp_scaling = torch.tensor(1.0)
        self.quan_act = args.QActFlag
        self.act_signed = args.act_signed
        self.set_act_scale = args.set_act_scale
        self.act_quant_type = gp.act_quant_type
        # self.act_scaling = torch.tensor(1.0)
        self.inp_max = torch.tensor(0.0)
        self.act_max = torch.tensor(0.0) 
        self.weight_max_train = gp.weight_max_train
        self.ADC_w_scaling = torch.tensor(1.0/self.weight_max_train) # ADC current range w.r.t. max RRAM current
        self.tile_size = gp.cb_size
        
        self.noise_type = None
        self.noise = args.noisy
        if args.noisy and args.noise_type is not None:
            self.noise_type = args.noise_type
            self.noise_sigma = args.noise_sigma
        
        # Dual array mapping for positive and negative weights
        self.in_channels = kwargs['in_channels']
        self.out_channels = kwargs['out_channels']
        self.kernel_size = kwargs['kernel_size']
        if 'groups' in kwargs.keys():
            self.groups = kwargs['groups']
        else:
            self.groups = 1
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels // self.groups, *self.kernel_size), requires_grad = False)
        self.Flag_bias = kwargs['bias']
        stdv = 1. / math.sqrt(self.in_channels)
        if self.Flag_bias:
            self.bias = nn.Parameter(data = torch.empty(self.out_channels), requires_grad=True)
            self.bias.data.uniform_(-stdv, stdv)

        self.fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.input_channels_per_tile = int(np.floor(self.tile_size[0]/(self.kernel_size[0]*self.kernel_size[1]))) # number of input channels per tile 
        n_rs = int(np.ceil(self.in_channels/self.input_channels_per_tile))
        self.n_rs = n_rs
        print('Conv layer # of tiles:', n_rs)
        input_ch_start = 0 
        input_ch_end = 0 
        
        self.partition_index= []
        self.tile_module = nn.ModuleList()
        
        for i in range(n_rs):
            input_ch_start = i*self.input_channels_per_tile
            input_ch_end = min(input_ch_start+self.input_channels_per_tile, self.in_channels)
            
            self.partition_index.append(([input_ch_start, input_ch_end]))
            
            kwargs_tmp = copy.deepcopy(self.kwargs)
            kwargs_tmp['in_channels'] = input_ch_end - input_ch_start
            CB = Dual_Conv2d(args, training_mode = training_mode, dual_array = dual_array, mode = mode, 
                            tile = True, fan_in = self.fan_in, **kwargs_tmp)
            CB.to(gp.device)
            self.tile_module.append(CB)
        
        self.Inp_quant = QInput(args, inp_bits = gp.inp_bits, signed = args.inp_signed, **kwargs)
        self.Act_quant = QAct(args, act_bits = gp.act_bits, signed = args.act_signed, **kwargs)
        
        self.register_buffer('init', torch.tensor([0]))
    
    def initialize(self, x):
        ## Configure uA  
        with torch.no_grad():
            if self.training_mode in ['fixed_w_full_range_CB_quant', 'fixed_w_range_CB_quant']:
                n_rs = self.n_rs
                if x.abs().max().data>self.inp_max:
                    self.inp_max = x.abs().max().data
                    print("Conv init inp_max:", self.inp_max)
                    self.Inp_quant.uI.data = self.inp_max

                if self.Act_quant.act_quant_type in ['dynamic_act', 'ADC_dynamic']:
                    self.Act_quant.uA.data = self.inp_max * self.ADC_w_scaling
                    print("Conv Init uA:", self.Act_quant.uA.data)

                ### Check initial maximum act max
                Qinp = self.Inp_quant(x)
                input_slice_list = torch.split(Qinp, self.input_channels_per_tile, dim=1)

                for i in range(self.n_rs):
                    qact_list = self.tile_module[i].forward(input_slice_list[i])
                    for j in range(len(qact_list)):
                        if qact_list[j].max().data > self.act_max:
                            self.act_max = qact_list[j].max().data
                print("Conv Init act_max:", self.act_max)

                if self.act_max > self.Act_quant.uA.data:
                    scaling_factor = self.Act_quant.uA.data / self.act_max
                    print("Reduce Conv weights by: %.3f" %(scaling_factor))
                    for i in range(n_rs):
                        for m in self.tile_module[i].dual_module:
                            m.weight.data = m.weight.data * scaling_factor
        
    def forward(self, x):
        if self.init == 1:
            self.initialize(x)
        
        ########################
        ## Inp quant
        Qinp = self.Inp_quant(x)
        input_slice_list = torch.split(Qinp, self.input_channels_per_tile, dim=1)

        ########################
        ## Act quant
        if self.act_quant_type == 'ADC_dynamic':
            with torch.no_grad():
                self.Act_quant.uA.data = self.ADC_w_scaling * torch.min(self.Inp_quant.uI.data, Qinp.abs().max())
        
        output_CB = []
        output_tensor = None
        self.act_max = 0*self.act_max
        n_rs = self.n_rs
        for i in range(n_rs):
            qact_list = self.tile_module[i].forward(input_slice_list[i])
            for j in range(len(qact_list)):
                if qact_list[j].max().data > self.act_max:
                    self.act_max = qact_list[j].max().data
                qact = ((-1)**j) * self.Act_quant(qact_list[j])
                output_CB.append(qact)
        output_tensor = torch.stack(output_CB, dim=0).sum(dim=0)

        if self.Flag_bias:
            output_tensor += self.bias.reshape(1, output_tensor.shape[1], 1, 1)
            
        return output_tensor
    
    def scale_weights(self, w_source):
        kernel = self.weight.data
        w_max = kernel.abs().max()
        if w_source in ['quant', 'float']:
            # Load from non CB model, map weights to tiles 
            weight_split = torch.split(kernel, self.input_channels_per_tile, dim=1)
            for i in range(self.n_rs):
                weight_scaling = self.tile_module[i].set_weights(weight_split[i], w_max)
            scaling_factor = weight_scaling/self.uA
            print('weight_scaling:', weight_scaling,'; wmax:', w_max, '; scaling_factor:', scaling_factor)
            
        with torch.no_grad():
            if 'bias' in self.kwargs.keys() and self.kwargs['bias']==True:
                self.bias.data = self.bias.data / scaling_factor
            if self.set_act_scale:
                self.weight_scaling.data = weight_scaling
                self.act_scaling.data = scaling_factor 

class Tile_Linear(nn.Module):
    """
        Tile module for Linear
        Split weights & inputs to tiles
        Support bias & input/activation scaling
    """
    def __init__(self, args, training_mode = 'dynamic_w_full_range', dual_array = True, mode = 'training', 
                 tile = True, fan_in = 1, **kwargs):
        super(Tile_Linear, self).__init__()
        self.kwargs = kwargs
        
        self.training_mode = training_mode
        self.dual_array = dual_array # dual_array if true; else dual_row 
        self.mode = mode
        self.quan_inp = args.QInpFlag
        self.inp_signed = args.inp_signed
        self.set_inp_scale = args.set_inp_scale
        self.inp_quant_type = gp.inp_quant_type
        # self.inp_scaling = torch.tensor(1.0)
        self.quan_act = args.QActFlag
        self.act_signed = args.act_signed
        self.set_act_scale = args.set_act_scale
        self.act_quant_type = gp.act_quant_type
        # self.act_scaling = torch.tensor(1.0)
        self.inp_max = torch.tensor(0.0)
        self.act_max = torch.tensor(0.0) 
        self.weight_max_train = gp.weight_max_train
        self.ADC_w_scaling = torch.tensor(1.0/self.weight_max_train) # ADC current range w.r.t. max RRAM current
        self.tile_size = gp.cb_size
        
        self.noise_type = None
        self.noise = args.noisy
        if args.noisy and args.noise_type is not None:
            self.noise_type = args.noise_type
            self.noise_sigma = args.noise_sigma
        
        # Dual array mapping for positive and negative weights
        self.in_features = kwargs['in_features']
        self.out_features = kwargs['out_features']
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features), requires_grad = False)
        self.Flag_bias = kwargs['bias']
        stdv = 1. / math.sqrt(self.in_features)
        if self.Flag_bias:
            self.bias = nn.Parameter(data = torch.empty(self.out_features), requires_grad=True)
            self.bias.data.uniform_(-stdv, stdv)

        self.fan_in = self.in_features
        self.input_channels_per_tile = int(self.tile_size[0]) # number of input channels per tile 
        n_rs = int(np.ceil(self.in_features/self.input_channels_per_tile))
        self.n_rs = n_rs
        print('Linear layer # of tiles:', n_rs)
        input_ch_start = 0 
        input_ch_end = 0 
        
        self.partition_index= []
        self.tile_module = nn.ModuleList()
        for i in range(n_rs):
            input_ch_start = i*self.input_channels_per_tile
            input_ch_end = min(input_ch_start+self.input_channels_per_tile, self.in_features)
            
            self.partition_index.append(([input_ch_start, input_ch_end]))
            
            kwargs_tmp = copy.deepcopy(self.kwargs)
            kwargs_tmp['in_features'] = input_ch_end - input_ch_start
            CB = Dual_Linear(args, training_mode = training_mode, dual_array = dual_array, mode = mode, 
                            tile = True, fan_in = self.fan_in, **kwargs_tmp)
            CB.to(gp.device)
            self.tile_module.append(CB)
            
        self.Inp_quant = QInput(args, inp_bits = gp.inp_bits, signed = args.inp_signed, **kwargs)
        self.Act_quant = QAct(args, act_bits = gp.act_bits, signed = args.act_signed, **kwargs)
        
        self.register_buffer('init', torch.tensor([0]))

    def initialize(self, x):
        ## Configure uA  
        with torch.no_grad():
            if self.training_mode in ['fixed_w_full_range_CB_quant', 'fixed_w_range_CB_quant']:
                n_rs = self.n_rs
                if x.abs().max().data>self.inp_max:
                    self.inp_max = x.abs().max().data
                    self.Inp_quant.uI.data = self.inp_max
                    print("FC init inp_max:", self.inp_max)

                if self.Act_quant.act_quant_type in ['dynamic_act', 'ADC_dynamic']:
                    self.Act_quant.uA.data = self.inp_max * self.ADC_w_scaling
                    print("FC Init uA:", self.Act_quant.uA.data)

                ### Check initial maximum act max
                Qinp = self.Inp_quant(x)
                input_slice_list = torch.split(Qinp, self.input_channels_per_tile, dim=1)

                for i in range(self.n_rs):
                    qact_list = self.tile_module[i].forward(input_slice_list[i])
                    for j in range(len(qact_list)):
                        if qact_list[j].max().data > self.act_max:
                            self.act_max = qact_list[j].max().data
                print("FC Init act_max:", self.act_max)

                if self.act_max > self.Act_quant.uA.data:
                    scaling_factor = self.Act_quant.uA.data / self.act_max
                    print("Reduce FC weights by: %.3f" %(scaling_factor))
                    for i in range(n_rs):
                        for m in self.tile_module[i].dual_module:
                            m.weight.data = m.weight.data * scaling_factor

    def forward(self, x):
        if self.init == 1:
            self.initialize(x)
        
        ########################
        ## Inp quant
        Qinp = self.Inp_quant(x)
        input_slice_list = torch.split(Qinp, self.input_channels_per_tile, dim=1)

        ########################
        ## Act quant
        if self.act_quant_type == 'ADC_dynamic':
            with torch.no_grad():
                self.Act_quant.uA.data = self.ADC_w_scaling * torch.min(self.Inp_quant.uI.data, Qinp.abs().max())

        output_CB = []
        output_tensor = None
        self.act_max = 0*self.act_max
        n_rs = self.n_rs
        for i in range(n_rs):
            qact_list = self.tile_module[i].forward(input_slice_list[i])
            for j in range(len(qact_list)):
                if qact_list[j].max().data > self.act_max:
                    self.act_max = qact_list[j].max().data
                qact = ((-1)**j) * self.Act_quant(qact_list[j])
                output_CB.append(qact)
        output_tensor = torch.stack(output_CB, dim=0).sum(dim=0)

        if self.Flag_bias:
            output_tensor += self.bias.reshape(1, output_tensor.shape[1], 1, 1)
            
        return output_tensor
        
    def scale_weights(self, w_source):
        kernel = self.weight.data
        w_max = kernel.abs().max()
        if w_source in ['quant', 'float']:
            # Load from non CB model, map weights to tiles 
            weight_split = torch.split(kernel, self.input_channels_per_tile, dim=1)
            for i in range(self.n_rs):
                weight_scaling = self.tile_module[i].set_weights(weight_split[i], w_max)
            scaling_factor = weight_scaling/self.uA
            print('weight_scaling:', weight_scaling,'; wmax:', w_max, '; scaling_factor:', scaling_factor)
            
        with torch.no_grad():
            if 'bias' in self.kwargs.keys() and self.kwargs['bias']==True:
                self.bias.data = self.bias.data / scaling_factor
            if self.set_act_scale:
                self.weight_scaling.data = weight_scaling
                self.act_scaling.data = scaling_factor 