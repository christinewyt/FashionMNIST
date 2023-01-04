import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import torch
import torch.nn as nn
from brevitas.nn import QuantIdentity, QuantConv2d, QuantLinear
from models.reparam_layers import QRPConv2d, QRPLinear
from collections import OrderedDict
from utils.misc import get_bytesio, print_dict
import copy


def quant_aware_wrapper(model_in, weight_bit=8, act_bit=8, noise_aware=False, **kwargs):
    '''
    Create a wrapper function to convert model layers to quant aware training layers,
    and include quantization for layer activations.
    
    Args:
        model_in: model input
        weight_bit: bitwidth for weight quantization
        act_bit: bitwidth for activation quantization
    
    Returns:
        qmodel: model with quantization aware layers
    '''
    
    
    if noise_aware:
        print(f'Create noise-aware model.')
        print_dict(kwargs)
    quant_cov2d = QuantConv2d if not noise_aware else QRPConv2d
    quant_linear = QuantLinear if not noise_aware else QRPLinear
    
    
    def replace_w_quantlayer(model, layername):
        qmodel = copy.deepcopy(model)
        
        for name, layer in list(qmodel.named_children()):
            new_layer = []
            sequential = False
            nested_module = False
            is_trainable = len([i for i in layer.parameters() if i.requires_grad]) > 0

            if isinstance(layer, nn.Conv2d):
                new_layer.append(quant_cov2d(in_channels=layer.in_channels, out_channels=layer.out_channels,
                                           kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding,
                                           dilation=layer.dilation, groups=layer.groups, padding_mode = layer.padding_mode,
                                           bias=True if layer.bias is not None else False,
                                           weight_bit_width=weight_bit, **kwargs))
                print('number of weight bits:', new_layer[-1].quant_weight_bit_width().int().item())

            elif isinstance(layer, nn.Linear):
                new_layer.append(quant_linear(in_features=layer.in_features, out_features=layer.out_features,
                                             bias=True if layer.bias is not None else False,
                                             weight_bit_width=weight_bit, **kwargs))

            elif isinstance(layer, nn.Sequential):
                sequential = True
                # to capture nested modules in nn.Sequential
                for subname, sublayer in layer.named_children():
                    if len(list(sublayer.children())) == 0 and len(list(sublayer.parameters())) > 0:
                        nested_module = True
                        sublayer = nn.Sequential(sublayer)
                    new_layer.append(replace_w_quantlayer(sublayer, layername))

            else:
                new_layer.append(layer)
                
            # do not apply quantization to 
            # 1) non-trainable layer
            # 2) the activation of last trainable layer
            # 3) end of nested sequential modules
            if name != layername and is_trainable and not sequential and not nested_module:
                new_layer.append(QuantIdentity(bit_width=act_bit))

            new_layer = nn.Sequential(*new_layer)
            setattr(qmodel, name, new_layer)
        return qmodel
    
    
    # get the last trainable layer before ouput
    trainable_params = [param[0] for param in model_in.named_parameters()]
    output_layer = trainable_params[-1].split('.')[0]
    qmodel = replace_w_quantlayer(model_in, output_layer)
    qmodel = nn.Sequential(QuantIdentity(act_bit=act_bit), qmodel)
    
    return qmodel



def load_qaware_model(ref_model, qmodel_name, weight_bit, act_bit, file_source='./', map_location=None, strict=True):
    '''
    Load quantization aware model files based on the given quantization bitwidth.
    
    Args:
        ref_model: reference model structure
        qmodel_name : name of quantization-aware model files (*.pt)
        weight_bit: weight quantization bitwidth
        act_bit: activation quantization bitwidth for the test
        file_source: 
            - (OrderedDict) model state_dict object
            - (str) model file directory path
            - (dict) dictionary containing model files fetched from NAS using AMAT NASAPI, i.e. {filename: io_file}
    
    Returns:
        qmodel: quantization aware model
    
    '''
    
    qmodel = quant_aware_wrapper(ref_model, weight_bit, act_bit)
    
    if isinstance(file_source, OrderedDict):
        state_dict = file_source
        
    elif isinstance(file_source, dict):
        name_key = qmodel_name + '.pt'
        if name_key not in file_source.keys():
            print(f'\n==> {name_key} is not found in the given model_path (check config.py).\n')
            return None
        state_dict = torch.load(get_bytesio(file_source[name_key]), map_location=map_location)
    
    elif isinstance(file_source, str):
        state_dict = torch.load(os.path.join(file_source, qmodel_name+'.pt'), map_location=map_location)
        state_dict = state_dict['state_dict'] if state_dict.get('state_dict', None) else state_dict
    
    else:
        print('file_source provided is not valid.')
        return None
        
    qmodel.load_state_dict(state_dict, strict=strict)
    return qmodel