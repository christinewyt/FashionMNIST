import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import torch.nn as nn
import torch.nn.functional as F
import brevitas
import copy


def quantize_fn(model):
    '''
    Function which quantise the model
    
    Args:
        model : model for quantization
    
    Returns:
        model_edit : Returns quantised model which has floating point quantised weights
    '''
    
    model_edit = copy.deepcopy(model)
    
    def traversal(module):
        ''' 
        Traverse the model structure and replace original weights to floating point quantized weights.
        '''
        for name, layer in module.named_children():
           
            if isinstance(layer, (brevitas.nn.QuantConv2d, brevitas.nn.QuantLinear)) and hasattr(layer, "quant_weight"):
                q_wt = layer.quant_weight()[0]
                st_dct = layer.state_dict()
                st_dct['weight'] = q_wt
                layer.load_state_dict(st_dct)
                
            elif isinstance(layer, nn.Sequential) and not isinstance(layer, brevitas.nn.QuantIdentity):
                traversal(layer)
            
            elif len(list(layer.children())) > 0 and not isinstance(layer, brevitas.nn.QuantIdentity):
                # to capture nested modules in nn.Sequential
                for subname, sublayer in layer.named_children():
                    # print('*', name, '**', subname)
                    traversal(sublayer)
    
    
    for idx, module in model_edit.named_children():
        # skip the input quantization layer
        if isinstance(module, brevitas.nn.QuantIdentity):
            continue
            
        traversal(module)

    return model_edit


def quantize_fn_int(model):
    '''
    Function which quantise the model
    
    Args:
        model : model for quantization
    
    Returns:
        model_edit : Returns quantised model which has quantised integer weights
        name_scale : This dictionary has quantise aware scaling values (important for reverting back to float)
    '''
    
    name_scale = dict()
    model_edit = copy.deepcopy(model)
    
    def traversal(module, datamap):
        ''' 
        Traverse the model structure and do 2 things in place
            - replace original weights to integer quantized weights
            - create a hash table for corresponding quantization scales of layers        
        '''
        for name, layer in module.named_children():
           
            if isinstance(layer, (brevitas.nn.QuantConv2d, brevitas.nn.QuantLinear)) and hasattr(layer, "int_weight"):
                datamap[name] = layer.quant_weight().scale
                int_wt = layer.int_weight()
                st_dct = layer.state_dict()
                # check if bias was quantized (to be tested)
                if layer._cached_bias is not None:
                    datamap[name+'_bias'] = layer.quant_bias().scale
                    int_bias = layer.int_bias()
                    st_dct['bias'] = int_bias
                st_dct['weight'] = int_wt
                layer.load_state_dict(st_dct)
                
            elif isinstance(layer, nn.Sequential) and not isinstance(layer, brevitas.nn.QuantIdentity):
                datamap[name] = dict()
                traversal(layer, datamap[name])
            
            elif len(list(layer.children())) > 0 and not isinstance(layer, brevitas.nn.QuantIdentity):
                # to capture nested modules in nn.Sequential
                datamap[name] = dict()
                for subname, sublayer in layer.named_children():
                    # print('*', name, '**', subname)
                    datamap[name][subname] = dict()
                    traversal(sublayer, datamap[name][subname])
    
    
    for idx, module in model_edit.named_children():
        # skip the input quantization layer
        if isinstance(module, brevitas.nn.QuantIdentity):
            continue
            
        traversal(module, name_scale)

    return model_edit, name_scale


def quantize_fn_revert(model, name_scale):
    '''
    Function which quantise the model
    
    Args:
        model : model for quantization
    
    Returns:
        model_edit : Returns quantised model which has quantised integer weights
        name_scale : This dictionary has quantise aware scaling values (important for reverting back to float)
    '''
    
    model_edit = copy.deepcopy(model)
    
    def traversal(module, datamap):
        ''' 
        Traverse the model structure and do 2 things in place
            - replace original weights to integer quantized weights
            - create a hash table for corresponding quantization scales of layers        
        '''
        for name, layer in module.named_children():
            
            if isinstance(layer, (brevitas.nn.QuantConv2d, brevitas.nn.QuantLinear)) and hasattr(layer, "quant_weight_scale"):
                st_dct = layer.state_dict()
                st_dct['weight'] = st_dct['weight'] * datamap[name]
                # check if bias was quantized (to be tested)
                if layer._cached_bias is not None:
                    st_dct['bias'] = st_dct['bias'] * datamap[name+'_bias']
                layer.load_state_dict(st_dct)
                
            elif isinstance(layer, nn.Sequential) and not isinstance(layer, brevitas.nn.QuantIdentity):
                traversal(layer, datamap[name])
            
            elif len(list(layer.children())) > 0 and not isinstance(layer, brevitas.nn.QuantIdentity):
                # to capture nested modules in nn.Sequential
                for subname, sublayer in layer.named_children():
                    traversal(sublayer, datamap[name][subname])   
    
    for idx, module in model_edit.named_children():
        # skip the input quantization layer
        if isinstance(module, brevitas.nn.QuantIdentity):
            continue
            
        traversal(module, name_scale)

    return model_edit


def quantize_fn_revert_split(model, name_scale):
    '''
    Revert int_weight to quant_weight with split positive and negative quantization layers
    
    Args:
        model : model for quantization
    
    Returns:
        model_edit : Returns quantised model which has quantised integer weights
        name_scale : This dictionary has quantise aware scaling values (important for reverting back to float)
    '''
    
    model_edit = copy.deepcopy(model)
    
    def traversal(module, datamap):
        
        for name, layer in module.named_children():
            if isinstance(layer, (brevitas.nn.QuantConv2d, brevitas.nn.QuantLinear)) and hasattr(layer, "quant_weight_scale"):
                st_dct = layer.state_dict()
                st_dct['weight'] = st_dct['weight'] * datamap[name]
                layer.load_state_dict(st_dct)
                
            elif isinstance(layer, nn.Sequential) and not isinstance(layer, brevitas.nn.QuantIdentity):
                traversal(layer, datamap[name])
            
            elif len(list(layer.children())) > 0 and not isinstance(layer, brevitas.nn.QuantIdentity):
                # to capture nested modules in nn.Sequential
                for subname, sublayer in layer.named_children():
                    # reuse the same name_scale for split quant layers
                    if subname in ['layer_pos', 'layer_neg']:
                        st_dct = sublayer.state_dict()
                        if not isinstance(datamap[name], dict):
                            # pos & neg split layer with datamap from single quant layer
                            if st_dct.get('weight') is not None:
                                st_dct['weight'] = st_dct['weight'] * datamap[name]
                            else:
                                st_dct['0.weight'] = st_dct['0.weight'] * datamap[name]
                        else:
                            # pos & neg split layer with datamap from pos & neg split layer
                            st_dct['0.weight'] = st_dct['0.weight'] * datamap[name][subname]['0']
                        sublayer.load_state_dict(st_dct)
                    else:
                        traversal(sublayer, datamap[name][subname])
                        
    
    for idx, module in model_edit.named_children():
        # skip the input quantization layer
        if isinstance(module, brevitas.nn.QuantIdentity):
            continue
            
        traversal(module, name_scale)

    return model_edit