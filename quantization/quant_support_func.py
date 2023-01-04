import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from Quant_Layer_Warrior import QInput, QAct, Quant_Conv2d, Quant_Linear, Dual_Conv2d, Dual_Linear, Tile_Conv2d, Tile_Linear

def quant_aware_wrapper(model_in, args, gp, input_quant = False, output_signed = True):
    '''
    Create a wrapper function to convert model layers to quant aware training layers,
    and include quantization for layer activations.
    
    Args:
        model_in: model input
        args: training parameters (whether to add noise to weight or act, dual array, tiled structure, etc.)
        gp: cross-bar parameters
        input_quant: whether add quantization to input for first layer
        output_signed: whether output of last layer is signed or not
    
    Returns:
        qmodel: model with hardware aware layers
    '''
    
    # create the args and gp copy
    args = copy.deepcopy(args)
    gp = copy.deepcopy(gp)
    
    noise_aware = args.noisy
    if args.noisy == 0:
        noise_type = None
    else:
        noise_dict = {'noise_type':gp.weight_noise_type, 'noise_sigma': gp.weight_noise_sigma}
        
    tile_aware = args.tile
    if tile_aware:
        print(f'########## Create tile-aware model ##########')
        if args.noisy:
            print(f'########## Create noise-aware model ##########')
            print(noise_dict)
    elif noise_aware:
        print(f'########## Create noise-aware model ##########')
        print(noise_dict)
        
    if args.dual_array:
        print('########## Dual array mapping ##########')
    else:
        print('########## Differential row mapping ##########')
      
    if tile_aware:
        quant_conv2d = Tile_Conv2d
        quant_linear = Tile_Linear
    else:
        quant_conv2d = Dual_Conv2d
        quant_linear = Dual_Linear

    def DFS(model, layer_list):
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                layer_list.append(layer)
                
    cnt = 0
    Act_signed = args.act_signed
    Inp_signed = args.inp_signed
   
    
    def replace_w_quantlayer(model, layer_list, input_quant=False, output_signed=False):
        """
            Replace the layers with customized quant layers
            args: 
                1) Model: reference model to be replaced
                2) input_quant: whether to add input quantization to first layer
                3) output_quant: whether to add output quantization to last layer
            return: model with customized quantized layers
        """
        qmodel = copy.deepcopy(model)
        
        nonlocal cnt
        
        ## Don't force output of last layer to be positive
        for name, layer in list(qmodel.named_children()):
            new_layer = []
            sequential = False
         
            if isinstance(layer, nn.Conv2d):
                print("---------------------------------")
                print(cnt, layer)
                print("Inp_quant:", args.QInpFlag, ";inp_quant_type:", args.inp_quant_type,";inp_signed:", args.inp_signed, "; inp_scale:", args.set_inp_scale)
                print("Act_quant:", args.QActFlag, ";act_quant_type:", args.act_quant_type, ";act_signed:", args.act_signed, "; act_scale:", args.set_act_scale)
                print("---------------------------------")
                new_layer.append(quant_conv2d(args = copy.copy(args), gp = copy.copy(gp), training_mode = args.training_mode, dual_array=args.dual_array, mode = args.mode,
                                           tile = args.tile, fan_in = 1,
                                           in_channels=layer.in_channels, out_channels=layer.out_channels,
                                           kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding,
                                           dilation=layer.dilation, groups=layer.groups, 
                                           bias= False,
                                            ))
                cnt += 1

            elif isinstance(layer, nn.Linear):
                if cnt == len(layer_list)-1 and output_signed==True:
                    args.act_signed = True
                print("---------------------------------")
                print(cnt, layer)
                print("Inp_quant:", args.QInpFlag, "; inp_quant_type:", args.inp_quant_type,"; inp_signed:", args.inp_signed, "; inp_scale:", args.set_inp_scale)
                print("Act_quant:", args.QActFlag, "; act_quant_type:", args.act_quant_type, "; act_signed:", args.act_signed, "; act_scale:", args.set_act_scale)
                print("---------------------------------")
                new_layer.append(quant_linear(args = copy.copy(args), gp = copy.copy(gp), training_mode = args.training_mode, dual_array=args.dual_array, mode = args.mode,
                                             tile = args.tile, fan_in = 1,
                                             in_features=layer.in_features, out_features=layer.out_features,
                                             bias=False,
                                             ))
                args.act_signed = Act_signed
                cnt += 1
            elif isinstance(layer, nn.Sequential):
                for subname, sublayer in layer.named_children():
                    if len(list(sublayer.children())) == 0 and len(list(sublayer.parameters())) > 0:
                        sublayer = nn.Sequential(sublayer)
                    new_layer.append(replace_w_quantlayer(sublayer, layer_list, input_quant=input_quant, output_signed=output_signed))
            else:
                new_layer.append(layer)
                
            
            # do not apply quantization to 
            # 1) non-trainable layer
            # 2) the activation of last trainable layer
            # 3) end of nested sequential modules
            new_layer = nn.Sequential(*new_layer)
            setattr(qmodel, name, new_layer)
        return qmodel
    

    layer_list = []
    DFS(model_in, layer_list)
    print(layer_list)
    print('Number of conv layers:', len(layer_list))
    
    qmodel = replace_w_quantlayer(model_in, layer_list, input_quant = input_quant, output_signed = output_signed)
    
    if args.data == 'mnist':
        args.inp_signed = False
    elif args.data == 'cifar10':
        args.inp_signed = True
    
    return qmodel

def init_quant_model(model, train_loader, device):
    for m in model.modules():
        if isinstance(m, (Quant_Conv2d, Quant_Linear)):
            m.init.data.fill_(1)
        elif isinstance(m, (Dual_Conv2d, Dual_Linear)):
            m.init.data.fill_(1)
        elif isinstance(m, (Tile_Conv2d, Tile_Linear)):
            m.init.data.fill_(1)
        elif isinstance(m, QInput):
            m.init.data.fill_(1)
            
    images, labels = next(iter(train_loader))
    images = images.to(device)
    
    model.train()
    model.forward(images)
        
    for m in model.modules():
        if isinstance(m, (Quant_Conv2d, Quant_Linear)):
            m.init.data.fill_(0)
        elif isinstance(m, (Dual_Conv2d, Dual_Linear)):
            m.init.data.fill_(0)
        elif isinstance(m, (Tile_Conv2d, Tile_Linear)):
            m.init.data.fill_(0)
        elif isinstance(m, QInput):
            m.init.data.fill_(0)
    
    return 
