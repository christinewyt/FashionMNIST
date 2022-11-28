import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from pathlib import Path
# import brevitas.nn as bnn
import re
import copy
import pickle


def count_parameters(model):
    ''' Return total trainable parameters in the model '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_report(model_in, model_name='lenet'):
    model = copy.deepcopy(model_in)
    layer_params = OrderedDict()

    def traversal(module, append_name='', layer_i=0):
        for name, layer in module.named_children():
            name = append_name +'_'+ name
            
            if isinstance(layer, (bnn.QuantConv1d, bnn.QuantConv2d)):
                if isinstance(layer, bnn.QuantConv1d):
                    layer_type = 'Conv1d'
                else:
                    layer_type = 'Conv2d'

                # key = f'{layer_type}_{layer_i}'
                key = layer_type + name
                layer_params[key] = pd.Series({
                    'num_params': count_parameters(layer),
                    'num_filter':layer.out_channels,
                    'kernel_size':layer.kernel_size,
                    'stride':layer.stride,
                    # 'padding':layer.padding,
                })

            elif isinstance(layer,bnn.QuantLinear):
                layer_type = 'Linear'

                # key = f'{layer_type}_{layer_i}'
                key = layer_type + name
                layer_params[key] = pd.Series({
                    'num_params': count_parameters(layer),
                    # 'in_features':layer.in_features,
                    # 'out_features':layer.out_features,
                    'num_filter':layer.out_features,
                })

            elif isinstance(layer, nn.Sequential) and not isinstance(layer, bnn.QuantIdentity):
                traversal(layer, name, layer_i)

            elif len(list(layer.children())) > 0 and not isinstance(layer, bnn.QuantIdentity):
                # to capture nested modules in nn.Sequential
                layer = nn.Sequential(layer)
                for subname, sublayer in layer.named_children():
                    name = name+'_'+subname
                    traversal(sublayer, name, layer_i)

            layer_i += 1

    for idx, module in model.named_children():
        # skip the input quantization layer
        traversal(module)
        
    #######################
    benchmark_file_path = '../models/benchmark_image_classification.pkl'
    model_results = pickle.load(open(benchmark_file_path, 'rb'))
    valid_keys = [k for k in model_results if k.lower().startswith(model_name)]

    def extract_result(df, noise_scale=0):
        output = df.loc[df.index==noise_scale, df.columns.str.contains('mean')]
        output.columns = [c.rstrip('_mean') for c in output.columns]
        output.sort_index(1, ascending=False, inplace=True)
        return output

    results = []
    for key in valid_keys:
        results.append(extract_result(model_results[key]))
    results = pd.concat(results, sort=False)
    results.index = valid_keys
    print(results)
    
    dfs = pd.DataFrame()
    for name, df in layer_params.items():
        df.name = name
        dfs = pd.concat([dfs, df], axis=1, sort=False)
    dfs.fillna('', inplace=True)

    return dfs.T.reset_index()



def model_summary(model):
    
    level = ' |'
    
    def get_layername(layer):
        if hasattr(layer, '__class__'):
            layer_class = str(layer.__class__)
        else:
            return str(layer)
        get_class = re.compile(r"\'(.+?)\'")
        _class = get_class.findall(layer_class)[0]
        _layer = _class.split('.')[-1]
        return _layer
    
    def traversal(module, i):
        i += 1
        if len(list(module.children())) == 0:
            print(level*i, get_layername(module), f': {count_parameters(module)}')
            
            
        elif len(list(module.children())) > 0:
            for name, layer in module.named_children():
                print(level*i, f'({name})', get_layername(layer), f': {count_parameters(layer)}')
                if isinstance(layer, nn.Sequential):
                    traversal(layer, i)
                # elif not isinstance(layer, (bnn.QuantConv2d, bnn.QuantLinear, bnn.QuantIdentity)):
                #     # to capture nested modules in nn.Sequential
                #     layer = nn.Sequential(layer)
                #     for j, (subname, sublayer) in enumerate(layer.named_children()):
                #         traversal(sublayer, i+j)
                            
    print(' _')
    for idx, module in model.named_children():
        traversal(module, 0)
        

def get_same_padding(in_size, filter_size, stride=1):
    
    '''
    Function which returns the padding dimension to sustain same size after Conv2d
    
    Args:
        in_size: input image size
        filter_size: the size of the kernel
        stride: steps to skip for convolution
    
    Returns:
        pad_left, pad_right, pad_top, pad_bottom: dimensions to pad for padding function
    '''
    
    in_height, in_width = in_size
    filter_height, filter_width = filter_size, filter_size
    strides = (None, stride, stride)
    out_height = np.ceil(float(in_height) / float(strides[1]))
    out_width  = np.ceil(float(in_width) / float(strides[2]))

    #The total padding applied along the height and width is computed as:

    if (in_height % strides[1] == 0):
        pad_along_height = max(filter_height - strides[1], 0)
    else:
        pad_along_height = max(filter_height - (in_height % strides[1]), 0)
    if (in_width % strides[2] == 0):
        pad_along_width = max(filter_width - strides[2], 0)
    else:
        pad_along_width = max(filter_width - (in_width % strides[2]), 0)

    #Finally, the padding on the top, bottom, left and right are:
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return pad_left, pad_right, pad_top, pad_bottom


def train_model(model, device, trainloader, optimizer, loss_fn, epoch, show_steps=0, lr_scheduler=None):
        
    '''
    Function which trains the neural network model
    
    Args:
    
        model : model for training
        device : cpu or gpu
        train_loader: the training set data loader
        optimizer : optimizer for the model
        loss_fn : criterion to optimize on (loss function)
        epoch : total number of epochs to train
        show_steps : display training status (number of steps to show or 0 to disable)
    
    Returns:
        loss : Returns loss value 
    '''
    
    model.train()
    total_step = len(trainloader)
    losses = []
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        if show_steps and (i+1) % show_steps == 0:
            print(f'-- epoch: {epoch+1}, step: {i+1}/{total_step}, loss: {loss.item():.4f}')
            
    return loss.item()


def eval_model(model, device, testloader, loss_fn=None):
    
    '''
    Function which evaluates the neural network model
    
    Args:
    
        model : model for evaluating
        device : cpu or gpu
        test_loader: the testing set data loader
        loss_fn : criterion to optimize on (loss function)
    
    Returns:
        mean_loss, accuracy : Returns mean loss and accuracy of the model on testing dataset 
    '''
    
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(testloader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            
            if loss_fn is not None:
                loss_eval = loss_fn(outputs, labels)
                losses.append(loss_eval.item())
                
    if not loss_fn:
        mean_loss = -1
    else:
        mean_loss = np.mean(losses)
        
    accuracy = correct.cpu().numpy()/total
    return mean_loss, accuracy


def save_checkpoint(state, checkpoint_dir='./checkpoints', name='best_checkpoint'):
    '''
    Saves model and training parameters at checkpoint + 'last.pth.tar'
    
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer, state_dict
        checkpoint_dir: folder where parameters are to be saved
    '''
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    filepath = checkpoint_dir / f'{name}.pth.tar'
    torch.save(state, filepath)
    
