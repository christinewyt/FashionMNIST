import numpy as np
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required
from typing import List, Optional
from quantization_new.Quant_Layer_Warrior import QInput, QAct, Quant_Conv2d, Quant_Linear, Dual_Conv2d, Dual_Linear, Tile_Conv2d, Tile_Linear
import math
import warnings
warnings.filterwarnings('ignore')

class SGD_custom(Optimizer):
    def __init__(self, params, copy_params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        """
        Perform mixed precision training 
        Two copies of weights:
            1. params store low precision weights: low precision weights are used in forward path
            2. copy_params store high precision weights: high precision weights for weight update
         
        Weight update are computed based on params (low precision weights) 
        However, weight update is performed on copy_params, not on params !!!!!!!!!!!!!!!!
        
        Gradients are computed every batch with standard SGD

        args: 
            params: low precision weights
            copy_params: high precision weights 
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
            
        copy_param_groups = list(copy_params)
        self.copy_param_groups = []
        if not isinstance(copy_param_groups[0], dict):
            copy_param_groups = [{'params': copy_param_groups}]

        for copy_param_group in copy_param_groups:
            self.add_copy_param_group(copy_param_group)
        
        super(SGD_custom, self).__init__(params, defaults)
    
    def add_copy_param_group(self, copy_param_group):
        """
        Add a param group to the :class:`Optimizer` s `param_groups`.
        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.
        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        assert isinstance(copy_param_group, dict), "copy param group must be a dict"

        params = copy_param_group['params']
        if isinstance(params, torch.Tensor):
            copy_param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            copy_param_group['params'] = list(params)

        for param in copy_param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            #if not param.is_leaf:
            #    raise ValueError("can't optimize a non-leaf Tensor")

#         for name, default in self.defaults.items():
#             if default is required and name not in param_group:
#                 raise ValueError("parameter group didn't specify a value of required optimization parameter " +
#                                  name)
#             else:
#                 param_group.setdefault(name, default)

        params = copy_param_group['params']
        if len(params) != len(set(params)):
            warnings.warn("optimizer contains a parameter group with duplicate parameters; "
                          "in future, this will cause an error; "
                          "see github.com/pytorch/pytorch/issues/40967 for more information", stacklevel=3)

        param_set = set()
        for group in self.copy_param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(copy_param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.copy_param_groups.append(copy_param_group)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for i in range(len(self.param_groups)):
            group = self.param_groups[i]
            group_copy = self.copy_param_groups[i]
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for j in range(len(group['params'])):
                p = group['params'][j]
                p_copy = group_copy['params'][j]
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                ###################################
                ## Modification
                p_copy.add_(d_p, alpha=-group['lr'])

        return loss


class AdamW_custom(Optimizer):
    def __init__(self, params, copy_params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, *, maximize: bool = False,
                 foreach: Optional[bool] = None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        foreach=foreach, maximize=maximize)
        super(AdamW_custom, self).__init__(params, defaults)
        
        copy_param_groups = list(copy_params)
        self.copy_param_groups = []
        if not isinstance(copy_param_groups[0], dict):
            copy_param_groups = [{'params': copy_param_groups}]

        for copy_param_group in copy_param_groups:
            self.add_copy_param_group(copy_param_group)
        
    def add_copy_param_group(self, copy_param_group):
        """
        Add a param group to the :class:`Optimizer` s `param_groups`.
        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.
        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        assert isinstance(copy_param_group, dict), "copy param group must be a dict"

        params = copy_param_group['params']
        if isinstance(params, torch.Tensor):
            copy_param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            copy_param_group['params'] = list(params)

        for param in copy_param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
    
        params = copy_param_group['params']
        if len(params) != len(set(params)):
            warnings.warn("optimizer contains a parameter group with duplicate parameters; "
                          "in future, this will cause an error; "
                          "see github.com/pytorch/pytorch/issues/40967 for more information", stacklevel=3)

        param_set = set()
        for group in self.copy_param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(copy_param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.copy_param_groups.append(copy_param_group)
        
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))


    def _single_tensor_adamw(self, params: List[Tensor],
                             params_copy: List[Tensor],
                             grads: List[Tensor],
                             exp_avgs: List[Tensor],
                             exp_avg_sqs: List[Tensor],
                             max_exp_avg_sqs: List[Tensor],
                             state_steps: List[Tensor],
                             *,
                             amsgrad: bool,
                             beta1: float,
                             beta2: float,
                             lr: float,
                             weight_decay: float,
                             eps: float,
                             maximize: bool):

        for i, param in enumerate(params):
            p_copy = params_copy[i]
            grad = grads[i] if not maximize else -grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step_t = state_steps[i]
            # update step
            step_t += 1
            step = step_t.item()
            
            #################################
            ## Modify
            # Perform stepweight decay
            p_copy.add_(param, alpha= - lr * weight_decay)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1

            p_copy.addcdiv_(exp_avg, denom, value=-step_size)
            
    def adamw(self, params: List[Tensor],
              params_copy: List[Tensor],
              grads: List[Tensor],
              exp_avgs: List[Tensor],
              exp_avg_sqs: List[Tensor],
              max_exp_avg_sqs: List[Tensor],
              state_steps: List[Tensor],
              # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
              # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
              foreach: bool = None,
              *,
              amsgrad: bool,
              beta1: float,
              beta2: float,
              lr: float,
              weight_decay: float,
              eps: float,
              maximize: bool):
        r"""Functional API that performs AdamW algorithm computation.
        See :class:`~torch.optim.AdamW` for details.
        """

        if not all([isinstance(t, torch.Tensor) for t in state_steps]):
            raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

        if foreach is None:
            # Placeholder for more complex foreach logic to be added when value is not set
            foreach = False

        if foreach and torch.jit.is_scripting():
            raise RuntimeError('torch.jit.script not supported with foreach optimizers')

        self._single_tensor_adamw(params,
             params_copy,
             grads,
             exp_avgs,
             exp_avg_sqs,
             max_exp_avg_sqs,
             state_steps,
             amsgrad=amsgrad,
             beta1=beta1,
             beta2=beta2,
             lr=lr,
             weight_decay=weight_decay,
             eps=eps,
             maximize=maximize)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for i in range(len(self.param_groups)):
            group = self.param_groups[i]
            group_copy = self.copy_param_groups[i]
            params_with_grad = []
            params_copy_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            for j in range(len(group['params'])):
                p = group['params'][j]
                p_copy = group_copy['params'][j]
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                params_copy_with_grad.append(p_copy)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = torch.tensor(0.)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                state_steps.append(state['step'])

            self.adamw(params_with_grad,
                  params_copy_with_grad,
                  grads,
                  exp_avgs,
                  exp_avg_sqs,
                  max_exp_avg_sqs,
                  state_steps,
                  amsgrad=amsgrad,
                  beta1=beta1,
                  beta2=beta2,
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  eps=group['eps'],
                  maximize=group['maximize'],
                  foreach=group['foreach'])

        return loss