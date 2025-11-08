"""
A collection of optimizers that can be used by passing in 
either a list (or generator) of parameters

optim = AdamW(model.parameters(), lr=0.0001)

or as a dictionary of parameter groups:

optimizer = AdamW([
    {'params': model.embeddings.parameters(), 'lr': 0.0001, 'weight_decay': 0.0},
    {'params': model.transformer.parameters(), 'lr': 0.001, 'weight_decay': 0.01},
    {'params': model.head.parameters(), 'lr': 0.002, 'weight_decay': 0.05},
])

"""
import mytorch
import cupy as cp
import numpy as np

class Optimizer:
    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError
    
    def _init_optimizer_state(self):
        raise NotImplementedError
    
    def state_dict(self):
        """
        By default an optimizer has no state
        """
        return None
    
    def load_state(self):
        """
        For optimizers without states this is no-op
        """
        pass
    
    def _update_lr(self, lr):
        """
        helper method to replace the lr for
        different param group is available! 
        """
        
        if hasattr(self, "param_groups"):
            for group in self.param_groups:
                group["lr"] = lr
        elif hasattr(self, "lr"):
            self.lr = lr

    def __repr__(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.001, weight_decay=0.0):
        
        ### Create optimizer groups ###
        if isinstance(parameters, (list, tuple)):
            if isinstance(parameters[0], dict):
                self.param_groups = []
                for group in parameters:
                    param_group = {
                        "params": [p for p in group["params"] if p.requires_grad],
                        "lr": group.get("lr", lr),
                        "weight_decay": group.get("weight_decay", weight_decay)
                    }    
                    self.param_groups.append(param_group)
            else:
                self.param_groups = [{
                    'params': [p for p in parameters if p.requires_grad],
                    'lr': lr,
                    'weight_decay': weight_decay,
                }]
        else:
            self.param_groups = [{
                'params': [p for p in parameters if p.requires_grad],
                'lr': lr,
                'weight_decay': weight_decay,
            }]
    
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.requires_grad and p.grad is not None:
                    g = p.grad
                    
                    # Apply weight decay to gradient (L2 regularization)
                    if weight_decay != 0.0:
                        g = g + weight_decay * p.data
                    
                    # Update parameters
                    p.data -= lr * g
    
    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    p.grad = None
    
    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += ("\n" if i == 0 else "")+ f'Parameter Group {i}\n'
            for key in sorted(group.keys()):
                if key == 'params':
                    num_params = sum(np.prod(p.shape) for p in group['params'])
                    format_string += f'  {key}: {len(group[key])} tensors ({num_params:,} parameters)\n'
                else:
                    format_string += f'  {key}: {group[key]}\n'
        format_string += ')'
        return format_string

class Adam(Optimizer):

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        
        ### Create optimizer groups ###
        if isinstance(parameters, (list, tuple)):
            if isinstance(parameters[0], dict):

                self.param_groups = []

                for group in parameters:

                    param_group = {
                        "params": [p for p in group["params"] if p.requires_grad],
                        "lr": group.get("lr", lr),
                        "beta1": group.get("beta1", beta1),
                        "beta2": group.get("beta2", beta2),
                        "eps": group.get("eps", eps),
                        "weight_decay": group.get("weight_decay", weight_decay)
                    }    

                    self.param_groups.append(param_group)

        else:

            self.param_groups = [{
                'params': [p for p in parameters if p.requires_grad],
                'lr': lr,
                'beta1': beta1,
                'beta2': beta2,
                'eps': eps,
                'weight_decay': weight_decay,
            }]

        ### Init everything we need for this optimizer ###
        self._init_optimizer_state()

    def _init_optimizer_state(self, device="cpu"):
        
        """
        All optimizer states need to be on the correct device so we
        can create them here with that context!
        """
        ### Create the optimizer states ###
        for group in self.param_groups:
            group['m'] = [mytorch.zeros_like(p, device=device).data for p in group['params']]
            group['v'] = [mytorch.zeros_like(p, device=device).data for p in group['params']]
            group['t'] = 0
            group['beta1_pow'] = 1.0
            group['beta2_pow'] = 1.0

    def step(self):
        for group in self.param_groups:

            group['t'] += 1
            group['beta1_pow'] *= group['beta1']
            group['beta2_pow'] *= group['beta2']
            
            lr_t = group['lr'] * (1 - group['beta2_pow'])**0.5 / (1 - group['beta1_pow'])
            
            for i, p in enumerate(group['params']):

                if p.requires_grad:
                    g = p.grad

                    if group["weight_decay"] != 0.0:
                        g = g + group["weight_decay"] * p.data
                    
                    # Update biased first moment estimate
                    group['m'][i] *= group['beta1']
                    group['m'][i] += (1 - group['beta1']) * g
                    
                    # Update biased second raw moment estimate
                    group['v'][i] *= group['beta2']
                    group['v'][i] += (1 - group['beta2']) * (g ** 2)
                    
                    # Parameter update
                    denom = group['v'][i]**0.5 + group['eps']
                    step_size = lr_t * group['m'][i] / denom
                    p.data = p.data - step_size
                    
                    # Apply decoupled weight decay directly to the parameter
                    if group['weight_decay'] != 0.0:
                        p.data = p.data - group['lr'] * group['weight_decay'] * p.data
    
    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                p.grad = None

    def state_dict(self):

        state_dict = {
            "state": [],
            "param_groups": []
        }

        ### Save state for each param group 
        for group in self.param_groups:

            group_state = {
                'm': [m.copy() for m in group['m']],
                'v': [v.copy() for v in group['v']],
                't': group['t'],
                'beta1_pow': group['beta1_pow'],
                'beta2_pow': group['beta2_pow'],
            }

            state_dict['state'].append(group_state)

            group_dict = {
                'lr': group['lr'],
                'beta1': group['beta1'],
                'beta2': group['beta2'],    
                'eps': group['eps'],
                'weight_decay': group['weight_decay'],
                'num_params': len(group['params']),
            }

            state_dict['param_groups'].append(group_dict)

        return state_dict
    

    def load_state_dict(self, state_dict):

        if len(state_dict['state']) != len(self.param_groups):
            raise ValueError(
                f"Loaded state dict has {len(state_dict['state'])} parameter groups, "
                f"but optimizer has {len(self.param_groups)} parameter groups"
            )

        for i, (group, group_state) in enumerate(zip(self.param_groups, state_dict["state"])):
            
            ### Check that number of parameters matches
            if len(group['params']) != len(group_state['m']):
                raise ValueError(
                    f"Parameter group {i}: loaded state has {len(group_state['m'])} parameters, "
                    f"but optimizer has {len(group['params'])} parameters"
                )
            
            ### Copy Everything Over ###
            group['m'] = [m.copy() for m in group_state['m']]
            group['v'] = [v.copy() for v in group_state['v']]
            group['t'] = group_state['t']
            group['beta1_pow'] = group_state['beta1_pow']
            group['beta2_pow'] = group_state['beta2_pow']
            
            loaded_group = state_dict['param_groups'][i]
            group['lr'] = loaded_group['lr']
            group['beta1'] = loaded_group['beta1']
            group['beta2'] = loaded_group['beta2']
            group['eps'] = loaded_group['eps']
            group['weight_decay'] = loaded_group['weight_decay']


    def __repr__(self):
        format_string = self.__class__.__name__ + f' ('
        for i, group in enumerate(self.param_groups):
            format_string += ("\n" if i == 0 else "")+f'Parameter Group {i}\n'
            for key in sorted(group.keys()):
                if key == 'params':
                    num_params = sum(np.prod(p.shape) for p in group['params'])
                    format_string += f'  {key}: {len(group[key])} tensors ({num_params:,} parameters)\n'
                elif key not in ['m', 'v', 't', 'beta1_pow', 'beta2_pow']:
                    format_string += f'  {key}: {group[key]}\n'
        format_string += ')'
        return format_string

class AdamW(Optimizer):
    """
    Identical to Adam but we now use a decoupled weight decay
    """

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        
        ### Create optimizer groups ###
        if isinstance(parameters, (list, tuple)):
            if isinstance(parameters[0], dict):

                self.param_groups = []

                for group in parameters:

                    param_group = {
                        "params": [p for p in group["params"] if p.requires_grad],
                        "lr": group.get("lr", lr),
                        "beta1": group.get("beta1", beta1),
                        "beta2": group.get("beta2", beta2),
                        "eps": group.get("eps", eps),
                        "weight_decay": group.get("weight_decay", weight_decay)
                    }    

                    self.param_groups.append(param_group)

        else:

            self.param_groups = [{
                'params': [p for p in parameters if p.requires_grad],
                'lr': lr,
                'beta1': beta1,
                'beta2': beta2,
                'eps': eps,
                'weight_decay': weight_decay,
            }]

        ### Init everything we need for this optimizer ###
        self._init_optimizer_state()

    def _init_optimizer_state(self, device="cpu"):
        
        """
        All optimizer states need to be on the correct device so we
        can create them here with that context!
        """
        ### Create the optimizer states ###
        for group in self.param_groups:
            group['m'] = [mytorch.zeros_like(p, device=device).data for p in group['params']]
            group['v'] = [mytorch.zeros_like(p, device=device).data for p in group['params']]
            group['t'] = 0
            group['beta1_pow'] = 1.0
            group['beta2_pow'] = 1.0

    def step(self):
        for group in self.param_groups:

            group['t'] += 1
            group['beta1_pow'] *= group['beta1']
            group['beta2_pow'] *= group['beta2']
            
            lr_t = group['lr'] * (1 - group['beta2_pow'])**0.5 / (1 - group['beta1_pow'])
            
            for i, p in enumerate(group['params']):

                if p.requires_grad:
                    g = p.grad
                    
                    # Update biased first moment estimate
                    group['m'][i] *= group['beta1']
                    group['m'][i] += (1 - group['beta1']) * g
                    
                    # Update biased second raw moment estimate
                    group['v'][i] *= group['beta2']
                    group['v'][i] += (1 - group['beta2']) * (g ** 2)
                    
                    # Parameter update
                    denom = group['v'][i]**0.5 + group['eps']
                    step_size = lr_t * group['m'][i] / denom
                    p.data -= step_size
                    
                    # Apply decoupled weight decay directly to the parameter
                    if group['weight_decay'] != 0.0:
                        p.data = p.data - group['lr'] * group['weight_decay'] * p.data
    
    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                p.grad = None

    def state_dict(self):

        state_dict = {
            "state": [],
            "param_groups": []
        }

        ### Save state for each param group 
        for group in self.param_groups:

            group_state = {
                'm': [m.copy() for m in group['m']],
                'v': [v.copy() for v in group['v']],
                't': group['t'],
                'beta1_pow': group['beta1_pow'],
                'beta2_pow': group['beta2_pow'],
            }

            state_dict['state'].append(group_state)

            group_dict = {
                'lr': group['lr'],
                'beta1': group['beta1'],
                'beta2': group['beta2'],    
                'eps': group['eps'],
                'weight_decay': group['weight_decay'],
                'num_params': len(group['params']),
            }

            state_dict['param_groups'].append(group_dict)

        return state_dict
    

    def load_state_dict(self, state_dict):

        if len(state_dict['state']) != len(self.param_groups):
            raise ValueError(
                f"Loaded state dict has {len(state_dict['state'])} parameter groups, "
                f"but optimizer has {len(self.param_groups)} parameter groups"
            )

        for i, (group, group_state) in enumerate(zip(self.param_groups, state_dict["state"])):
            
            ### Check that number of parameters matches
            if len(group['params']) != len(group_state['m']):
                raise ValueError(
                    f"Parameter group {i}: loaded state has {len(group_state['m'])} parameters, "
                    f"but optimizer has {len(group['params'])} parameters"
                )
            
            ### Copy Everything Over ###
            group['m'] = [m.copy() for m in group_state['m']]
            group['v'] = [v.copy() for v in group_state['v']]
            group['t'] = group_state['t']
            group['beta1_pow'] = group_state['beta1_pow']
            group['beta2_pow'] = group_state['beta2_pow']
            
            loaded_group = state_dict['param_groups'][i]
            group['lr'] = loaded_group['lr']
            group['beta1'] = loaded_group['beta1']
            group['beta2'] = loaded_group['beta2']
            group['eps'] = loaded_group['eps']
            group['weight_decay'] = loaded_group['weight_decay']


    def __repr__(self):
        format_string = self.__class__.__name__ + f' ('
        for i, group in enumerate(self.param_groups):
            format_string += ("\n" if i == 0 else "")+f'Parameter Group {i}\n'
            for key in sorted(group.keys()):
                if key == 'params':
                    num_params = sum(np.prod(p.shape) for p in group['params'])
                    format_string += f'  {key}: {len(group[key])} tensors ({num_params:,} parameters)\n'
                elif key not in ['m', 'v', 't', 'beta1_pow', 'beta2_pow']:
                    format_string += f'  {key}: {group[key]}\n'
        format_string += ')'
        return format_string

class RMSProp():
    pass

class Adagrad():
    pass

class Adadelta():
    pass