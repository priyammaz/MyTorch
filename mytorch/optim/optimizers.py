import mytorch
import cupy as cp
import numpy as np

class Optimizer:
    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError
    
class SGD(Optimizer):
    def __init__(self, parameters, lr=0.001):
        self.parameters = parameters
        self.lr = lr

    def step(self):

        for param in self.parameters:
            if param.requires_grad:
                param.data = param.grad - param.grad * self.lr

    def zero_grad(self):
        for param in self.parameters:
            if param.requires_grad:
                param.grad = None

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        
        # Only keep trainable parameters
        self.params = [p for p in parameters if p.requires_grad]

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [mytorch.zeros_like(p).data for p in self.params]
        self.v = [mytorch.zeros_like(p).data for p in self.params]

        self.t = 0
        self.beta1_pow = 1.0
        self.beta2_pow = 1.0

    def step(self):

        self.t += 1
        self.beta1_pow *= self.beta1
        self.beta2_pow *= self.beta2

        lr_t = self.lr * (1 - self.beta2_pow)**0.5 / (1 - self.beta1_pow)

        for i, p in enumerate(self.params):
            
            g = p.grad

            # Apply standard weight decay (L2)
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p.data

            # Update biased first moment estimate
            self.m[i] *= self.beta1
            self.m[i] += (1 - self.beta1) * g

            # Update biased second raw moment estimate
            self.v[i] *= self.beta2
            self.v[i] += (1 - self.beta2) * (g ** 2)

            # Parameter update
            p.data = p.data - lr_t * self.m[i] / (self.v[i]**0.5 + self.eps)
    
    def zero_grad(self):
        for p in self.params:
            p.grad = None

class AdamW(Optimizer):
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        
        # Only keep trainable parameters
        self.params = [p for p in parameters if p.requires_grad]
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [mytorch.zeros_like(p).data for p in self.params]
        self.v = [mytorch.zeros_like(p).data for p in self.params]

        self.t = 0
        self.beta1_pow = 1.0
        self.beta2_pow = 1.0

    def step(self):

        self.t += 1
        self.beta1_pow *= self.beta1
        self.beta2_pow *= self.beta2

        lr_t = self.lr * (1 - self.beta2_pow)**0.5 / (1 - self.beta1_pow)

        for i, p in enumerate(self.params):

            g = p.grad

            # Update biased first moment estimate
            self.m[i] *= self.beta1
            self.m[i] += (1 - self.beta1) * g

            # Update biased second raw moment estimate
            self.v[i] *= self.beta2
            self.v[i] += (1 - self.beta2) * (g ** 2)

            # Parameter update
            denom = self.v[i]**0.5 + self.eps
            step_size = lr_t * self.m[i] / denom
            p.data = p.data - step_size  

            # Apply decoupled weight decay directly to the parameter
            if self.weight_decay != 0.0:
                p.data = p.data - self.lr * self.weight_decay * p.data

    def zero_grad(self):
        for p in self.params:
            p.grad = None

class FusedAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        # Collect flattened tensors
        self.params = [p for p in params if p.requires_grad]
        sizes = [np.prod(p.shape) for p in self.params]

        # Flatten parameter, momentum, and variance buffers
        self.flat_param = mytorch.concatenate([p.reshape(-1) for p in self.params])
        self.m = mytorch.zeros_like(self.flat_param).data
        self.v = mytorch.zeros_like(self.flat_param).data
        self.flat_param = self.flat_param.data

        # Pointers to restore views after updates
        self.views = []
        offset = 0
        for s in sizes:
            self.views.append(slice(offset, offset + s))
            offset += s

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.t = 0
        self.beta1_pow = 1.0
        self.beta2_pow = 1.0

    def step(self):
        self.t += 1
        self.beta1_pow *= self.beta1
        self.beta2_pow *= self.beta2

        lr_t = self.lr * (1 - self.beta2_pow) ** 0.5 / (1 - self.beta1_pow)

        # Flatten gradients into a single contiguous tensor
        flat_grad = cp.concatenate([p.grad.reshape(-1) for p in self.params])

        # Compute all updates in parallel
        self.m *= self.beta1
        self.m += (1 - self.beta1) * flat_grad

        self.v *= self.beta2
        self.v += (1 - self.beta2) * (flat_grad ** 2)

        denom = self.v ** 0.5 + self.eps
        step = self.m / denom
        self.flat_param += step*lr_t

        # Decoupled weight decay
        if self.weight_decay != 0.0:
            self.flat_params -= self.lr * self.weight_decay * self.flat_params

        # Write updated flat buffer back to param tensors (no data copy, just views)
        for p, view in zip(self.params, self.views):
            p.data.copy_(self.flat_param[view].view_as(p))

    def zero_grad(self):
        for p in self.params:
            p.grad = None