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
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):

        """
        Identical to AdamW, but we avoid repeated GPU kernel ops. We instead flatten and do it all at once
        and then reassemble the weights. This runs about 2x as fast normal AdamW!
        """
        # Only keep trainable parameters
        self.params = [p for p in parameters if p.requires_grad]
        assert all("cuda" in p.device for p in self.params), "FusedAdamW expects all model parameters to be on GPU!"

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        # Store shapes and flattened sizes for reshaping
        self.shapes = [p.shape for p in self.params]
        self.sizes = [int(cp.prod(cp.array(shape))) for shape in self.shapes]
        self.offsets = np.cumsum([0] + self.sizes)[:-1]
        self.total_size = sum(self.sizes)

        # Concatenate flattened parameters, moments, and initialize
        self.flat_params = cp.concatenate([p.data.reshape(-1) for p in self.params])
        self.flat_m = cp.zeros(self.total_size, dtype=self.flat_params.dtype)
        self.flat_v = cp.zeros(self.total_size, dtype=self.flat_params.dtype)

        self.t = 0
        self.beta1_pow = 1.0
        self.beta2_pow = 1.0

    def step(self):
        self.t += 1
        self.beta1_pow *= self.beta1
        self.beta2_pow *= self.beta2

        lr_t = self.lr * (1 - self.beta2_pow)**0.5 / (1 - self.beta1_pow)

        # Concatenate flattened gradients
        flat_grads = cp.concatenate([p.grad.reshape(-1) for p in self.params])

        # Update biased first moment estimate
        self.flat_m *= self.beta1
        self.flat_m += (1 - self.beta1) * flat_grads

        # Update biased second raw moment estimate
        self.flat_v *= self.beta2
        self.flat_v += (1 - self.beta2) * (flat_grads ** 2)

        # Parameter update
        denom = self.flat_v**0.5 + self.eps
        step_size = lr_t * self.flat_m / denom
        self.flat_params -= step_size

        # Apply decoupled weight decay
        if self.weight_decay != 0.0:
            self.flat_params -= self.lr * self.weight_decay * self.flat_params

        # Update original parameters
        for param, offset, size, shape in zip(self.params, self.offsets, self.sizes, self.shapes):
            param.data = self.flat_params[offset:offset + size].reshape(shape)

    def zero_grad(self):
        for p in self.params:
            p.grad = None
