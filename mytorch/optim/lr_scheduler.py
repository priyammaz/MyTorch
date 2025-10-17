import math 

class LinearLRScheduler:
    def __init__(self, optimizer, max_lr, min_lr=0.0, total_steps=1000, warmup_steps=0):
        """
        Linearly decay LR from max_lr to min_lr after warmup.
        """
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.step_count = 0

    def step(self):
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            lr = self.max_lr * self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.max_lr - (self.max_lr - self.min_lr) * progress

        self.optimizer.lr = lr

    def get_last_lr(self):
        return self.optimizer.lr

class ExponentialLRScheduler:
    def __init__(self, optimizer, max_lr, gamma=0.99, warmup_steps=0):
        """
        Exponentially decay LR after warmup.
        gamma: decay factor per step (0 < gamma < 1)
        """
        self.optimizer = optimizer
        self.base_lr = max_lr
        self.gamma = gamma
        self.warmup_steps = warmup_steps
        self.step_count = 0

    def step(self):
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            lr = self.base_lr * self.step_count / max(1, self.warmup_steps)
        else:
            steps_since_warmup = self.step_count - self.warmup_steps
            lr = self.base_lr * (self.gamma ** steps_since_warmup)

        self.optimizer.lr = lr

    def get_last_lr(self):
        return self.optimizer.lr
    
class CosineLRScheduler:
    def __init__(self, optimizer, max_lr, min_lr=0.0, total_steps=1000, warmup_steps=0):
        """
        Cosine learning rate scheduler.
        
        optimizer: your Adam/AdamW optimizer
        max_lr: initial / max learning rate
        min_lr: minimum learning rate at the end
        total_steps: total number of training steps
        warmup_steps: number of steps to linearly increase LR at start
        """
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.step_count = 0

    def step(self):
        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * self.step_count / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        # Update optimizer LR
        self.optimizer.lr = lr

    def get_last_lr(self):
        return self.optimizer.lr
    
class StepLRScheduler:
    def __init__(self, optimizer, initial_lr, step_size, gamma=0.1, warmup_steps=0):
        """
        Step learning rate scheduler.

        initial_lr: starting LR
        step_size: number of steps before decaying
        gamma: multiplicative factor for decay
        warmup_steps: number of steps to linearly increase LR at start
        """
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_steps = warmup_steps
        self.step_count = 0

    def step(self):
        self.step_count += 1

        # Linear warmup
        if self.step_count <= self.warmup_steps:
            lr = self.initial_lr * self.step_count / max(1, self.warmup_steps)
        else:
            # Step decay
            steps_since_warmup = self.step_count - self.warmup_steps
            factor = self.gamma ** (steps_since_warmup // self.step_size)
            lr = self.initial_lr * factor

        self.optimizer.lr = lr

    def get_last_lr(self):
        return self.optimizer.lr