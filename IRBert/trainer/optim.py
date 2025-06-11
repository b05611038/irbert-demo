import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


__all__ = ['create_scheduled_optimizer', 'LinearWarmupDecayScheduler']


def create_scheduled_optimizer(args, model = None):
    if model is not None:
        assert isinstance(model, nn.Module)
    else:
        raise RuntimeError('Please set the model in the trainer.')

    total_steps = args.get('train_steps', None)
    assert total_steps is not None
    learning_rate = args.get('learning_rate', 5e-5)
    weight_decay = args.get('weight_decay', 0.01)
    adam_beta1 = args.get('adam_beta1', 0.9)
    adam_beta2 = args.get('adam_beta2', 0.999)    
    adam_epsilon = args.get('adam_epsilon', 1e-8)
    lr_scheduler_type = args.get('lr_scheduler_type', 'linear')
    warmup_ratio = args.get('warmup_ratio', 0.)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr = learning_rate,
                            betas = (adam_beta1, adam_beta2),
                            eps = adam_epsilon,
                            weight_decay = weight_decay)

    warmup_steps = round(float(total_steps) * warmup_ratio)

    lr_scheduler = LinearWarmupDecayScheduler(optimizer,
                                              total_steps = total_steps,
                                              warmup_steps = warmup_steps,
                                              decay_type = lr_scheduler_type)

    return optimizer, lr_scheduler


class LinearWarmupDecayScheduler(_LRScheduler):
    def __init__(self, 
            optimizer,
            total_steps,
            warmup_steps,
            eta_min = 0.,
            last_epoch = -1,
            decay_type = 'linear'):

        assert isinstance(optimizer, optim.Optimizer)
        assert isinstance(total_steps, int)
        assert isinstance(warmup_steps, int)
        assert total_steps >= warmup_steps
        assert isinstance(eta_min, float)
        assert eta_min >= 0.
        assert isinstance(last_epoch, int)
        assert isinstance(decay_type, str)
        assert decay_type in ['linear', 'cosine_annealing']

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_type = decay_type
        self.eta_min = eta_min

        super(LinearWarmupDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch
        if current_step < self.warmup_steps:
            warmup_factor = current_step / max(1, self.warmup_steps)
            lrs = [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            if self.decay_type == 'linear':
                decay_factor = (self.total_steps - current_step) / max(1, self.total_steps - self.warmup_steps)
                lrs = [base_lr * decay_factor for base_lr in self.base_lrs]
            elif self.decay_type == 'cosine_annealing':
                cosine_decay_steps = self.total_steps - self.warmup_steps
                decay_step = current_step - self.warmup_steps
                cosine_factor = 0.5 * (1 + math.cos(math.pi * decay_step / cosine_decay_steps))
                lrs = [self.eta_min + (base_lr - self.eta_min) * cosine_factor for base_lr in self.base_lrs]

        return lrs


