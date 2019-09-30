import numpy as np
import torch
import random


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_random_seed(random_seed=0):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


def time_lr_scheduler(optimizer, epoch, lr_decay=0.5, lr_decay_epoch=10):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch and epoch == 0:
        return optimizer

    print("Optimizer learning rate has been decreased.")

    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1. / (1. + lr_decay * epoch))
    return optimizer
