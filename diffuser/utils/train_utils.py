import torch

def get_lr(optimizer: torch.optim.Optimizer):
    """Get the learning rate of current optimizer."""
    return optimizer.param_groups[0]['lr']
