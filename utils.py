import torch
from torch import nn

def debug(txt);
    print()

def sample_noise(batch_size, channels):
    return torch.randn((batch_size, channels, 1, 1))

def weights_init_truncated_normal(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data = truncated_normal(m.weight.data, 0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data = truncated_normal(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)

def truncated_normal(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor