import torch
from .gan import Generator, Discriminator
from torch import nn

class WACGenerator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

	def forward(self, x):
		pass

class WACDiscriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

	def forward(self, x):
		pass