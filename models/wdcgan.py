import torch
from .gan import Generator, Discriminator
from torch import nn

c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32
input_channels = 3
output_channels = 3

class WDCGenerator(Generator):

	def __init__(self, z_dim=100):
		super(Generator, self).__init__()
		"""
		Args:
			z_dim: default: 100
		"""
		self.z_dim = z_dim
		self.latent_dim = self.z_dim

		self.netG = nn.Sequential(
			# latent_dim
			nn.ConvTranspose2d(self.latent_dim, c4, kernel_size=4, stride=1, padding=0, bias=True),
			# nn.BatchNorm2d(num_features=c4),
			nn.ReLU(inplace=True),
			# 512*(4*4)
			nn.ConvTranspose2d(c4, c8, kernel_size=4, stride=2, padding=1, bias=True),
			nn.BatchNorm2d(num_features=c8),
			nn.ReLU(inplace=True),
			# 256*(8*8)
			nn.ConvTranspose2d(c8, c16, kernel_size=4, stride=2, padding=1, bias=True),
			nn.BatchNorm2d(num_features=c16),
			nn.ReLU(inplace=True),
			# 128*(16*16)
			nn.ConvTranspose2d(c16, c32, kernel_size=4, stride=2, padding=1, bias=True),
			nn.BatchNorm2d(num_features=c32),
			nn.ReLU(inplace=True),
			# 64*(32*32)
			nn.ConvTranspose2d(c32, c64, kernel_size=4, stride=2, padding=1, bias=True),
			nn.BatchNorm2d(num_features=c64),
			nn.ReLU(inplace=True),
			# 32*(64*64)
			nn.ConvTranspose2d(c64, output_channels, kernel_size=4, stride=2, padding=1, bias=True),
			nn.Tanh()
			# 3*(128*128)
			)

	def forward(self, z):
		output = self.netG(z)

		return output
		

class WDCDiscriminator(Discriminator):
	def __init__(self, z_dim=100):
		super(Discriminator, self).__init__()
		"""
		Args:
			z_dim: dimension of z latent space (noise), default=100
		"""
		self.z_dim = z_dim
		self.latent_dim = self.z_dim

		self.netD = nn.Sequential(
			# 3*(128*128)
			nn.Conv2d(output_channels, c64, kernel_size=4, stride=2, padding=1, bias=True),
			nn.BatchNorm2d(c64),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			# 32*(64*64)
			nn.Conv2d(c64, c32, kernel_size=4, stride=2, padding=1, bias=True),
			nn.BatchNorm2d(c32),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			# 64*(32*32)
			nn.Conv2d(c32, c16, kernel_size=4, stride=2, padding=1, bias=True),
			nn.BatchNorm2d(c16),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			# 128*(16*16)
			nn.Conv2d(c16, c8, kernel_size=4, stride=2, padding=1, bias=True),
			nn.BatchNorm2d(c8),
			nn.LeakyReLU(negative_slope=0.2, inplace=True),
			# 256*(8*8)
			# nn.Conv2d(c8, c4, kernel_size=4, stride=2, padding=1, bias=True),
			# nn.BatchNorm2d(c4),
			# nn.LeakyReLU(negative_slope=0.2, inplace=True),
			# 128*(4*4)
			nn.Conv2d(c8, 1, kernel_size=8, stride=1, padding=0, bias=True),
			nn.Sigmoid()
			)

	def forward(self, x):
		output = self.netD(x)

		return output