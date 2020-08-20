import torch
from .gan import Generator, Discriminator
from torch import nn

c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32

class ResidualBlock(nn.Module):
	def __init__(self, in_channels):
		self.block = nn.Sequential(
				nn.ReflectionPad2d(1),
				nn.Conv2d(in_channels, in_channels, 3),
				nn.InstanceNorm2d(in_channels),
				nn.RelU(inplace=True),
				nn.ReflectionPad2d(1),
				nn.Conv2d(in_channels, in_channels, 3),
				nn.InstanceNorm2d(in_channels)
			)

	def forward(self, x):
		return self.block(x)

class WGenerator(nn.Module):
	def __init__(self, input_shape, num_residual_blocks):
		super(Generator, self).__init__()

		self.channels = input_shape[0]
		self.num_residual_blocks = num_residual_blocks

		in_channels = self.channels
		out_channels = self.channels
		model = [
			nn.ReflectionPad2d(3),
			nn.Conv2d(in_channels, out_channels, 7),
			nn.InstanceNorm2d(out_channels),
			nn.ReLU(inplace=True)
		]

		in_channels = out_channels

		# downsample
		for _ in range(2):
			out_channels = in_channels*2
			model += [
				nn.Conv(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
				nn.InstanceNorm2d(out_channels),
				nn.ReLU(inplace=True)
			]
			in_channels = out_channels

		# residual
		for _ in self.num_residual_blocks:
			model.append(ResidualBlock(in_channels))

		# unsample
		for _ in range(2):
			out_channels = in_channels // 2
			model += [
				nn.UpSample(scale_factor=2),
				nn.Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
				nn.InstanceNorm2d(out_channels),
				nn.ReLU(inplace=True)
			]

			in_channels = out_channels

		model += [
			nn.ReflectionPad2d(3),
			nn.Conv2d(in_channels, self.channels, kernel_size=7, stride=1, padding=0),
			nn.Tanh()
		]

		self.model = nn.Sequential(*model)

	def forward(self, x):
		return self.model(x)

class WDiscriminator(nn.Module):
	def __init__(self, input_shape):
		super(Discriminator, self).__init__()
		# patchGAN

		channel, height, width = input_shape
		self.output_shape = (1, height//2*4, width//2*4)

		def dis_block(in_channels, out_channels, normalize=True):
			layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
			if normalize:
				layers.append(nn.InstanceNorm2d(out_channels))
			layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
			return layers

		self.model = nn.Sequential(
				*dis_block(channel, c64),
				*dis_block(c64, c32),
				*dis_block(c32, c16),
				*dis_block(c16, c8),
				nn.ZeroPad2d((1, 0, 1, 0)),
				nn.Conv2d(c8, 1, kernel_size=4, stride=1, padding=1)
			)

	def forward(self, x):
		return self.model(x)