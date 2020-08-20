import torch
import random
from .gan import Generator, Discriminator
from torch import nn
from torch.nn import functional as F
from math import sqrt

class EqualizedLR:
	def __init__(self, name):
		self.name = name

	def compute_weight(self, module):
		weight = getattr(module, self.name + '_origin')
		fan_in = weight.data.size(1) * weight.data[0][0].numel()

		return weight * sqrt(2 / fan_in)

	@staticmethod
	def apply(module, name):
		func = EqualizedLR(name)

		weight = getattr(module, name)
		del module._parameters[name]
		module.register_parameter(name + '_origin', nn.Parameter(weight.data))
		module.register_forward_pre_hook(func)

		return func

	def __call__(self, module, input):
		weight = self.compute_weight(module)
		setattr(module, self.name, weight)

def equalize_lr(module, name='weight'):
	EqualizedLR.apply(module, name)
	return module


class EqualizedLinear(nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()
		linear = nn.Linear(in_dim, out_dim)
		linear.weight.data.normal_()
		linear.bias.data.zero_()

		self.linear = equalize_lr(linear)

	def forward(self, x):
		self.linear(x)

class EqualizedConv2d(nn.Module):
	def __init__(self, *args, **kargs):
		super().__init__()
		conv = nn.Conv2d(*args, **kargs)
		conv.weight.data.normal_()
		conv.bias.data.zero_()
		self.conv = equalize_lr(conv)

	def forward(self, x):
		self.conv(x)

class BlurBackwardFunction(nn.Function):
	@staticmethod
	def forward(self, ctx, output_grad, kernel, kernel_flip):
		ctx.save_for_backward(kernel, kernel_flip)

		input_grad = F.conv(input, kernel_flip, padding=1, groups=input.shape[1])

		return input_grad

	@staticmethod
	def backward(ctx, output_gradgrad):
		kernel, kernel_flip = ctx.saved_tensors

		grad_input = F.conv2d(output_gradgrad, kernel, padding=1, groups=input.shape[1])


class BlurFunction(nn.Function):
	@staticmethod
	def forward(ctx, input, kernel, kernel_flip):
		ctx.save_for_backward(kernel, kernel_flip)
		output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

		return output

	@staticmethod
	def backward(ctx, output_grad):
		kernel, kernel_flip = ctx.saved_tensors

		input_grad = BlurBackwardFunction.apply(output_grad, kernel, kernel_flip)

		return input_grad

blur = BlurFunction.apply

class Blur(nn.Module):
	def __init__(self, channel):
		super().__init__()

		kernel = torch.Tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
		kernel = kernel.view(1, 1, 3, 3)
		kernel = kernel / kernel.sum()
		kernel_flip = torch.flip(kernel, [2, 3])

		self.register_buffer('kernel', kernel.repeat(channel, 1, 1, 1))
		self.register_buffer('kernel_flip', kernel_flip.repeat(channel, 1, 1, 1))

	def forward(self, x):
		return blur(x, self.kernel, self,kernel_flip)


	def forward(self, x):
		pass

class PixelNorm(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x / torch.sqrt(torch.mean(x**2, dim=1, keep_dim=True) + 1e-8)

class ConvBlock(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, padding, kernel_size2=None, padding2=None, downsample=True):
		super().__init__()

		self.kernel_size = kernel_size
		if kernel_size2 is not None:
			self.kernel_size2 = kernel_size2
		else:
			self.kernel_size2 = kernel_size

		self.padding = padding
		if padding2 is not None:
			self.padding2 = padding2
		else:
			self.padding2 = padding

		self.conv1 = nn.Sequential(
				EqualizedConv2d(in_channel, out_channel, self.kernel_size, self.padding),
				nn.LeakyReLU(negative_slope=0.2)
			)

		if downsample:
			self.conv2 = nn.Sequential(
					Blur(out_channel),
					EqualizedConv2d(out_channel, out_channel, self.kernel_size2, self.padding2),
					nn.AvgPool2d(2),
					nn.LeakyReLU(negative_slope=0.2)
				)
		else:
			self.conv2 = nn.Sequential(
					EqualizedConv2d(out_channel, out_channel, self.kernel_size2, self.padding2),
					nn.LeakyReLU(negative_slope=0.2)
				)

	def forward(self, x):
		x = self.conv1(x)
		output = self.conv2(x)

		return output

class AdaptiveInstanceNorm(nn.Module):
	def __init__(self, style_dim, in_channel):
		super().__init__()

		self.style = EqualizedLinear(style_dim, in_channel*2)
		self.norm = nn.InstanceNorm2d(in_channel)

		# initialize
		self.style.linear.bias.data[:in_channel] = 1
		self.style.linear.bias.data[in_channel:] = 0

	def forward(self, x, style):
		style = self.style(style).unsqueeze(2).unsqueeze(3)
		gamma, beta = style.chunk(2, 1)
		output = self.norm(x)

		return gamma*output + beta


class NoiseInjection(nn.Module):
	def __init__(self, channel):
		super().__init__()

		self.weight = nn.Parameter(torch.randn((1, channel, 1, 1)))

	def forward(self, x, noise):
		return x + noise*weight

class ConstantConv(nn.Module):
	def __init__(self, channel, size):
		super().__init__()

		self.weight = nn.Parameter(torch.rand((1, channel, size, size)))

	def forward(self, x):
		batch_size = x.size(0)
		return self.weight.repeat(batch_size, 1, 1, 1)

class StyleConvBlock(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, padding, style_dim=512, upsample=True, initial=False):
		super().__init__()

		if initial:
			self.conv1 = ConstantConv(in_channel)
		else:
			if upsample:
				self.conv1 = nn.Sequential(
						nn.Upsample(scale_factor=2, mode='nearest'),
						EqualizedConv2d(in_channel, out_channel ,kernel_size, padding),
						Blur(out_channel)
					)
			else:
				self.conv1 = EqualizedConv2d(in_channel, out_channel, kernel_size, padding)

		self.noise1 = equalize_lr(NoiseInjection(out_channel))
		self.norm1 = AdaptiveInstanceNorm(style_dim, out_channel)
		self.lrelu1 = nn.LeakyReLU(negative_slope=0.2)

		self.conv2 = EqualizedConv2d(out_channel, out_channel, kernel_size, padding)
		self.noise2 = equalize_lr(NoiseInjection(out_channel))
		self.norm2 = AdaptiveInstanceNorm(style_dim, out_channel)
		self.lrelu2 = nn.LeakyReLU(negative_slope=0.2)

	def forward(self, x, style, noise):
		x = self.conv1(x)
		x = self.noise1(x, noise)
		x = self.norm1(x, style)
		x = self.lrelu1(x)

		x = self.conv2(x)
		x = self.noise2(x, noise)
		x = self.norm2(x, style)
		output = self.lrelu2(x)

		return output

class Generator(Generator):
	def __init__(self):
		super(Generator, self).__init__()

		self.progression = nn.ModuleList(
				StyleConvBlock(512, 512, 3, 1, upsampl=False, initial=True), # 4*4
				StyleConvBlock(512, 512, 3, 1), # 8*8
				StyleConvBlock(512, 256, 3, 1), # 16*16
				StyleConvBlock(256, 128, 3, 1), # 32*32
				StyleConvBlock(128, 64, 3, 1), # 64*64
				StyleConvBlock(64, 32, 3, 1) # 128*128
			)

		self.to_rgb = nn.ModuleList(
				EqualizedConv2d(512, 3, 1),
				EqualizedConv2d(512, 3, 1),
				EqualizedConv2d(512, 3, 1),
				EqualizedConv2d(256, 3, 1),
				EqualizedConv2d(128, 3, 1),
				EqualizedConv2d(64, 3, 1)
			)

	def forward(self, style, noise, step=0, alpha):
		output = noise[0] # let pre_output reference/input for constant conv

		if style.shape[0] < 2: # no style mixing
			split_index = [len(self.progression) - 1] # no split point
		else:
			split_index = random.sample(list(range(self.progression)), style.shape[0]-1) # randomly generate split point

		style_index = 0

		for i, (conv, to_rgb) in enumerate(zip(self.progression, self,to_rgb)):
			if style_index < len(split_index) and i > split_index[style_index]:
				style_index = min(len(style), stlye_index+1)
			
			choosed_style = style[style_index]

			if i>0 and step>0:
				pre_output = output

			output = conv(output, choosed_style, noise[i])

			if i == step:
				output = to_rgb(output)
				skip_output = self.to_rbg[i-1](pre_output)
				skip_output = F.interpolate(skip_output, scale_factor=2, mode='nearest')
				output = alpha*output + (1-alpha)*skip_output

				break
		return output

	# step, phases

class StyleGenerator(Generator):
	def __init__(self, code_dim=512, n_mlp=8):
		super(Generator, self).__init__()

		self.generator = Geneartor()

		layers = [PixelNorm()]
		for i in range(n_mlp):
			layers.append(EqualizedLinear(code_dim))
			layers.append(nn.LeakReLU(0.2))

		self.map = nn.Sequential(*layers)

	def forward(self, latent, noise, step, alpha):
		styles = []
		noises = []

		batch_size = latent.shape(1)

		for i in latent:
			styles.append(self.map(i))

		for i in range(step+1):
			size = 2 ** (i+2)
			noises.append(torch.randn((batch_size, 1, size, size)))

		styles = torch.FloatTensor(styles)
		noises = torch.FloatTensor(noises)

		return self.generator(styles, noises, step, alpha)
		

class StyleDiscriminator(Discriminator):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.progression = nn.ModuleList(
				ConvBlock(32, 64, 3, 1),
				ConvBlock(64, 128, 3, 1),
				ConvBlock(128, 256, 3, 1),
				ConvBlock(256, 512, 3, 1),
				ConvBlock(512, 512, 3, 1),
				ConvBlock(513, 512, 3, 1, 4, 0, downsample=False) # output: 512*(4*4)
			)

		self.from_rgb = nn.ModuleList(
				EqualizedConv2d(3, 32, 1),
				EqualizedConv2d(3, 64, 1),
				EqualizedConv2d(3, 128, 1),
				EqualizedConv2d(3, 256, 1),
				EqualizedConv2d(3, 512, 1),
				EqualizedConv2d(3, 512, 1)
			)

		self.linear = EqualizedLinear(512, 1)

	def forward(self, x, step, alpha):
		for i in range(step, -1, -1):
			index = len(self.progression) - i - 1

			if i==step: # if the first layer
				output = self.from_rgb[index](x)

			if i==0: # if the last layer
				output = torch.cat([torch.sqrt(output.var(0, unbiased=False) + 1e-8).mean().expand(output.size(0), 1, 4, 4), output], dim=1)

			output = self.progression[index](output)

			if i>0: # connect skip net, check if current is last
				if i==step:
					skip_output = F.avg_pool2d(x, 2)
					skip_output = self.from_rgb[index+1](skip_output)

				output = alpha*output + (1-alpha)*skip_output

		output = output.squeeze(2).squeeze(2) # get rid of height and width dimension
		output = self.linear(output)

		return output