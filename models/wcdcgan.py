import torch
from .gan import Generator, Discriminator
from torch import nn

c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32
input_channels = 3
output_channels = 3

class Concat_Embedded(nn.Module):
	def __init__(self, cond_dim=18, kernel_size=8):
		super(Concat_Embedded, self).__init__()
		self.cond_dim = 18
		self.projected_cond_dim = cond_dim//2

		self.projection = nn.Sequential(
			nn.Linear(in_features=self.cond_dim, out_features=self.projected_cond_dim),
			# nn.BatchNorm2d(num_features=self.projected_cond_dim),
			nn.LeakyReLU(negative_slope=0.2, inplace=True)
			)

		self.kernel_size = kernel_size
	def forward(self, x, cond): # x:256*(self.kernal_size*self.kernel_size)
		projected_cond = self.projection(cond) # (batch, 9)
		replicated_cond = projected_cond.repeat(self.kernel_size, self.kernel_size, 1, 1).permute(2, 3, 0, 1)
		hidden_concat = torch.cat([replicated_cond, x], 1)

		return hidden_concat

class WCDCGenerator(Generator):

	def __init__(self, z_dim=100, cond_dim=18):
		super(Generator, self).__init__()
		"""
		Args:
			z_dim: default: 100
			cond_dim: 18
		"""
		self.z_dim = z_dim
		self.cond_dim = cond_dim
		self.projected_cond_dim = cond_dim//2
		self.latent_dim = self.z_dim + self.projected_cond_dim

		self.projection = nn.Sequential(
			nn.Linear(in_features=self.cond_dim, out_features=self.projected_cond_dim),
			nn.ReLU(inplace=True)
			)

		self.netG = nn.Sequential(
			# latent_dim
			nn.ConvTranspose2d(self.latent_dim, c4, kernel_size=4, stride=1, padding=0),
			# nn.BatchNorm2d(num_features=c4),
			nn.ReLU(inplace=True),
			# 512*(4*4)
			nn.ConvTranspose2d(c4, c8, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(num_features=c8),
			nn.ReLU(inplace=True),
			# 256*(8*8)
			nn.ConvTranspose2d(c8, c16, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(num_features=c16),
			nn.ReLU(inplace=True),
			# 128*(16*16)
			nn.ConvTranspose2d(c16, c32, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(num_features=c32),
			nn.ReLU(inplace=True),
			# 64*(32*32)
			nn.ConvTranspose2d(c32, c64, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(num_features=c64),
			nn.ReLU(inplace=True),
			# 32*(64*64)
			nn.ConvTranspose2d(c64, output_channels, kernel_size=4, stride=2, padding=1),
			nn.Tanh()
			# 3*(128*128)
			)

	def forward(self, z, cond):
		projected_cond = self.projection(cond.view(cond.size(0), -1)).unsqueeze(2).unsqueeze(3)
		latent_vector = torch.cat([projected_cond, z], 1)
		output = self.netG(latent_vector)

		return output
		

class WCDCDiscriminator(Discriminator):
	def __init__(self, z_dim=100, cond_dim=18, mode='gp'):
		super(Discriminator, self).__init__()
		"""
		Args:
			z_dim: dimension of z latent space (noise), default=100
			cond_dim: dimension of conditional input, default=18
			mode: gp(gradient penalty) or wc(weight clipping)
		"""
		self.z_dim = z_dim
		self.cond_dim = cond_dim
		self.projected_cond_dim = cond_dim//2
		self.mode = model

		# construct main net
		channels = [c64, c32, c16, c8]
		netD_list = [nn.Conv2d(output_channels, c64, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2, inplace=True)]
		for i in range(len(channels)-1):
			netD_list += [nn.Conv2(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1)]
			if self.mode == 'gp':
				netD_list += [nn.BatchNorm2d(channels[i+1])]
			netD_list += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]

		self.netD_1 = nn.Sequential(*netD_list)

		# self.netD_1 = nn.Sequential(
		# 	# 3*(128*128)
		# 	nn.Conv2d(output_channels, c64, kernel_size=4, stride=2, padding=1),
		# 	nn.LeakyReLU(negative_slope=0.2, inplace=True),
		# 	# 32*(64*64)
		# 	nn.Conv2d(c64, c32, kernel_size=4, stride=2, padding=1),
		# 	# nn.BatchNorm2d(c32),
		# 	nn.LeakyReLU(negative_slope=0.2, inplace=True),
		# 	# 64*(32*32)
		# 	nn.Conv2d(c32, c16, kernel_size=4, stride=2, padding=1),
		# 	# nn.BatchNorm2d(c16),
		# 	nn.LeakyReLU(negative_slope=0.2, inplace=True),
		# 	# 128*(16*16)
		# 	nn.Conv2d(c16, c8, kernel_size=4, stride=2, padding=1),
		# 	# nn.BatchNorm2d(c8),
		# 	nn.LeakyReLU(negative_slope=0.2, inplace=True),
		# 	# 256*(8*8)
		# 	# nn.Conv2d(c8, c4, kernel_size=4, stride=2, padding=1),
		# 	# # nn.BatchNorm2d(c4),
		# 	# nn.LeakyReLU(negative_slope=0.2, inplace=True)
		# 	# 256*(4*4)
		# 	)

		self.projector = Concat_Embedded(self.cond_dim, 8)

		if self.mode == 'gp':
			self.netD_2 = nn.Sequential(
				nn.Linear((c8 + self.projected_cond_dim)*8*8, 1)
			)
		else:
			self.netD_2 = nn.Sequential(
				# 256*(4*4)
				nn.Conv2d(c8 + self.projected_cond_dim, 1, kernel_size=8, stride=1, padding=0),
				nn.Sigmoid()
				)

		

	def forward(self, x, cond):
		x = self.netD_1(x)
		x = self.projector(x, cond.view(cond.size(0), -1)):
		if self.mode == 'gp':
			x = x.view(x.size(0), -1)
		output = self.netD_2(x)

		return output