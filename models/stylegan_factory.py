from .gan_factory import GANFactory
from .stylegan import StyleGenerator, StyleDiscriminator

class StyleGANFactory(GANFactory):

	def __init__(self, code_dim=512, n_mlp=8):
		self.code_dim = code_dim
		self.n_mlp = n_mlp

	def produce_generator(self):
		return StyleGenerator(self.code_dim, self.n_mlp)

	def produce_discriminator(self):
		return StyleDiscriminator()