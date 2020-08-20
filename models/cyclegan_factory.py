from .gan_factory import GANFactory
from .cyclegan import CycleGenerator, CycleDiscriminator

class CycleGANFactory(GANFactory):

	def __init__(self, input_shape, z_dim=100, num_residual_blocks=9):
		self.input_shape = input_shape
		self.z_dim = z_dim
		self.num_residual_blocks = num_residual_blocks

	def produce_generator(self):
		return CycleGenerator(input_shape=(self.z_dim, 1, 1), num_residual_blocks=self.num_residual_blocks)

	def produce_discriminator(self):
		return CycleDiscriminator(self.input_shape)