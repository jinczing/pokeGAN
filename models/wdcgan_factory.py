from .gan_factory import GANFactory
from .wdcgan import WDCGenerator, WDCDiscriminator

class WDCGANFactory(GANFactory):

	def __init__(self, z_dim=100):
		self.z_dim = z_dim

	def produce_generator(self):
		return WDCGenerator(self.z_dim)

	def produce_discriminator(self):
		return WDCDiscriminator(self.z_dim)