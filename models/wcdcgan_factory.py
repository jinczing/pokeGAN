from .gan_factory import GANFactory
from .wcdcgan import WCDCGenerator, WCDCDiscriminator

class WCDCGANFactory(GANFactory):

	def __init__(self, z_dim=100, cond_dim=18):
		self.z_dim = z_dim
		self.cond_dim = cond_dim

	def produce_generator(self):
		return WCDCGenerator(self.z_dim, self.cond_dim)

	def produce_discriminator(self):
		return WCDCDiscriminator(self.z_dim, self.cond_dim)