from abc import ABC, abstractmethod

class GANFactory(ABC):

	def __init__(self):
		pass

	@abstractmethod
	def produce_generator(self):
		pass

	@abstractmethod
	def produce_discriminator(self):
		pass