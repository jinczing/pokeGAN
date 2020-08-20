from abc import ABC, abstractmethod

class Trainer(ABC):

	def __init__(self, iterations=1, batch_size=16, num_workers=1, learning_rate=0.0005, optimizer='adam',
	 split=None, dataloader=None, loss_function='w'):
		'''
		args:
			iterations: number of iterations for training
			batch_size: number of batch size
			num_workers: number of works to load dataset
			learning_rate:
			optimizer: 
				'ADAM', ...

			split:
				train-test dataset split ratio, 0.0-1.0 for training data

			dataloader: 
				pytorch dataloader

			loss_function: 'bce'(binary cross entropy loss), 'w'(Wasserstein-1 Loss)
		'''
		self.iterations = iterations
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.learning_rate = learning_rate
		self.optimizer = optimizer
		self.split = split
		self.dataloader = dataloader
		self.loss_function = loss_function

	@abstractmethod
	def Train(self):
		raise NonImplementedError

	@abstractmethod
	def Predict(self):
		raise NonImplementedError