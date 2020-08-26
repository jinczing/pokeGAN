import torch
import torchvision
import importlib
import numpy as np
from torch.utils.data import DataLoader
from torch import autograd
from torch.autograd import Variable
from trainer_base import Trainer
from utils import weights_init_truncated_normal, sample_noise
from data.pokegan_dataset import PokeGanDataset
from models.wcdcgan_factory import WCDCGANFactory
from bitarray import bitarray
from bitarray.util import int2ba

log_statistics = importlib.util.find_spec('tensorboardX') is not None
if log_statistics:
	from tensorboardX import SummaryWriter

'''
How to create new trainer?
1. create new factory.
2. modified inputs of generator and discriminator.
'''

class WCDCGANTrainer(Trainer):

	def __init__(self, iterations=1, batch_size=16, num_workers=1, g_lr=0.0005, d_lr=0.0001, g_iterations=1, d_iterations=5, mode='gp',
		beta1=0.5, optimizer='adam', split=None, dataloader=None, loss_function=None, z_dim=100, penalty=10, weight_clipping_limit=0.01,
		cond_dim=18, debug_sample_size=16, log_dir='wcdcgan', model_save_iterations=500, model_dir='models/wcdcgan/'):
		'''
		args:
			iterations: number of iterations for training
			batch_size: number of batch size
			num_workers: number of works to load dataset
			g_lr: generator learning rate
			d_lr: discriminator learning rate
			g_iterations: how many iterations per interation for generator
			d_iterations: how many iterations per interation for discriminator
			mode:
				'gp': gradient penalty
				'wc': weights clipping
			optimizer: 
				'ADAM', ...

			split:
				train-test dataset split ratio, 0.0-1.0 for training data

			dataloader: 
				pytorch dataloader

			loss_function: 'bce'(binary cross entropy loss), 'w'(Wasserstein-1 Loss)
			z_dim: size of sampled noise 
			penalty: coefficiency of gradient penalty if it is enabled
			weight_clipping_limit: weight clipping limit of weight clipping if it is employed
			cond_dim: size of conditinal vector
			debug_sample_size: how many image sampled per epoch for logging
			log_dir: where to log tensorboard
			model_save_iterations: cycle of iterations to save model
			model_dir: where to save model
		'''
		super(WCDCGANTrainer, self).__init__(iterations=iterations, batch_size=batch_size, num_workers=num_workers
			, dataloader=dataloader, loss_function=loss_function)
		self.g_lr = g_lr
		self.d_lr = d_lr
		self.g_iterations = g_iterations
		self.d_iterations = d_iterations
		self.beta1 = beta1
		self.z_dim = z_dim
		self.weight_clipping_limit = weight_clipping_limit
		self.cond_dim = cond_dim
		self.penalty = penalty
		self.debug_sample_size = debug_sample_size
		self.log_dir = log_dir
		self.model_save_iterations = model_save_iterations
		self.model_dir = model_dir

		try:
			assert (mode=='gp' or mode=='wc')
		except:
			raise ValueError('mode can only be gp(gradient penalty) or wc(weight clipping)')
		self.mode = mode

		try:
			assert (optimizer=='adam' or optimizer=='rmsprop')
		except:
			raise ValueError('optimizer can only be adam or rmsprop')

	def Train(self):
		self.log_gpu_stats()
		dataset = PokeGanDataset(images_path='./pokemon_images/pre/', csv_path='./attributes.csv')
		dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers
			, shuffle=True)

		if log_statistics:
			writer = SummaryWriter(log_dir=self.log_dir)

		batch_num = len(dataloader.dataset)//self.batch_size + 1

		# device = torch.device('cpu')
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		debug_sample_noises = sample_noise(self.debug_sample_size, self.z_dim).to(device)
		debug_sample_conds = torch.from_numpy(dataset.attributes[torch.randint(0, len(dataset), (self.debug_sample_size,))].astype(float)).float().to(device)

		self.generator, self.discriminator = self.get_models(device)

		g_opt, d_opt = self.get_optimizers()

		one = torch.tensor(1, dtype=torch.float).to(device)
		mone = one * -1

		self.generator.apply(weights_init_truncated_normal)
		self.discriminator.apply(weights_init_truncated_normal)

		for iter in range(self.iterations):
			d_loss_average = 0
			fake_loss_average = 0
			real_loss_average = 0
			g_loss_average = 0
			gradient_penalty_average = 0

			self.generator.train()
			self.discriminator.train()

			for idx, dic in enumerate(dataloader):

				batch_size_t = dic['attribute'].size(0)

				self.generator.train()
				self.discriminator.train()

				real_images = self.get_pytorch_variable(dic['image'], device)
				cond = self.get_pytorch_variable(dic['attribute'].view(batch_size_t, -1, 1, 1), device)

				for p in self.discriminator.parameters():
					p.requires_grad = True

				# train discriminator
				for i in range(self.d_iterations):
					self.discriminator.zero_grad()

					if self.mode =='wc':
						for p in self.discriminator.parameters():
							p.data.clamp_(-self.weight_clipping_limit, self.weight_clipping_limit)

					z_noise = self.get_pytorch_variable(sample_noise(batch_size_t, self.z_dim), device)
					fake_cond = self.get_pytorch_variable(self.get_random_cond(batch_size_t), device)
					fake_images = self.generator(z_noise, fake_cond).detach() 
					d_fake_loss = self.discriminator(fake_images, fake_cond).view(-1).mean()
					d_fake_loss.backward(one)
					fake_loss_average += d_fake_loss.item()

					d_real_loss = self.discriminator(real_images, cond).view(-1).mean()
					d_real_loss.backward(mone)
					real_loss_average += d_real_loss.item()

					if self.mode=='gp':
						gradient_penalty = self.gradient_penalty(real_images, fake_images, batch_size_t, device)
						gradient_penalty.backward(one)
						gradient_penalty_average += gradient_penalty.item()
						d_loss = d_fake_loss.item() - d_real_loss.item() + gradient_penalty.item()
					else:
						d_loss = d_fake_loss.item() - d_real_loss.item()

					d_loss_average += d_loss
					d_opt.step()

				for p in self.discriminator.parameters():
					p.requires_grad = False

				# train generator
				for i in range(self.g_iterations):
					self.generator.zero_grad()
					z_noise = self.get_pytorch_variable(sample_noise(batch_size_t, self.z_dim), device)
					fake_cond = self.get_pytorch_variable(self.get_random_cond(batch_size_t), device)
					fake_images = self.generator(z_noise, fake_cond)
					g_fake_loss = self.discriminator(fake_images, fake_cond).view(-1).mean()
					g_fake_loss.backward(mone)
					g_loss_average += -g_fake_loss.item()
					g_opt.step()

				
			# print epoch summary
			d_loss_average /= batch_num*self.d_iterations
			fake_loss_average /= batch_num*self.d_iterations
			real_loss_average /= batch_num*self.d_iterations
			g_loss_average /= batch_num*self.g_iterations
			if self.mode=='gp':
				gradient_penalty_average /= batch_num*self.d_iterations
			print('epoch: ', iter+1, ', d_loss: ', d_loss_average, ', fake_loss: ', fake_loss_average,
				', real_loss: ', real_loss_average, ', g_loss: ', g_loss_average)

			if log_statistics:
				writer.add_scalar('data/discriminator_loss', d_loss_average, iter+1)
				writer.add_scalar('data/generator_loss', g_loss_average, iter+1)
				writer.add_scalar('data/real_loss', real_loss_average, iter+1)
				writer.add_scalar('data/fake_loss', fake_loss_average, iter+1)
				if self.mode=='gp':
					writer.add_scalar('data/gradient_penalty', gradient_penalty_average, iter+1)

				with torch.no_grad():
					sample_images = self.generator(debug_sample_noises, debug_sample_conds).cpu().data
				grid_images = torchvision.utils.make_grid(sample_images, padding=2)
				writer.add_image('images', grid_images, iter+1)

			if (iter+1)%self.model_save_iterations == 0:
				self.save_models(iter)


	def Predict(self):
		pass

	def save_models(self, iter):
		torch.save(self.generator.state_dict(), self.model_dir + 'generator_'+str(iter+1)+'.pt')
		torch.save(self.discriminator.state_dict(), self.model_dir + 'discriminator_'+str(iter+1)+'.pt')

	def get_pytorch_variable(self, tensor, device):
		return Variable(tensor).to(device)

	def get_random_cond(self, batch_size):
		cond = []
		for i in range(batch_size):
			labels = list(int2ba(np.random.randint(0, 17)).to01().zfill(5))
			labels += list(int2ba(np.random.randint(0, 17)).to01().zfill(5))
			labels = list(map(int, labels))
			labels += list(np.clip((np.random.randn(8)+1)*0.5, 0, 1))
			cond.append(labels)
		return torch.FloatTensor(cond).view(batch_size, -1, 1, 1)

	def get_models(self, device):
		factory = WCDCGANFactory(z_dim=self.z_dim, cond_dim=self.cond_dim)
		generator = factory.produce_generator().to(device)
		discriminator = factory.produce_discriminator().to(device)

		return generator, discriminator

	def get_optimizers(self):
		if self.mode == 'adam':
			g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.g_lr, betas=(self.beta1, 0.999))
			d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_lr, betas=(self.beta1, 0.999))
		else:
			g_opt = torch.optim.RMSprop(self.generator.parameters(), lr=self.g_lr)
			d_opt = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.d_lr)

		return g_opt, d_opt

	def gradient_penalty(self, real_data, fake_data, batch_size, device):
		n_elements = real_data.nelement()
		channels = real_data.size()[1]
		width = real_data.size()[2]
		height = real_data.size()[3]
		alphas = torch.rand(batch_size, 1).expand(batch_size, n_elements//batch_size).contiguous()
		alphas = self.get_pytorch_variable(alphas.view(batch_size, channels, width, height), device)
		interpolates = alphas * real_data.detach() + ((1-alphas) * fake_data.detach())
		interpolates = Variable(interpolates, requires_grad=True)
		fake_cond = self.get_pytorch_variable(self.get_random_cond(batch_size), device)
		d_interpolates = self.discriminator(interpolates, fake_cond)
		ones = Variable(torch.ones(d_interpolates.size()), requires_grad=False).to(device)
		gradients = autograd.grad(
				outputs=d_interpolates,
				inputs=interpolates,
				grad_outputs=ones,
				create_graph=True,
				retain_graph=True,
				only_inputs=True
			)[0]
		gradients = gradients.view(gradients.size(0), -1)
		gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.penalty
		return gradient_penalty

	def log_gpu_stats(self):
		return
		print(torch.cuda.memory_stats(torch.cuda.current_device())['allocated_bytes.all.current']/(1024*1024*1024),
torch.cuda.memory_stats(torch.cuda.current_device())['allocated_bytes.all.allocated']/(1024*1024*1024))
