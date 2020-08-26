import torch
import torchvision
import importlib
import numpy as np
import math
import random
from torch.utils.data import DataLoader
from torch import autograd
from torch.autograd import Variable
from trainer_base import Trainer
from utils import weights_init_truncated_normal, sample_noise
from data.pokegan_dataset import PokeGanDataset
from models.stylegan_factory import StyleGANFactory
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

class StyleGANTrainer(Trainer):

	def __init__(self, iterations=5000, batch_size=16, num_workers=1, lr=0.001, g_iterations=1, d_iterations=1, init_size=8, max_size=128, lr_schedule=None, 
		batch_size_schedule=None, mixing=True, mixing_prob=0.9, code_dim=512, n_mlp=8, decay=0.999, phase=5_000, loss='gp', optimizer='adam', dataloader=None, 
		style_dim=100, penalty=10, debug_sample_size=16, log_dir='stylegan', model_save_iterations=100, model_dir='trained_model/stylegan/'):
		'''
		args:
			iterations: number of iterations for training
			batch_size: default batch size
			num_workers: number of works to load dataset
			lr: learning rate for generator and discriminator
			g_iterations: how many iterations per interation for generator
			d_iterations: how many iterations per interation for discriminator
			loss:
				'gp': gradient penalty
			optimizer: 
				'ADAM', ...
			init_size: the starting size of generated image of progressive GAN
			max_size: the final size of generated image
			lr_schedule: learning rate schedule, a dictionary with image size paired with learning rate
			batch_size_schedule: batch size schedule, a dictionary image paired with batch size
			mixing: whether use style mixing then training
			mixing_prob: if mixing is used, what probabilitiy to use it
			code_dim: 512
			n_mlp: 8
			dataloader: 
				pytorch dataloader
			style_dim: size of latent vector
			penalty: coefficiency of gradient penalty if it is enabled
			debug_sample_size: how many image sampled per epoch for logging
			log_dir: where to log tensorboard
			model_save_iterations: cycle of iterations to save model
			model_dir: where to save model
		'''
		super(StyleGANTrainer, self).__init__(iterations=iterations, batch_size=batch_size, num_workers=num_workers
			, dataloader=dataloader)
		self.lr = lr
		self.g_iterations = g_iterations
		self.d_iterations = d_iterations
		self.init_size = init_size
		self.max_size = max_size
		self.mixing = mixing
		self.mixing_prob = mixing_prob
		self.code_dim = code_dim
		self.n_mlp = n_mlp
		self.decay = decay
		self.phase = phase
		self.penalty = penalty
		self.debug_sample_size = debug_sample_size
		self.log_dir = log_dir
		self.model_save_iterations = model_save_iterations
		self.model_dir = model_dir

		if lr_schedule is None:
			self.lr_schedule = {128: 0.0002}
		else:
			self.lr_schedule = lr_schedule

		if batch_size_schedule is None:
			self.batch_size_schedule = {4: 256, 8: 128, 16: 128, 32: 64, 64: 64, 128: 64}
		else:
			self.batch_size_schedule = batch_size_schedule

		try:
			assert (loss=='gp')
		except:
			raise ValueError('loss can only be gp(gradient penalty)')
		self.loss = loss

		try:
			assert (optimizer=='adam' or optimizer=='rmsprop')
		except:
			raise ValueError('optimizer can only be adam or rmsprop')

	def Train(self):
		if log_statistics:
			writer = SummaryWriter(log_dir=self.log_dir)

		step = int(math.log2(self.init_size))-2
		alpha = 0
		used_sample = 0
		resolution = 2**(step+2)

		max_step = int(math.log2(self.max_size))-2
		
		batch_size = self.batch_size_schedule.get(resolution, self.batch_size)
		lr = self.lr_schedule.get(resolution, self.lr)
		dataset = PokeGanDataset(images_path='./pokemon_images/pre/', csv_path='./attributes.csv')
		dataloader = self.get_dataloader(dataset, batch_size, resolution/self.max_size)
		
		batch_num = len(dataloader.dataset)//batch_size + 1

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		debug_sample_latents = self.get_pytorch_variable(torch.randn(1, self.debug_sample_size, self.code_dim), device)

		self.generator, self.average_generator, self.discriminator = self.get_models(device)
		self.moving_average(self.average_generator, self.generator, self.decay)

		g_opt, d_opt = self.get_optimizers(lr)

		one = torch.tensor(1, dtype=torch.float).to(device)
		mone = one * -1

		final_progress = False # if all block is added

		for iter in range(self.iterations):

			alpha = min(1, (1/self.phase) * used_sample)

			if final_progress: # if all block is added, then not need for interpolation
				alpha = 1

			if used_sample > self.phase*2:
				step += 1
				used_sample = 0

				if step > max_step:
					step = max_step
					final_progess = True
				else:
					alpha = 0 # reset alpha if new step

				resolution = 2**(step+2)

				batch_size = self.batch_size_schedule.get(resolution, self.batch_size)
				lr = self.lr_schedule.get(resolution, self.lr)

				dataloader = self.get_dataloader(dataset, batch_size, resolution/self.max_size)
				
				self.adjust_lr(g_opt, lr)

			batch_num = len(dataloader.dataset)//batch_size + 1

			d_loss_average = 0
			fake_loss_average = 0
			real_loss_average = 0
			g_loss_average = 0
			gradient_penalty_average = 0

			self.generator.train()
			self.discriminator.train()
			self.log_gpu_stats()
			for idx, dic in enumerate(dataloader):	

				batch_size_t = dic['attribute'].size(0)

				self.generator.train()
				self.discriminator.train()

				real_images = self.get_pytorch_variable(dic['image'], device)
				#print('debug_real_shape: ', real_images.shape)
				for p in self.discriminator.parameters():
					p.requires_grad = True
				self.log_gpu_stats()
				# train discriminator
				for i in range(self.d_iterations):
					self.discriminator.zero_grad()

					# fake loss
					if self.mixing and random.random() < self.mixing_prob:
						latent1, latent2 = torch.randn((4, batch_size_t, self.code_dim)).chunk(2, 0)
					else:
						latent1, latent2 = torch.randn((2, batch_size_t, self.code_dim)).chunk(2, 0)
					latent1 = self.get_pytorch_variable(latent1, device)
					latent2 = self.get_pytorch_variable(latent2, device)
					self.log_gpu_stats()
					fake_images = self.generator(latent1, step, alpha).detach() 
					d_fake_loss = self.discriminator(fake_images, step, alpha).view(-1).mean()
					d_fake_loss.backward(one)
					fake_loss_average += d_fake_loss.item()
					self.log_gpu_stats()
					# real loss
					d_real_loss = self.discriminator(real_images, step, alpha).view(-1).mean()
					d_real_loss.backward(mone)
					real_loss_average += d_real_loss.item()

					if self.loss=='gp':
						gradient_penalty = self.gradient_penalty(real_images, fake_images, batch_size_t, step, alpha, device)
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
					fake_images = self.generator(latent2, step, alpha)
					g_fake_loss = self.discriminator(fake_images, step, alpha).view(-1).mean()
					g_fake_loss.backward(mone)
					self.moving_average(self.average_generator, self.generator, self.decay)
					g_loss_average += -g_fake_loss.item()
					g_opt.step()

				used_sample += batch_size_t

				
			# print epoch summary
			d_loss_average /= batch_num*self.d_iterations
			fake_loss_average /= batch_num*self.d_iterations
			real_loss_average /= batch_num*self.d_iterations
			g_loss_average /= batch_num*self.g_iterations
			if self.loss=='gp':
				gradient_penalty_average /= batch_num*self.d_iterations
			print('epoch: ', iter+1, ', d_loss: ', d_loss_average, ', fake_loss: ', fake_loss_average,
				', real_loss: ', real_loss_average, ', g_loss: ', g_loss_average)

			if log_statistics:
				writer.add_scalar('data/discriminator_loss', d_loss_average, iter+1)
				writer.add_scalar('data/generator_loss', g_loss_average, iter+1)
				writer.add_scalar('data/real_loss', real_loss_average, iter+1)
				writer.add_scalar('data/fake_loss', fake_loss_average, iter+1)
				if self.loss=='gp':
					writer.add_scalar('data/gradient_penalty', gradient_penalty_average, iter+1)

				with torch.no_grad():
					sample_images = self.average_generator(debug_sample_latents, step, alpha).cpu().data
				grid_images = torchvision.utils.make_grid(sample_images, padding=2)
				writer.add_image('images/average_generator', grid_images, iter+1)

				with torch.no_grad():
					sample_images = self.generator(debug_sample_latents, step, alpha).cpu().data
				grid_images = torchvision.utils.make_grid(sample_images, padding=2)
				writer.add_image('images/generator', grid_images, iter+1)

				# grid_images = torchvision.utils.make_grid(real_images[:self.debug_sample_size].cpu().data, padding=2)
				# writer.add_image('images/real', grid_images, iter+1)

			if (iter+1)%self.model_save_iterations == 0:
				self.save_models(iter)

	def get_dataloader(self, dataset, batch_size, scale):
		dataset.scale = scale
		dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=self.num_workers
					, shuffle=True)

		return dataloader

	def Predict(self):
		pass

	def save_models(self, iter):
		torch.save(self.generator.state_dict(), self.model_dir + 'generator_'+str(iter+1)+'.pt')
		torch.save(self.discriminator.state_dict(), self.model_dir + 'discriminator_'+str(iter+1)+'.pt')

	def adjust_lr(self, optimizer, lr):
		for group in optimizer.param_groups:
			mult = group.get('mult', 1)
			group['lr'] = mult * lr

	def moving_average(self, average_generator, generator, decay=0.999):
		avg_params = dict(average_generator.named_parameters())
		params = dict(generator.named_parameters())

		for key in params.keys():
			avg_params[key].data = avg_params[key].data.mul_(decay).add_(1-decay, params[key].data)

	def get_pytorch_variable(self, tensor, device):
		return Variable(tensor).to(device)

	def get_models(self, device):
		factory = StyleGANFactory(code_dim=self.code_dim, n_mlp=self.n_mlp)
		generator = factory.produce_generator().to(device)
		average_generator = factory.produce_generator().to(device)
		average_generator.train(False)
		discriminator = factory.produce_discriminator().to(device)

		return generator, average_generator, discriminator

	def get_optimizers(self, lr):
		if self.optimizer == 'adam':
			g_opt = torch.optim.Adam(self.generator.generator.parameters(), lr=lr, betas=(0.0, 0.999))
			d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.0, 0.999))

			g_opt.add_param_group(
					{
						'params': self.generator.map.parameters(),
						'lr': lr * 0.01,
						'mult': 0.01
					}
				)
		else:
			g_opt = torch.optim.RMSprop(self.generator.generator.parameters(), lr=self.lr)
			d_opt = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.lr)

		return g_opt, d_opt

	def gradient_penalty(self, real_data, fake_data, batch_size, step, alpha, device):
		n_elements = real_data.nelement()
		channels = real_data.size()[1]
		width = real_data.size()[2]
		height = real_data.size()[3]
		alphas = torch.rand(batch_size, 1).expand(batch_size, n_elements//batch_size).contiguous()
		alphas = self.get_pytorch_variable(alphas.view(batch_size, channels, width, height), device)
		interpolates = alphas * real_data.detach() + ((1-alphas) * fake_data.detach())
		interpolates = Variable(interpolates, requires_grad=True)
		d_interpolates = self.discriminator(interpolates, step, alpha)
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
