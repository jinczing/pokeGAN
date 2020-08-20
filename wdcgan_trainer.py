import torch
import torchvision
import importlib
from torch.utils.data import DataLoader
from torch import autograd
from torch.autograd import Variable
from trainer_base import Trainer
from utils import weights_init_truncated_normal, sample_noise
from data.pokegan_dataloader import PokeGanDataset
from models.wdcgan_factory import WDCGANFactory


log_statistics = importlib.util.find_spec('tensorboardX') is not None
if log_statistics:
	from tensorboardX import SummaryWriter

class WDCGANTrainer(Trainer):

	def __init__(self, iterations=1, batch_size=16, num_workers=1, g_lr=0.0005, d_lr=0.0001, g_iterations=1, d_iterations=5, mode='gp', 
		beta1=0.5, optimizer='adam', split=None, dataloader=None, loss_function=None, z_dim=100, penalty=10, weight_clipping_limit=0.01, 
		debug_sample_size=16, log_dir='wcdcgan'):
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
			bata1: first beta parameter of Adam optimizer
			optimizer: 
				'ADAM', ...

			split:
				train-test dataset split ratio, 0.0-1.0 for training data

			dataloader: 
				pytorch dataloader

			loss_function: 'bce'(binary cross entropy loss), 'w'(Wasserstein-1 Loss)
			z_dim: size of sampled noise 
			penalty: the coefficiency of gradient penalty if it is enabled
			debug_sample_size: how many image sampled per epoch for logging
			log_dir: logging directory of tensorboard
		'''
		super(WDCGANTrainer, self).__init__(iterations=iterations, batch_size=batch_size, num_workers=num_workers
			, dataloader=dataloader, loss_function=loss_function)
		self.g_lr = g_lr
		self.d_lr = d_lr
		self.g_iterations = g_iterations
		self.d_iterations = d_iterations
		self.beta1 = beta1
		self.z_dim = z_dim
		self.penalty = penalty
		self.weight_clipping_limit = weight_clipping_limit
		self.debug_sample_size = debug_sample_size
		self.log_dir = log_dir

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

		# device = torch.device('cpu')
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		debug_sample_noises = self.get_pytorch_variable(sample_noise(self.debug_sample_size, self.z_dim), device)

		generator, discriminator = self.get_models(device)

		g_opt, d_opt = self.get_optimizers(generator, discriminator)

		one = torch.tensor(1, dtype=torch.float).to(device)
		mone = one * -1

		generator.apply(weights_init_truncated_normal)
		discriminator.apply(weights_init_truncated_normal)

		batch_num = len(dataloader.dataset)//self.batch_size + 1

		for iter in range(self.iterations):
			d_loss_average = 0
			fake_loss_average = 0
			real_loss_average = 0
			g_loss_average = 0
			gradient_penalty_average = 0

			discriminator.train()
			generator.train()

			for idx, dic in enumerate(dataloader):

				batch_size_t = dic['attribute'].size(0)

				generator.train()
				discriminator.train()

				real_images = self.get_pytorch_variable(dic['image'], device)

				# train discriminator
				for p in discriminator.parameters():
					p.requires_grad = True

				for i in range(self.d_iterations):
					discriminator.zero_grad()

					if self.mode =='wc':
						for p in discriminator.parameters():
							p.data.clamp_(-self.weight_clipping_limit, self.weight_clipping_limit)

					z_noise = self.get_pytorch_variable(sample_noise(batch_size_t, self.z_dim), device)
					fake_images = generator(z_noise).detach() 
					d_fake_loss = discriminator(fake_images).view(-1)
					d_fake_loss = d_fake_loss.mean()
					d_fake_loss.backward(one)
					fake_loss_average += d_fake_loss.item()

					d_real_loss = discriminator(real_images).view(-1)
					d_real_loss = d_real_loss.mean()
					d_real_loss.backward(mone)
					real_loss_average += d_real_loss.item()

					if self.mode == 'gp':
						gradient_penalty = self.gradient_penalty(discriminator, real_images, fake_images, batch_size_t, device)
						gradient_penalty.backward(one)
						gradient_penalty_average += gradient_penalty.item()

					if self.mode == 'gp':
						d_loss = d_fake_loss - d_real_loss + gradient_penalty
					else:
						d_loss = d_fake_loss - d_real_loss

					d_loss_average += d_loss.item()
					d_opt.step()

				# train generator
				for p in discriminator.parameters():
					p.requires_gard = False

				for i in range(self.g_iterations):
					generator.zero_grad()
					z_noise = self.get_pytorch_variable(sample_noise(batch_size_t, self.z_dim), device)
					fake_images = generator(z_noise)
					g_fake_loss = discriminator(fake_images).view(-1)
					g_fake_loss = g_fake_loss.mean()
					g_fake_loss.backward(mone)
					g_loss_average += -g_fake_loss.item()
					g_opt.step()

			# print epoch summary
			d_loss_average /= batch_num*self.d_iterations
			fake_loss_average /= batch_num*self.d_iterations
			real_loss_average /= batch_num*self.d_iterations
			g_loss_average /= batch_num*self.g_iterations
			if self.mode == 'gp':
				gradient_penalty_average /= batch_num*self.d_iterations
			print('epoch: ', iter+1, ', d_loss: ', d_loss_average, ', fake_loss: ', fake_loss_average,
				', real_loss: ', real_loss_average, ', g_loss: ', g_loss_average)

			if log_statistics:
				writer.add_scalar('data/discriminator_loss', d_loss_average, iter+1)
				writer.add_scalar('data/generator_loss', g_loss_average, iter+1)
				writer.add_scalar('data/real_loss', real_loss_average, iter+1)
				writer.add_scalar('data/fake_loss', fake_loss_average, iter+1)
				if self.mode == 'gp':
					writer.add_scalar('data/gradient_penalty', gradient_penalty_average, iter+1)

				# generator.eval()
				with torch.no_grad():
					sample_images = generator(debug_sample_noises).cpu().data
				grid_images = torchvision.utils.make_grid(sample_images, padding=2)
				writer.add_image('images', grid_images, iter+1)

			
	def Predict(self):
		pass

	def get_pytorch_variable(self, tensor, device):
		return Variable(tensor).to(device)

	def get_optimizers(self, generator, discriminator):
		if self.mode == 'adam':
			g_opt = torch.optim.Adam(generator.parameters(), lr=self.g_lr, betas=(self.beta1, 0.999))
			d_opt = torch.optim.Adam(discriminator.parameters(), lr=self.d_lr, betas=(self.beta1, 0.999))
		else:
			g_opt = torch.optim.RMSprop(generator.parameters(), lr=self.g_lr)
			d_opt = torch.optim.RMSprop(discriminator.parameters(), lr=self.d_lr)

		return g_opt, d_opt

	def gradient_penalty(self, discriminator, real_data, fake_data, batch_size, device):
		n_elements = real_data.nelement()
		channels = real_data.size()[1]
		width = real_data.size()[2]
		height = real_data.size()[3]
		alphas = torch.rand(batch_size, 1).expand(batch_size, n_elements//batch_size).contiguous()
		alphas = self.get_pytorch_variable(alphas.view(batch_size, channels, width, height), device)
		interpolates = alphas * real_data.detach() + ((1-alphas) * fake_data.detach())
		interpolates = Variable(interpolates, requires_grad=True)
		d_interpolates = discriminator(interpolates)
		ones = self.get_pytorch_variable(torch.ones(d_interpolates.size()), device)
		gradients = autograd.grad(
				outputs=d_interpolates,
				inputs=interpolates,
				grad_outputs=ones,
				create_graph=True,
				retain_graph=True
			)[0]
		gradients = gradients.view(gradients.size(0), -1)
		gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.penalty
		return gradient_penalty

	def get_models(self, device):
		factory = WDCGANFactory(z_dim=self.z_dim)
		generator = factory.produce_generator().to(device)
		discriminator = factory.produce_discriminator().to(device)

		return generator, discriminator

	def log_gpu_stats(self):
		return
		print(torch.cuda.memory_stats(torch.cuda.current_device())['allocated_bytes.all.current']/(1024*1024*1024),
torch.cuda.memory_stats(torch.cuda.current_device())['allocated_bytes.all.allocated']/(1024*1024*1024))
