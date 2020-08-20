import torch
import torchvision
from torch.utils.data import DataLoader
from torch import autograd
from trainer_base import Trainer
from .utils import weights_init_truncated_normal, sample_noise
from .data.pokegan_dataloader import PokeGanDataset
from .models.wcdcgan_factory import WCDCGANFactory


log_statistics = importlib.util.find_spec('tensorboardX') is not None
if log_statistics:
	from tensorboardX import SummaryWriter

class Trainer(ABC):

	def __init__(self, iteratinos, batch_size, num_workers, g_lr, d_lr, beta1, optimizer,
	 split, dataloader, loss_function, z_dim, cond_dim, panelty, debug_sample_size):
		'''
		args:
			iterations: number of iterations for training
			batch_size: number of batch size
			num_workers: number of works to load dataset
			g_lr: generator learning rate
			d_lr: discriminator learning rate
			optimizer: 
				'ADAM', ...

			split:
				train-test dataset split ratio, 0.0-1.0 for training data

			dataloader: 
				pytorch dataloader

			loss_function: 'bce'(binary cross entropy loss), 'w'(Wasserstein-1 Loss)
			z_dim: size of sampled noise 
			cond_dim: size of conditinal vector
			debug_sample_size: how many image sampled per epoch for logging
		'''
		super(WCDCGANTrainer, self).__init__(iterations, batch_size, num_workers, optimizer, split,
			dataloader, loss_function)
		self.g_lr = g_lr
		self.d_lr = d_lr
		self.beta1 = beta1
		self.z_dim = z_dim
		self.cond_dim = cond_dim
		self.panelty = panelty

	def Train(self):
		dataset = PokeGanDataset(images_path='./pokemon_images/pre/', csv_path='./attributes.csv')
		dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers
			, shuffle=True)

		if log_statistics:
			writer = SummaryWriter()

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		debug_sample_noises = torch.randn((self.debug_sample_size, z_dim, 1, 1)).to(device)
		debug_sample_conds = dataset.attributes[torch.randint(0, len(dataset), (self.debug_sample_size,))].to(device)

		factory = WCDCGANFactory()
		generator = factory.produce_generator().to(device)
		discriminator = factory.produce_discriminator().to(device)

		g_opt = torch.optim.Adam(generator.parameters(), lr=self.g_lr, betas=(beta1, 0.999))
		d_opt = torch.optim.Adam(discriminator.parameters(), lr=self.d_lr, betas=(beta1, 0.999))

		minus_one = torch.FloatTensor([-1]).to(device)

		generator.apply(weights_init_truncated_normal)
		discriminator.apply(weights_init_truncated_normal)

		d_loss_average = 0
		fake_loss_average = 0
		real_loss_average = 0
		g_loss_average = 0


		for iter in range(self.iterations):
			for idx, dic in enumerate(DataLoader):
				generator.train()
				discriminator.train()

				real_images = dic['image'].to(device)
				cond = dic['attribute'].view(self.batch_size, -1, 1, 1).to(device)

				# train discriminator
				discriminator.zero_grad()
				z_noise = sample_noise(self.batch_size, self.z_dim).to(device)
				fake_images = generator(z_noise, cond)
				fake_loss = discriminator(fake_images, cond).mean()
				fake_loss_average += fake_loss.item()
				real_loss = discriminator(real_images, cond).mean()
				real_loss_average += real_loss.item()
				d_loss = fake_loss - real_loss + self.gradient_panelty(discriminator, real_images, fake_images, device)
				d_loss.backward()
				d_loss_average += d_loss.item()
				d_opt.step()

				# train generator
				generator.zero_grad()
				z_noise = sample_noise(self.batch_size, self.z_dim)
				fake_images = generator(z_noise, cond)
				fake_loss = discriminator(fake_images, cond).mean()
				fake_loss.backward(minus_one)
				g_loss_average += fake_loss.item()
				g_opt.step()

			# print epoch summary
			d_loss_average /= len(dataloader.dataset)
			fake_loss_average /= len(dataloader.dataset)
			real_loss_average /= len(dataloader.dataset)
			g_loss_average /= len(dataloader.dataset)
			print('epoch: ', iter+1, ', d_loss: ', d_loss_average, ', fake_loss: ', fake_loss_average,
				', real_loss: ', real_loss_average, ', g_loss: ', g_loss_average)

			if log_statistics:
				writer.add_scalar('data/discriminator_loss', d_loss_average, iter+1)
				writer.add_scalar('data/generator_loss', g_loss_average, iter+1)
				writer.add_scalar('data/real_loss', real_loss_average, iter+1)
				writer.add_scalar('data/fake_loss', fake_loss_average, iter+1)

				generator.eval()
				with torch.no_grad():
					sample_images = generator(debug_sample_noises, debug_sample_conds)
				grid_images = torchvision.utils.make_grid(sample_images, padding=2, normalize=Ture)
				writer.add_image('images', grid_images, iter+1)


	def Predict(self):
		pass

	def gradient_panelty(discriminator, real_data, fake_data, device):
		n_elements = real_data.nelement()
		channels = real_data.size()[1]
		width = real_data.size()[2]
		height = real_data.size()[3]
		alphas = torch.randn(self.batch_size, 1).expand(self.batch_size, n_elements/self.batc_size).contiguous()
		alphas = alphas.view(self.batch_size, channels, width, hight).to(device)
		interpolates = alphas * real_data.detach() + (1-alphas) * fake_data.detach()
		interpolates = interpolates.to(device)
		interpolates.require_grad_(True)
		d_interpolates = discriminator(interpolates)
		gradients = autograd.grad(
				outputs=d_interpolates,
				inputs=interpolates,
				grad_outputs=torch.ones(d_interpolates.size()).to(device),
				create_graph=True,
				retain_graph=True,
				only_inputs=True
			)[0]
		gradients = gradients.view(gradients.size(0), -1)
		gradient_panelty = ((gradients.norm(dim=1) - 1) ** 2).mean() * panelty
		return gradient_panelty

