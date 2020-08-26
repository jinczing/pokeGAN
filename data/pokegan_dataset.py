import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from skimage import transform, io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class PokeGanDataset(Dataset):

	def __init__(self, images_path, csv_path, scale=1):
		"""
		Args:
			image_path: path that contains pokemon images
			csv_path: path that contains pokemon attributes csv
			scale: used to rescale image to certain size with regard to 128x128
		"""
		self.images_path = images_path
		self.csv_path = csv_path
		self.scale = scale

		with open(self.csv_path, 'r') as f:
			self.attributes = np.asarray(list(csv.reader(f)))

		self.transforms = transforms.Compose([transforms.ToTensor()])

		self.images = []

		for i in range(len(self.attributes)):
			self.images.append(io.imread(os.path.join(self.images_path, str(i+1).zfill(3) + '.png')))

	def __len__(self):
		return self.attributes.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		pk_id = str(idx+1).zfill(3)
		
		if self.scale==1:
			image = self.transforms(self.images[idx]).float()
		else:
			image = self.transforms(transform.rescale(self.images[idx], (self.scale, self.scale, 1), anti_aliasing=False)).float()
		attribute = torch.from_numpy(self.attributes[idx, :].astype(float)).float()
		sample = {'image':image, 'attribute':attribute}

		return sample
