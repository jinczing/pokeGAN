import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
from skimage import transform, io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class PokeGanDataset(Dataset):

	def __init__(self, images_path, csv_path):
		"""
		Args:
			image_path: path that contains pokemon images
			csv_path: path that contains pokemon attributes csv
		"""
		self.images_path = images_path
		self.csv_path = csv_path

		with open(self.csv_path, 'r') as f:
			self.attributes = np.asarray(list(csv.reader(f)))

		self.transforms = transforms.Compose([transforms.ToTensor()])

	def __len__(self):
		return self.attributes.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		pk_id = str(idx+1).zfill(3)
		image = self.transforms(io.imread(os.path.join(self.images_path, pk_id+'.png')))
		attribute = torch.from_numpy(self.attributes[idx, :].astype(float)).float()
		sample = {'image':image, 'attribute':attribute}

		return sample
