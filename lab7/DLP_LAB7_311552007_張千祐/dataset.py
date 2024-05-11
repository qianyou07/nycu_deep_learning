import torch
import os
import numpy as np
import csv
import json
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

default_transform = transforms.Compose([
	transforms.Resize((64, 64)),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class iclevr_dataset(Dataset):
	def __init__(self, args, mode='train', transform=default_transform):
		assert mode == 'train' or mode == 'test'
		self.root = args.data_root
		self.mode = mode
		self.transform = transform
		with open('objects.json') as f:
			self.object = json.load(f)
		self.num_classes = len(self.object)
		if mode == 'train':
			with open('train.json') as f:
				self.labels = json.load(f)
			self.key = list(self.labels)
		else:
			with open('test.json') as f:
				self.labels = json.load(f)
			with open('new_test.json') as f:
				self.labels = self.labels + json.load(f)
			
	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		if self.mode == 'train':
			fname = '{}/{}'.format(self.root, self.key[index])
			img = Image.open(fname).convert('RGB')
			img = self.transform(img)
			label = self.labels[self.key[index]]
			label = list(map(lambda x: self.object[x], label))
			label = nn.functional.one_hot(torch.LongTensor(label), num_classes=self.num_classes).sum(dim=0)
			return img, label
		else:
			label = self.labels[index]
			label = list(map(lambda x: self.object[x], label))
			label = nn.functional.one_hot(torch.LongTensor(label), num_classes=self.num_classes).sum(dim=0)
			return label