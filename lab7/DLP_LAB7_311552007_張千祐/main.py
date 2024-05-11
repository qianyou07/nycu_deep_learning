import torch
import torchvision
import argparse
import random
import os
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from model import *
from dataset import *
from evaluator import *

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
	parser.add_argument('--batch_size', default=64, type=int, help='batch size')
	parser.add_argument('--log_dir', default='log', help='base directory to save logs')
	parser.add_argument('--model_dir', default='.', help='base directory to save model')
	parser.add_argument('--data_root', default='iclevr', help='root directory for data')
	parser.add_argument('--epoch', default=50, type=int, help='epoch size')
	parser.add_argument('--seed', default=1, type=int, help='manual seed')
	parser.add_argument('--timesteps', default=1000, type=int, help='training timesteps')
	parser.add_argument('--pretrained', default=False, action='store_true')  
	args = parser.parse_args()
	return args

def train(model, args):
	# Redefining the dataloader to set the batch size higher than the demo of 8
	train_dataset = iclevr_dataset(args, mode='train')
	test_dataset = iclevr_dataset(args, mode='test')
	train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

	# How many runs through the data should we do?
	n_epochs = args.epoch

	# Our loss finction
	loss_fn = nn.MSELoss()

	# The optimizer
	opt = torch.optim.AdamW(model.parameters(), lr=args.lr) 

	scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, steps_per_epoch=len(train_dataloader), epochs=args.epoch)


	# Keeping a record of the losses for later viewing
	losses = []

	highest_acc = 0

	# The training loop
	for epoch in range(n_epochs):
		for x, y in tqdm(train_dataloader):
			
			# Get some data and prepare the corrupted version
			x = x.to(device) # Data on the GPU (mapped to (-1, 1))
			y = y.to(device)
			noise = torch.randn_like(x)
			timesteps = torch.randint(0, args.timesteps-1, (x.shape[0],)).long().to(device)
			noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

			# Get the model prediction
			pred = model(noisy_x, timesteps, y) # Note that we pass in the labels y

			# Calculate the loss
			loss = loss_fn(pred, noise) # How close is the output to the noise

			# Backprop and update the params:
			opt.zero_grad()
			loss.backward()
			opt.step()

			# Store the loss for later
			losses.append(loss.item())

		# Print our the average of the last 100 loss values to get an idea of progress:
		avg_loss = sum(losses[-100:])/100
		writer.add_scalar('Train/Avg loss', avg_loss, epoch)
		with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
			train_record.write(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}\n')
		print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')

		cur_acc = test(model, epoch, test_dataloader, args)
		if cur_acc > highest_acc:
			model.save(args)
			highest_acc = cur_acc

def test(model, epoch, test_dataloader, args):
	for y in tqdm(test_dataloader):
		test_img, newtest_img = inference(model, y, args)
		test_label, newtest_label = y[0:32].to(device), y[32:64].to(device)
	test_fn = './{}/test{}.png'.format(args.log_dir, epoch)
	newtest_fn = './{}/newtest{}.png'.format(args.log_dir, epoch)
	save_image(test_img.detach().cpu(), test_fn, normalize=True)
	save_image(newtest_img.detach().cpu(), newtest_fn, normalize=True)
	test_acc = evaluator.eval(test_img, test_label)
	newtest_acc = evaluator.eval(newtest_img, newtest_label)
	writer.add_scalar('Test/Test acc', test_acc, epoch)
	writer.add_scalar('Test/Newtest acc', newtest_acc, epoch)
	with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
		train_record.write(f'Epoch {epoch}. Test accuracy: {test_acc:05f}, Newtest accuracy: {newtest_acc:05f}\n')
	print(f'Epoch {epoch}. Test accuracy: {test_acc:05f}, Newtest accuracy: {newtest_acc:05f}')
	return test_acc+newtest_acc
	

def inference(model, labels, args):

	# Prepare random x to start from, plus some desired labels y
	x = torch.randn(labels.shape[0], 3, 64, 64).to(device)
	labels = labels.to(device)

	# Sampling loop
	for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

		# Get model pred
		with torch.no_grad():
			residual = model(x, t, labels)  # Again, note that we pass in our labels y

		# Update sample with step
		x = noise_scheduler.step(residual, t, x).prev_sample
	return x[0:32], x[32:64]


if __name__ == '__main__':
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	args = parse_args()

	writer = SummaryWriter(args.log_dir)

	os.makedirs(args.log_dir, exist_ok=True)
	
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	if os.path.exists('./{}/train_record.txt'.format(args.log_dir)):
		os.remove('./{}/train_record.txt'.format(args.log_dir))

	with open('./{}/train_record.txt'.format(args.log_dir), 'a') as train_record:
		train_record.write('args: {}\n'.format(args))

	# Our network 
	model = ClassConditionedUnet(args).to(device)
	evaluator = evaluation_model()
	evaluator.resnet18.to(device)

	# Create a scheduler
	noise_scheduler = DDPMScheduler(num_train_timesteps=args.timesteps)
	noise_scheduler.set_timesteps(num_inference_steps=40)

	train(model, args)
