import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from dataloader import RetinopathyLoader
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

def train(model):
	train_acc, test_acc = [], []
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
	loss_fn = nn.CrossEntropyLoss()
	for epoch in range(args.epoch):
		model.train()
		total_num = 0
		correct_num = 0
		for img, label in tqdm(train_dataloader):
			img, label = img.to(device), label.to(device)
			optimizer.zero_grad()

			predicts = model(img)
			loss = loss_fn(predicts, label.long())
			loss.backward()

			optimizer.step()
			predicts = torch.argmax(predicts, dim=1)
			correct_num += sum(predicts==label).cpu().item()
			total_num += len(img)
		train_acc.append(round(100 * correct_num / total_num, 3))
		test_acc.append(evaluate(model, test_dataloader))
		if args.pretrained:
			writer.add_scalars('runs/Accuracy', {'Train(with pretraining)': train_acc[-1], 'Test(with pretraining)': test_acc[-1]}, epoch)
		else:
			writer.add_scalars('runs/Accuracy', {'Train(w/o pretraining)': train_acc[-1], 'Test(w/o pretraining)': test_acc[-1]}, epoch)

		print('Epoch {}, training accuracy: {}%, testing accuracy: {}%'.format(epoch, train_acc[-1], test_acc[-1]))
	return train_acc, test_acc

def evaluate(model, dataloader):
	model.eval()
	total_num = 0
	correct_num = 0

	with torch.no_grad():
		for img, label in tqdm(dataloader):
			img, label = img.to(device), label.to(device)

			predicts = torch.argmax(model(img), dim=1)
			correct_num += sum(predicts==label).cpu().item()
			total_num += len(img)

	return round(100 * correct_num / total_num, 3) # in percentage

def plot(model, dataloader, path):
	model.eval()
	y_true, y_pred = [], []
	with torch.no_grad():
		for img, label in tqdm(dataloader):
			img, label = img.to(device), label.to(device)
			y_true += label.tolist()
			y_pred += torch.argmax(model(img), dim=1).tolist()
	confusion_matrix = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize='all')
	plt.savefig(path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = ArgumentParser()
parser.add_argument("--pretrained", default=False, help="Use pretrained model", action="store_true")
parser.add_argument("--lr", default=1e-3, help="Learning rate", type=float)
parser.add_argument("--momentum", default=0.9, help="Momentum", type=float)
parser.add_argument("--decay", default=5e-4, help="Weight decay", type=float)
parser.add_argument("--batch", default=4, help="Batch size", type=int)
parser.add_argument("--epoch", default=10, help="Training epochs", type=int)
parser.add_argument("--model", default='resnet18', type=str)
parser.add_argument("--plot", default=False, help="Plot confusion matrix", action="store_true")
args = parser.parse_args()

writer = SummaryWriter()

train_dataloader = DataLoader(RetinopathyLoader('data/new_train/', 'train'), batch_size=args.batch, shuffle=True)
test_dataloader = DataLoader(RetinopathyLoader('data/new_test/', 'test'), batch_size=args.batch, shuffle=False)

num_class = 5

if args.model == 'resnet18':
	if args.pretrained:
		model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
	else:
		model = models.resnet18()
elif args.model == 'resnet50':
	if args.pretrained:
		model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
	else:
		model = models.resnet50()
else:
	model = torch.load(args.model)
model.fc = nn.Linear(model.fc.in_features, num_class)
model = model.to(device)

timestr = time.strftime("%Y_%m_%d_%H_%M_%S")
if args.plot:
	path = 'plots/'+timestr+'.png'
	plot(model, test_dataloader, path)
else:
	train(model)
	path = 'models/'+args.model+'_'+timestr
	torch.save(model, path)