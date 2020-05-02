from __future__ import print_function

import os
import argparse

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam

import data
import config_frequentist as cfg
from models.NonBayesianModels.AlexNet import AlexNet
from models.NonBayesianModels.LeNet import LeNet
from models.NonBayesianModels.ThreeConvThreeFC import ThreeConvThreeFC
import metrics
import pdb

def getModel(net_type, inputs, outputs):
	if (net_type == 'lenet'):
		return LeNet(outputs, inputs)
	elif (net_type == 'alexnet'):
		return AlexNet(outputs, inputs)
	elif (net_type == '3conv3fc'):
		return ThreeConvThreeFC(outputs, inputs)
	else:
		raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')


def train_model(net, optimizer, criterion, train_loader):
	train_loss = 0.0
	net.train()
	for data, target in train_loader:
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = net(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()*data.size(0)
	return train_loss


def validate_model(net, criterion, valid_loader):
	valid_loss = 0.0
	accs = []
	net.eval()
	for data, target in valid_loader:
		data, target = data.to(device), target.to(device)
		output = net(data)
		loss = criterion(output, target)
		valid_loss += loss.item()*data.size(0)
		accs.append(metrics.acc(output.data, target))

	return valid_loss, np.mean(accs)

def run(dataset, net_type):

	# Hyper Parameter settings
	n_epochs = cfg.n_epochs
	lr = cfg.lr
	# beta_type = cfg.beta_type
	num_workers = cfg.num_workers
	valid_size = cfg.valid_size
	batch_size = cfg.batch_size

	trainset, testset, inputs, outputs = data.getDataset(dataset)
	train_loader, valid_loader, test_loader = data.getDataloader(
		trainset, testset, valid_size, batch_size, num_workers)
	net = getModel(net_type, inputs, outputs).to(device)

	ckpt_dir = 'checkpoints/{}/frequentist'.format(dataset)
	ckpt_name = 'checkpoints/{}/frequentist/model_{}.pt'.format(dataset, net_type)

	if not os.path.exists(ckpt_dir):
		os.makedirs(ckpt_dir, exist_ok=True)

	criterion = nn.CrossEntropyLoss()
	optimizer = Adam(net.parameters(), lr=lr)
	valid_loss_min = np.Inf
	for epoch in range(1, n_epochs+1):
		train_loss = train_model(net, optimizer, criterion, train_loader)
		valid_loss, valid_acc = validate_model(net, criterion, valid_loader)

		train_loss = train_loss/len(train_loader.dataset)
		valid_loss = valid_loss/len(valid_loader.dataset)
			
		print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Accuracy: {:.6f}'.format(
			epoch, train_loss, valid_loss, valid_acc))
		
		# save model if validation loss has decreased
		if valid_loss <= valid_loss_min:
			print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
				valid_loss_min, valid_loss))
			torch.save(net.state_dict(), ckpt_name)
			valid_loss_min = valid_loss

def evaluate(dataset, net_type):

	valid_size = cfg.valid_size
	batch_size = cfg.batch_size
	num_workers = cfg.num_workers

	trainset, testset, inputs, outputs = data.getDataset(dataset)
	train_loader, valid_loader, test_loader = data.getDataloader(
		trainset, testset, valid_size, batch_size, num_workers)
	net = getModel(net_type, inputs, outputs).to(device)

	ckpt_dir = 'checkpoints/{}/frequentist'.format(dataset)
	ckpt_name = 'checkpoints/{}/frequentist/model_{}.pt'.format(dataset, net_type)

	net.load_state_dict(torch.load(ckpt_name))
	criterion = nn.CrossEntropyLoss().to(device)
	valid_loss, valid_acc = validate_model(net, criterion, test_loader)

	valid_loss = valid_loss/len(test_loader.dataset)
	valid_acc = np.mean(valid_acc)

	print('Validation Loss: {:.4f} \tValidation Accuracy: {:.4f}'.format(valid_loss, valid_acc))	

if __name__ == '__main__':
	# CUDA settings
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	parser = argparse.ArgumentParser(description = "PyTorch Frequentist Model Training")
	parser.add_argument('--net_type', default='lenet', type=str, help='model')
	parser.add_argument('--dataset', default='FashionMNIST', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100/FashionMNIST]')
	parser.add_argument('--evaluate', default=False, type=bool, help='')
	args = parser.parse_args()

	if args.evaluate:
		# evaluate the model here
		evaluate(args.dataset, args.net_type)
	else:
		run(args.dataset, args.net_type)
