import os
gpu=1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
import copy
import argparse
import pdb
import collections
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision
from utils.utils import progress_bar
import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval

assert torch.__version__.split('.')[1] == '4'
from tensorboardX import SummaryWriter
log_dir='logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

print('CUDA available: {}'.format(torch.cuda.is_available()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from tensorboardX import SummaryWriter
log_dir='logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

def freeze_bn(net):
	'''Freeze BatchNorm layers.'''
	for layer in net.modules():
		if isinstance(layer, nn.BatchNorm2d):
			layer.eval()

def main(args=None):

	parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', default="wider_pedestrain", help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', default="/media/milton/ssd1/research/competitions/data_wider_pedestrian/", help='Path to COCO directory')
	parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
	parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

	parser = parser.parse_args(args)

	# Create the data loaders
	if parser.dataset == 'coco':

		if parser.coco_path is None:
			raise ValueError('Must provide --coco_path when training on COCO,')

		dataset_train = CocoDataset(parser.coco_path, set_name='train_wider_pedestrian', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
		dataset_val = CocoDataset(parser.coco_path, set_name='val_wider_pedestrian', transform=transforms.Compose([Normalizer(), Resizer()]))

	elif parser.dataset == 'wider_pedestrain':
		dataset_train = CocoDataset(parser.coco_path, set_name='train_wider_pedestrian',
									transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
		dataset_val = CocoDataset(parser.coco_path, set_name='val_wider_pedestrian',
								  transform=transforms.Compose([Normalizer(), Resizer()]))

		# dataset_test = CocoDataset(parser.coco_path, set_name='test_wider_pedestrian',
		# 						  transform=transforms.Compose([Normalizer()]))

	elif parser.dataset == 'csv':

		if parser.csv_train is None:
			raise ValueError('Must provide --csv_train when training on COCO,')

		if parser.csv_classes is None:
			raise ValueError('Must provide --csv_classes when training on COCO,')


		dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

		if parser.csv_val is None:
			dataset_val = None
			print('No validation annotations provided.')
		else:
			dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')
	batch_size=6
	num_classes=1
	print("Total Train:{}".format(len(dataset_train)))
	sampler = AspectRatioBasedSampler(dataset_train, batch_size=batch_size, drop_last=False)
	dataloader_train = DataLoader(dataset_train, num_workers=4, collate_fn=collater, batch_sampler=sampler)

	print("Total Validation:{}".format(len(dataset_val)))
	if dataset_val is not None:
		sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=batch_size, drop_last=False)
		dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)
	best_saved_model_name = "checkpoint/resnet{}_{}_best_model.pth".format(parser.depth, parser.dataset)
	mAP=0
	try:
		print("Loading model and optimizer from checkpoint '{}'".format(best_saved_model_name))
		checkpoint = torch.load(best_saved_model_name)
		model.load_state_dict(checkpoint['model_state'])
		# optimizer.load_state_dict(checkpoint['optimizer_state'])
		print("Loaded checkpoint '{}' (epoch {})"
			  .format(args.resume, checkpoint['epoch']))
		start_epoch = checkpoint['epoch']
		print('==> Resuming from checkpoint from epoch {} with mAP {}..'.format(start_epoch, mAP))

	except Exception as e:
		print("\nExcpetion: {}".format(repr(e)))
		print('\n==> Resume Failed and Building model..')
		# Create the model
		if parser.depth == 18:
			retinanet = model.resnet18(num_classes=num_classes, pretrained=True)
		elif parser.depth == 34:
			retinanet = model.resnet34(num_classes=num_classes, pretrained=True)
		elif parser.depth == 50:
			retinanet = model.resnet50(num_classes=num_classes, pretrained=True)
		elif parser.depth == 101:
			retinanet = model.resnet101(num_classes=num_classes, pretrained=True)
		elif parser.depth == 152:
			retinanet = model.resnet152(num_classes=num_classes, pretrained=True)
		else:
			raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()
		retinanet = torch.nn.DataParallel(retinanet)

	retinanet.training = True

	optimizer = optim.Adam(retinanet.parameters(), lr=1e-3)

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

	total_loss = losses.loss

	loss_hist = collections.deque(maxlen=500)

	retinanet.train()
	freeze_bn(retinanet)

	print('Num training images: {}'.format(len(dataset_train)))

	for epoch_num in range(parser.epochs):

		retinanet.train()
		freeze_bn(retinanet)

		epoch_loss = []

		for iter_num, data in enumerate(dataloader_train):
			iter_per_epoch = len(dataset_train) / batch_size

			step = epoch_num * iter_per_epoch + iter_num

			if iter_num==0:
				print('Iteration PEr eEpoch: {}'.format(iter_per_epoch))

			try:
				optimizer.zero_grad()

				classification, regression, anchors = retinanet(data['img'].cuda().float())
				
				classification_loss, regression_loss = total_loss(classification, regression, anchors, data['annot'])

				loss = classification_loss + regression_loss
				
				if bool(loss == 0):
					continue

				loss.backward()

				torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

				optimizer.step()

				loss_hist.append(float(loss))

				epoch_loss.append(float(loss))
				writer.add_scalar('Classification loss',classification_loss,step)
				writer.add_scalar('Regression loss',regression_loss,step)
				writer.add_scalar("Running Loss",np.mean(loss_hist),step)

				msg='Epoch:{}, Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, float(classification_loss), float(regression_loss), np.mean(loss_hist))
				progress_bar(iter_num,iter_per_epoch,msg)
				# print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
				# break
				if iter_num>50:
					break
			except Exception as e:
				print(e)
		
		
		if parser.dataset == 'coco':
			print('\n==>Evaluating dataset')
			coco_eval.evaluate_coco(dataset_val, retinanet, threshold=0.05)

		elif parser.dataset == 'wider_pedestrain':
			from data_reader import get_test_loader_for_upload
			test_data=get_test_loader_for_upload(1)
			validation_score=coco_eval.evaluate_wider_pedestrian(epoch_num, dataset_val, retinanet)
			print("epoch:{}, test score:{}".format(epoch_num, validation_score))
			retinanet.train()

		elif parser.dataset == 'csv' and parser.csv_val is not None:
			print('Evaluating dataset')

			total_loss_joint = 0.0
			total_loss_classification = 0.0
			total_loss_regression = 0.0

			for iter_num, data in enumerate(dataloader_val):

				if iter_num % 100 == 0:
					print('{}/{}'.format(iter_num, len(dataset_val)))

				with torch.no_grad():			
					classification, regression, anchors = retinanet(data['img'].cuda().float())
					
					classification_loss, regression_loss = total_loss(classification, regression, anchors, data['annot'])

					total_loss_joint += float(classification_loss + regression_loss)
					total_loss_regression += float(regression_loss)
					total_loss_classification += float(classification_loss)

			total_loss_joint /= float(len(dataset_val))
			total_loss_classification /= float(len(dataset_val))
			total_loss_regression /= float(len(dataset_val))

			print('Validation epoch: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Total loss: {:1.5f}'.format(epoch_num, float(total_loss_classification), float(total_loss_regression), float(total_loss_joint)))
		
		
		scheduler.step(np.mean(epoch_loss))	

		torch.save(retinanet, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

	retinanet.eval()

	torch.save(retinanet, '1_'.format(epoch_num,best_saved_model_name))

if __name__ == '__main__':
 main()