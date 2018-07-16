import os
# gpu=1
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
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
from data_reader_pedestrian import get_test_loader_for_upload
import coco_eval
assert torch.__version__.split('.')[1] == '4'
from tensorboardX import SummaryWriter
import time
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

	"""
	todo s:
		################## ToDo ########################
		1. download more images using image_utils and isic-arhive. Also, use more online resources for data.
		2. Use Augmentations fromPytorchSSD using pascal voc data format.
		3. use pair augmentation, random erase
		4. download more images for each classes.
		5. preprocessing and feature extraction
		6. bigger 500 px image size. big image tends to make
		7. use ResNet-152 for better peromance.
		8. adversarial training, use crosssentropy, focal loss
		9. use similar optimizatio adam and learning rate schedule like wider face pedestrian dataset.
		10.BGR to RGB
		11. multi scale testing.
		12. soft nms
		13. save model and load from previous epoch
	"""

	parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
	parser.add_argument('--dataset', default="wider_pedestrain", help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', default="/media/milton/ssd1/research/competitions/data_wider_pedestrian/", help='Path to COCO directory')
	parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

	parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
	parser.add_argument('--epochs', help='Number of epochs', type=int, default=200)

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
	batch_size=10
	num_classes=1
	print("Total Train:{}".format(len(dataset_train)))
	sampler = AspectRatioBasedSampler(dataset_train, batch_size=batch_size, drop_last=False)
	dataloader_train = DataLoader(dataset_train, num_workers=4, collate_fn=collater, batch_sampler=sampler)

	print("Total Validation:{}".format(len(dataset_val)))
	if dataset_val is not None:
		sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=batch_size, drop_last=False)
		dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)
	best_saved_model_name = "checkpoint/resnet{}_{}_best_model.pth".format(parser.depth, parser.dataset)
	best_mAP=0
	start_epoch=0;

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

	optimizer = optim.Adam(retinanet.parameters(), lr=0.001)

	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

	if use_gpu:
		retinanet_sk=copy.deepcopy(retinanet.cpu()) # will hold the raw model, later it will be loaded with new model weight to test in seperate gpus.
		retinanet = retinanet.cuda()
		retinanet = torch.nn.DataParallel(retinanet)

	try:
		print("Loading model and optimizer from checkpoint '{}'".format(best_saved_model_name))
		checkpoint = torch.load(best_saved_model_name)
		retinanet.load_state_dict(checkpoint['model'].state_dict())
		best_mAP=checkpoint['map']
		start_epoch=checkpoint['epoch']
		# optimizer.load_state_dict(checkpoint['optimizer_state'])
		print("Loaded checkpoint '{}' (epoch {})"
			  .format(best_saved_model_name, checkpoint['epoch']))
		start_epoch = checkpoint['epoch']
		print('==> Resuming Sucessfully from checkpoint from epoch {} with mAP {:.7f}..'.format(start_epoch, best_mAP))

	except Exception as e:
		print("\nExcpetion: {}".format(repr(e)))
		print('\n==> Resume Failed...')
	retinanet.training = True



	total_loss = losses.loss

	loss_hist = collections.deque(maxlen=500)

	retinanet.train()
	freeze_bn(retinanet)

	print('Num training images: {}'.format(len(dataset_train)))

	for epoch_num in range(start_epoch,parser.epochs):

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
				if iter_num>250:
					break
			except Exception as e:
				print(e)

		if parser.dataset == 'coco':
			print('\n==>Evaluating dataset')
			coco_eval.evaluate_coco(dataset_val, retinanet, threshold=0.05)

		elif parser.dataset == 'wider_pedestrain':
			retinanet.eval()
			test_data=get_test_loader_for_upload(1)
			new_map=coco_eval.evaluate_wider_pedestrian(epoch_num, dataset_val, retinanet,retinanet_sk ) # to validate
			# print("\nepoch:{}, validation average precision score:{}".format(epoch_num, new_map))
			if new_map==None:
				continue
			writer.add_scalar('validation mAP',new_map,epoch_num)
			scheduler.step(np.mean(epoch_loss))
			epoch_saved_model_name = "checkpoint/resnet{}_{}_epoch_{}.pth".format(parser.depth, parser.dataset, epoch_num)
			save_model(retinanet,optimizer,epoch_saved_model_name,new_map,epoch_num)
			if new_map>best_mAP:
				print("Found new best model with mAP:{:.7f}, over {:.7f}".format(new_map, best_mAP))
				save_model(retinanet,optimizer,best_saved_model_name,new_map,epoch_num)
				best_mAP=new_map

		retinanet.train()

# torch.save(retinanet, '1_'.format(epoch_num,best_saved_model_name))

def save_model(retinanet,optimizer,best_saved_model_name, mAP, epoch):
	print('\n Saving model with mAP {} in {}'.format(mAP, best_saved_model_name))
	state = {
		'model': retinanet,
		'map': mAP, # mAP on validation set.
		'epoch': epoch+1,
		'optimizer':optimizer
	}
	if not os.path.isdir('checkpoint'):
		os.mkdir('checkpoint')
	torch.save(state, best_saved_model_name)

if __name__ == '__main__':
 main()