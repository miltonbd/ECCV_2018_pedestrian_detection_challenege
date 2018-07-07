import  os
gpu=0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
os.environ['CUDA_LAUNCH_BLOCKING'] = str(gpu)
from object_detector import Detector
from torch import optim
from augment_data import augment_images
from model_loader import *
from loss_loader import *
from data_reader import *

import argparse
import pickle
import time

import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

from data import VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, \
    COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
from layers.functions import Detect, PriorBox
from layers.modules import MultiBoxLoss
from utils.nms_wrapper import nms
from utils.timer import Timer


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

classes=VOC_CLASSES
classes_delimited=','.join(classes)
num_classes=len(classes)

parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')

parser.add_argument('-gpu', default=gpu,
                    type=int, help='gpu index for training.')
parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg RFB_mobile SSD_vgg version.')
parser.add_argument('-s', '--size', default='300',type=int,
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')

parser.add_argument('-classes', default=classes_delimited,type=str,
                    help='class names delimited by ,')
parser.add_argument('-num_classes', default=num_classes, type=int,
                    help='total classes')

parser.add_argument(
    '--basenet', default='weights/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=8,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=2, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate',
                    default=4e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('--resume_net', default=False, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')
parser.add_argument('-epochs', '--epochs', default=300,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('-we', '--warm_epoch', default=1,
                    type=int, help='max epoch for retraining')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')

parser.add_argument('--freeze_layers', default=0.80,
                    type=float, help='PErcentage of weight to be freezed.')

parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='weights/',
                    help='Location to save checkpoint models')
parser.add_argument('--date', default='1213')
parser.add_argument('--save_frequency', default=10)
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
parser.add_argument('--test_frequency', default=10)
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False,
                    help='Sample a random image from each 10th batch, send it to visdom after augmentations step')
args = parser.parse_args()

"""
sudo nvidia-smi -pl 180
sudo nvidia-smi --gpu-reset -i 0
use command line to run the training.

todo download more images using image_utils and isic-arhive. Also, use more online resources for data. 

"""

from layers.modules.multibox_loss import MultiBoxLoss

from statics import *
def get_loss_function(classifier):
    return  MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)

def get_model(args):
    return get_ssd_model(args)

def get_optimizer(model_trainer):
    epsilon=1e-8
    momentum = 0.9
    weight_decay=5e-4
    # model_trainer.writer.add_scalar("leanring rate", learning_rate)
    # model_trainer.writer.add_scalar("epsilon", epsilon)
    # optimizer=optim.SGD(filter(lambda p: p.requires_grad, model_trainer.model.parameters()),
    #                      lr=0.001,momentum=momentum,weight_decay=weight_decay)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_trainer.model.parameters()),lr=0.01)
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_trainer.model.parameters()), lr=0.001, momentum=0.9,
    #                       weight_decay=weight_decay)
    return optimizer

def get_prior():
    cfg = (VOC_300, VOC_512)[args.size == '512']
    priorbox = PriorBox(cfg)
    priors = Variable(priorbox.forward(), volatile=True)
    return priors

class ModelDetails(object):
    def __init__(self,args):
        self.args=args
        self.priors=get_prior()
        self.model,self.model_name_str = get_model(args)
        self.logs_dir  = "logs/{}/{}".format(args.gpu,self.model_name_str)
        self.augment_images = augment_images
        self.dataset_loader=get_data_loader(args)
        self.get_loss_function = get_loss_function
        self.get_optimizer = get_optimizer
        self.dataset=data_set_name
        self.class_names=VOC_CLASSES


def start_training(args):
    model_details=ModelDetails(args)
    detector=Detector(model_details)
    detector.load_data()
    detector.load_model()
    for epoch in range(detector.start_epoch, detector.start_epoch + args.epochs):
        try:
          detector.train(epoch)
          detector.test(epoch)
        except KeyboardInterrupt:
          detector.test(epoch)
          break;
        detector.load_data()

start_training(args)