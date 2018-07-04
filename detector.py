'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

from sklearn import metrics
import torch

import numpy as np
import os
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchsummary import summary
# from data_reader_isic import get_data_loaders
from utils import *
from torch.backends import cudnn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import os
import argparse
import time
from models import *
from utils.utils import progress_bar
from eval import *

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/tmp.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
from statics import *


class Detector(object):
    def __init__(self,model_details):
        self.device_ids=[0,1]
        self.model_details=model_details
        self.log_dir=self.model_details.logs_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        self.use_cuda = torch.cuda.is_available()
        self.best_map = 0  # best test accuracy
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.net = None
        self.get_loss_function = model_details.get_loss_function
        self.optimizer = None
        self.criterion = None

        self.model_name_str = None

    def load_data(self):
        print('==> Preparing data of {}..'.format(self.model_details.dataset))
        self.trainloader, self.testloader = self.model_details.dataset_loader #[trainloader, test_loader]
        train_count = len(self.trainloader) * self.model_details.batch_size
        test_count = len(self.testloader) * self.model_details.batch_size
        print('==> Total examples, train: {}, test:{}'.format(train_count, test_count))

    def load_model(self):
        model_details=self.model_details
        model_name_str = model_details.model_name_str
        print('\n==> using model {}'.format(model_name_str))
        self.model_name_str="{}".format(model_name_str)
        model = model_details.model
        # Model
        try:
            # Load checkpoint.
            assert (os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!')
            checkpoint = torch.load('./checkpoint/{}_ckpt.t7'.format(self.model_name_str ))
            model.load_state_dict(checkpoint['model'].state_dict())
            self.best_map = checkpoint['map']
            self.start_epoch = checkpoint['epoch']
            print('==> Resuming from checkpoint with Accuracy {}..'.format(self.best_map))

        except Exception as e:
            print('==> Resume Failed and Building model..')

        if self.use_cuda:
            model=model.cuda()
            # model = torch.nn.DataParallel(model)
            cudnn.benchmark = True
        self.model=model
        self.optimizer=self.model_details.get_optimizer(self)
        self.criterion=self.model_details.get_loss_function(self)

    def adjust_weight_with_steps(self):
        pass

    def adjust_learning_rate(optimizer, gamma, step):
        """Sets the learning rate to the initial LR decayed by 10 at every
            specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        lr = args.lr * (gamma ** (step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # Training
    def train(self, epoch):
        print('\n==> Training started with Epoch : %d' % epoch)
        model=self.model
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        trainloader=self.trainloader
        optimizer=self.optimizer
        step_index = 0
        loc_loss = 0
        conf_loss = 0
        print("\n==>Total train iteration per epoch:{}".format(len(trainloader)))
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            iteration = epoch * len(trainloader) + batch_idx
            # if iteration in voc['lr_steps']:
            #     step_index += 1
            #     self.adjust_learning_rate(optimizer, args.gamma, step_index)
                # inputs = Variable(inputs.cuda())
                # targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            # else:
            inputs = Variable(inputs.cuda())
            targets = [Variable(ann.cuda(), volatile=True).cuda() for ann in targets]

            # forward
            t0 = time.time()
            inputs=inputs
            out = model(inputs)
            # backprop
            optimizer.zero_grad()
            out=[o.cuda() for o in out]
            loss_l, loss_c = self.criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.data[0]
            conf_loss += loss_c.data[0]

            if iteration % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('epoch:{} '+str(epoch)+', Iteration:{} ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
                # progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #              % (batch_loss, 100. * correct / total, correct, total))

            # if iteration != 0 and iteration % 1000 == 0:
            #     print('Saving state, iter:', iteration)
            #     torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
            #                repr(iteration) + '.pth')


            # step = epoch * len(self.trainloader) + batch_idx
            # # if not self.augment_images==None:
            # #     inputs=torch.from_numpy(self.augment_images(inputs.numpy()))
            # inputs, targets = inputs.to(device), targets.to(device)
            # self.optimizer.zero_grad()
            #
            #
            # outputs = model(inputs)
            # loss = self.criterion(outputs, targets)
            # loss.backward()
            # self.optimizer.step()
            # train_loss += loss.item()
            # _, predicted = outputs.max(1)
            # batch_loss = train_loss / (batch_idx + 1)
            # if batch_idx % 5 == 0:
            #     self.writer.add_scalar('step loss', batch_loss, step)
            # total += targets.size(0)
            # correct += predicted.eq(targets.data).cpu().sum()
            #
            # progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (batch_loss, 100. * correct / total, correct, total))
        self.writer.add_scalar('train loss',train_loss, epoch)
        torch.save(model.state_dict(), args.trained_model)

    def save_model(self, map, epoch):
        print('\n Saving new model with mAP {}'.format(map))
        state = {
            'model': self.model,
            'map': map,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}_ckpt.t7'.format(self.model_name_str ))


    def test(self,epoch):
        print('\n ==> Test started ')
        num_classes = len(labelmap) + 1  # +1 for background
        model = build_ssd('test', 300, num_classes)  # initialize SSD
        model.load_state_dict(torch.load(args.trained_model))
        model.eval()
        print('Finished loading model!')
        # load data
        dataset = VOCDetection(VOC_ROOT, [('2007', set_type)],
                               BaseTransform(300, dataset_mean),
                               VOCAnnotationTransform())

        mAP=test_net(args.save_folder, model, args.cuda, dataset,
                 BaseTransform(model.size, dataset_mean), args.top_k, 300,
                 thresh=args.confidence_threshold)

        # Save checkpoint.
        self.writer.add_scalar('mAP', mAP, epoch)
        # self.writer.add_scalar('test loss', test_loss, epoch)


        print("MAP:{}".format(mAP))
        if mAP > self.best_map:
            self.save_model(mAP, epoch)
            self.best_map = mAP

class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

