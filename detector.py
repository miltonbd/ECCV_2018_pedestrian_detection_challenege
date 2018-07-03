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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
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
        self.best_acc = 0  # best test accuracy
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
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
            print('==> Resuming from checkpoint with Accuracy {}..'.format(self.best_acc))

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
        print('\nEpoch: %d' % epoch)
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
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                iteration = epoch * len(trainloader) + batch_idx
                if iteration in voc['lr_steps']:
                    step_index += 1
                    self.adjust_learning_rate(optimizer, args.gamma, step_index)
                    inputs = Variable(inputs.cuda())
                    targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
                else:
                    inputs = Variable(inputs)
                    targets = [Variable(ann, volatile=True) for ann in targets]

                # forward
                t0 = time.time()
                inputs=inputs.cuda()
                out = model(inputs)
                # backprop
                optimizer.zero_grad()
                loss_l, loss_c = self.criterion(out, targets)
                loss = loss_l + loss_c
                loss.backward()
                optimizer.step()
                t1 = time.time()
                loc_loss += loss_l.data[0]
                conf_loss += loss_c.data[0]

                if iteration % 10 == 0:
                    print('timer: %.4f sec.' % (t1 - t0))
                    print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data[0]), end=' ')
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

    def save_model(self, acc, epoch):
        print('\n Saving new model with accuracy {}'.format(acc))
        state = {
            'model': self.model,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}_ckpt.t7'.format(self.model_name_str ))

    def test(self,epoch):
        model=self.model
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        target_all = []
        predicted_all = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                predicted_reshaped = predicted.cpu().numpy().reshape(-1)
                predicted_all = np.concatenate((predicted_all, predicted_reshaped), axis=0)

                targets_reshaped = targets.data.cpu().numpy().reshape(-1)
                target_all = np.concatenate((target_all, targets_reshaped), axis=0)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        self.writer.add_scalar('test accuracy', acc, epoch)
        self.writer.add_scalar('test loss', test_loss, epoch)


        print("Accuracy:{}".format(acc))
        if acc > self.best_acc:
            self.save_model(acc, epoch)
            self.best_acc = acc
        cm = metrics.confusion_matrix(target_all, predicted_all)
        print("\nConfsusion metrics: \n{}".format(cm))


