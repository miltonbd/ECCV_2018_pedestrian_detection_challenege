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
parser.add_argument('--voc_root', default=VOCroot,
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
        train_count = len(self.trainloader) * self.model_details.args.batch_size
        test_count = len(self.testloader) * self.model_details.args.batch_size
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

    def adjust_learning_rate(self,optimizer, gamma, epoch, step_index, iteration, epoch_size):
        """Sets the learning rate
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        args=self.model_details.args
        if epoch < args.warm_epoch:
            lr = 1e-6 + (args.lr - 1e-6) * iteration / (epoch_size * args.warm_epoch)
        else:
            lr = args.lr * (gamma ** (step_index))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    # Training
    def train(self, epoch):
        print('\n==> Training started with Epoch : %d' % epoch)
        net=self.model_details.model
        net.train()
        # loss counters
        loc_loss = 0  # epoch
        conf_loss = 0
        epoch = 0
        args=self.model_details.args
        if args.resume_net:
            epoch = 0 + args.resume_epoch
        # epoch_size = len(train_dataset) // args.batch_size
        epoch_size=100
        max_iter = args.epochs * epoch_size

        stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
        stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
        stepvalues = (stepvalues_VOC, stepvalues_COCO)[args.dataset == 'COCO']
        # print('Training', args.version, 'on', train_dataset.name)
        step_index = 0
        optimizer=self.optimizer
        loc_loss = 0
        conf_loss = 0

        if args.visdom:
            # initialize visdom loss plot
            lot = viz.line(
                X=torch.zeros((1,)).cpu(),
                Y=torch.zeros((1, 3)).cpu(),
                opts=dict(
                    xlabel='Iteration',
                    ylabel='Loss',
                    title='Current SSD Training Loss',
                    legend=['Loc Loss', 'Conf Loss', 'Loss']
                )
            )
            epoch_lot = viz.line(
                X=torch.zeros((1,)).cpu(),
                Y=torch.zeros((1, 3)).cpu(),
                opts=dict(
                    xlabel='Epoch',
                    ylabel='Loss',
                    title='Epoch SSD Training Loss',
                    legend=['Loc Loss', 'Conf Loss', 'Loss']
                )
            )
        if args.resume_epoch > 0:
            start_iter = args.resume_epoch * epoch_size
        else:
            start_iter = 0

        batch_iterator = None # tood set None after epoch
        mean_loss_c = 0 # tood set None after epoch
        mean_loss_l = 0# tood set None after epoch
        batch_iterator = iter(self.trainloader)

        for iteration in range(start_iter, max_iter + 10):

            load_t0 = time.time()
            if iteration in stepvalues:
                step_index = stepvalues.index(iteration) + 1
                if args.visdom:
                    viz.line(
                        X=torch.ones((1, 3)).cpu() * epoch,
                        Y=torch.Tensor([loc_loss, conf_loss,
                                        loc_loss + conf_loss]).unsqueeze(0).cpu() / epoch_size,
                        win=epoch_lot,
                        update='append'
                    )
            lr = self.adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

            # load train data
            images, targets = next(batch_iterator)

            # print(np.sum([torch.sum(anno[:,-1] == 2) for anno in targets]))

            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]
            # forward
            out = net(images)
            # backprop
            optimizer.zero_grad()
            # arm branch loss
            loss_l, loss_c = self.criterion(out, self.model_details.priors, targets)
            # odm branch loss

            mean_loss_c += loss_c.data[0]
            mean_loss_l += loss_l.data[0]

            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            load_t1 = time.time()
            if iteration % 10 == 0:
                print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                      + '|| Totel iter ' +
                      repr(iteration) + ' || L: %.4f C: %.4f||' % (
                          mean_loss_l / 10, mean_loss_c / 10) +
                      'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr))
                # log_file.write(
                #     'Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                #     + '|| Totel iter ' +
                #     repr(iteration) + ' || L: %.4f C: %.4f||' % (
                #         mean_loss_l / 10, mean_loss_c / 10) +
                #     'Batch time: %.4f sec. ||' % (load_t1 - load_t0) + 'LR: %.8f' % (lr) + '\n')

                mean_loss_c = 0
                mean_loss_l = 0
                if args.visdom and args.send_images_to_visdom:
                    random_batch_index = np.random.randint(images.size(0))
                    viz.image(images.data[random_batch_index].cpu().numpy())
        # torch.save(net.state_dict(), os.path.join(save_folder,
        #                                           'Final_' + args.version + '_' + args.dataset + '.pth'))



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

