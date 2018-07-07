'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

from sklearn import metrics
import torch
from utils.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import os
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchsummary import summary
# from data_reader_isic import get_data_loaders
from utils import *
from torch.backends import cudnn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from utils.nms_wrapper import nms

import os
import argparse
import time
from models import *
from utils.utils import progress_bar
from eval import *

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from statics import *


def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005,model_details=None):
    args=model_details.args
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)

    num_classes = (21, 81)[args.dataset == 'COCO']
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file, 'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return

    for i in range(num_images):
        img = testset.pull_image(i)
        x = Variable(transform(img).unsqueeze(0), volatile=True)
        if cuda:
            x = x.cuda()

        _t['im_detect'].tic()
        out = net(x=x, test=True)  # forward pass
        boxes, scores = detector.forward(out, model_details.priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]]).cpu().numpy()
        boxes *= scale

        _t['misc'].tic()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            if args.dataset == 'VOC':
                cpu = False
            else:
                cpu = False

            keep = nms(c_dets, 0.45, force_cpu=cpu)
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                  .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    if args.dataset == 'VOC':
        APs, mAP = testset.evaluate_detections(all_boxes, save_folder)
        return APs, mAP
    else:
        testset.evaluate_detections(all_boxes, save_folder)



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
        args=self.model_details.args
        epoch_size = len(self.trainloader)// args.batch_size
        print("iterations per epoch:{}".format(epoch_size))
        max_iter = args.epochs * epoch_size

        stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
        stepvalues_COCO = (90 * epoch_size, 120 * epoch_size, 140 * epoch_size)
        stepvalues = (stepvalues_VOC, stepvalues_COCO)[args.dataset == 'COCO']
        # print('Training', args.version, 'on', train_dataset.name)
        step_index = 0
        optimizer=self.optimizer

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

        mean_loss_c = 0 # tood set None after epoch
        mean_loss_l = 0# tood set None after epoch
        batch_iterator = iter(self.trainloader)

        for batch_idx, (images, targets) in enumerate(self.trainloader):
            iteration = epoch * len(self.trainloader) + batch_idx

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
        net=self.model_details.model
        net.eval()
        rgb_std = (1, 1, 1)
        print('\n ==> Test started ')
        args=self.model_details.args
        top_k = (300, 200)[args.dataset == 'COCO']
        save_folder = os.path.join(args.save_folder, args.version + '_' + str(args.size), args.date)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        test_save_dir = os.path.join(save_folder, 'ss_predict')
        if not os.path.exists(test_save_dir):
            os.makedirs(test_save_dir)
        if 'vgg' in args.version:
            rgb_means = (104, 117, 123)
        elif 'mobile' in args.version:
            rgb_means = (103.94, 116.78, 123.68)
        testset, detector=self.testloader
        APs, mAP = test_net(test_save_dir, net, detector, args.cuda,testset ,
                            BaseTransform(args.size, rgb_means, rgb_std, (2, 0, 1)),
                            top_k, thresh=0.01,model_details=self.model_details)
        APs = [str(num) for num in APs]
        mAP = str(mAP)
        print("mAP:{}".format(mAP))
        # torch.save(net.state_dict(), os.path.join(save_folder,
        #                                           'Final_' + args.version + '_' + args.dataset + '.pth'))



