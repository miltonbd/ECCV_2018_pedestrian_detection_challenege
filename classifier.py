from __future__ import print_function
from __future__ import division

from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import os
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchsummary import summary
from torch.backends import cudnn
from augment_data import augment_images
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Classifier(object):
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
        self.criterion = None
        self.optimizer = None
        self.model_name_str = None



    def load_model(self):
        model_details=self.model_details
        self.learning_rate=model_details.learning_rate
        model_name = model_details.model_name
        model_name_str = model_details.model_name_str
        print('\n==> using model {}'.format(model_name_str))
        self.model_name_str="{}".format(model_name_str)

        # Model
        try:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert (os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!')
            checkpoint = torch.load('./checkpoint/{}_ckpt.t7'.format(self.model_name_str ))
            model = checkpoint['model']
            self.best_acc = checkpoint['acc']
            self.start_epoch = checkpoint['epoch']
        except Exception as e:
            model = model_details.model
            print('==> Resume Failed and Building model..')

        model=model.cuda()

        # if self.use_cuda:
        #     model = torch.nn.DataParallel(model)
        #     cudnn.benchmark = True
        self.model=model

        # summary(model, (3, 224, 224))
        self.criterion = nn.CrossEntropyLoss()
        if model_details.optimizer=="adam":
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate, eps=model_details.epsilon)

        self.writer.add_scalar("leanring rate", self.learning_rate)
        self.writer.add_scalar("epsilon", model_details.epsilon)

    def train(self, epoch):
        print('\n Training Epoch:{} '.format(epoch))
        model = self.model
        optimizer=self.optimizer
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            # if batch_idx%2==0:
            #     time.sleep(1)

            time.sleep(2)
            step = epoch * len(self.trainloader) + batch_idx
            inputs=torch.from_numpy(augment_images(inputs.numpy()))
            inputs=inputs.type(torch.FloatTensor).cuda(async=True)

            optimizer.zero_grad()
            inputs = Variable(inputs, requires_grad=True).cuda()
            targets = Variable(targets, requires_grad=False).cuda()

            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            batch_loss=train_loss / (batch_idx + 1)
            if batch_idx%2==0:
                self.writer.add_scalar('step loss',batch_loss,step)

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (batch_loss, 100.*correct/total, correct, total))
        self.writer.add_scalar('train loss',train_loss, epoch)


    def save_model(self, acc, epoch):
        print('\n Saving new model with accuracy {}'.format(acc))
        state = {
            'model': self.model.module if self.use_cuda else self.net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}_ckpt.t7'.format(self.model_name_str ))

    def test(self, epoch):
        writer=self.writer
        model=self.model
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        target_all=[]
        predicted_all=[]
        print("\ntesting with previous accuracy {}".format(self.best_acc))
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            if self.use_cuda:
                inputs = inputs.cuda()
                targets=np.asarray(targets).astype(np.int64)
                targets=torch.from_numpy(targets).cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            predicted_batch=predicted.eq(targets.data).cpu()
            predicted_reshaped=predicted_batch.numpy().reshape(-1)
            predicted_all=np.concatenate((predicted_all,predicted_reshaped),axis=0)

            targets_reshaped = targets.data.cpu().numpy().reshape(-1)
            target_all = np.concatenate((target_all, targets_reshaped), axis=0)
            total += targets.size(0)
            correct += predicted_batch.sum()

            progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        self.writer.add_scalar('test loss',test_loss, epoch)
        # Save checkpoint.
        acc = 100.*correct/total
        writer.add_scalar('test accuracy', acc, epoch)
        if acc > self.best_acc:
            pass
        self.save_model(acc, epoch)
        self.best_acc = acc
        print("Accuracy:{}".format(acc))
        cm = metrics.confusion_matrix(target_all, predicted_all)
        print("Confsusion metrics: \n{}".format(cm))
