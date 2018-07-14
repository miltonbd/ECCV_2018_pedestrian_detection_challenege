from statics import *
import os
from scipy.io import loadmat
from torch.utils.data.dataset import Dataset
import numpy as np
import imageio
import torch
from torchvision import transforms
import glob
from PIL import Image
import os
from torchvision.transforms import *
import threading
data_set_name="ISIC 2018"
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

"""
 	    Train 	Val 	Test
Images 	11500 	5000 	3500
Labels 	46513 	19696 	

todo ignore parts set zero
"""

data_set_name="Wider Face Pedestrian dataset."

def read_train_gt():
    annotations=[]
    with open(train_bbx_gt_file,'r') as train_bbx_file:
        content=train_bbx_file.readlines();
        for line in content:
            line_list=line.split(" ")
            file_name=line_list[0]
            row=[]
            for idx in range(1,len(line_list)-1,5):
                class_num=line_list[idx]
                left=line_list[idx+1]
                top=line_list[idx+2]
                w=line_list[idx+3]
                h=line_list[idx+4].strip()
                obj=[class_num, left, top, w, h]
                if len(obj)>0:
                    row+=obj
            if len(row)>0:
                annotations.append([file_name,row[:]])
    return annotations


def read_val_gt():
    annotations = []
    with open(val_bbx_gt_file, 'r') as train_bbx_file:
        content = train_bbx_file.readlines();
        for line in content:
            line_list = line.split(" ")
            file_name = line_list[0]
            row = []
            for idx in range(1, len(line_list) - 1, 5):
                class_num = line_list[idx]
                left = line_list[idx + 1]
                top = line_list[idx + 2]
                w = line_list[idx + 3]
                h = line_list[idx + 4].strip()
                obj = [class_num, left, top, w, h]
                if len(obj) > 0:
                    row += obj
            if len(row) > 0:
                annotations.append([file_name, row[:]])
    return annotations


# annotations=read_train_gt()
# print(len(annotations))
#
# count=0
# for anno in annotations:
#     count+=len(annotations[anno])
# print(count)
# # annos= read_train_gt()
# # for anno in annos:
# #     print(annos[anno])

def test_read_data():
    train_gt=read_train_gt()
    for row in train_gt:
        print(row)


def get_validation_data():
    return

class DatasetReader(Dataset):
    """
    """
    def __init__(self, data,mode='train',):
        print("{} count:{}".format(mode,len(data)))
        self.mode=mode
        self.data=np.asarray(data)
        self.transform_train_image=transforms.Compose([
            RandomCrop([224,224]),
            RandomHorizontalFlip(p=.2),
            # ColorJitter(.6),
            # RandomVerticalFlip(p=.2),
            # RandomGrayscale(p=.2),
            # transforms.RandomRotation(10),
            # transforms.RandomAffine(10),
            # ColorJitter(.6),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]);

        self.transform_test_image = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()]);


    def __getitem__(self, index):
        img_path=self.data[index,0]
        label=int(self.data[index,1])

        if not os.path.exists(img_path):
            print("{} image not found".format(img_path))
            exit(0);
        img = Image.open(img_path)
        if self.mode=="train":
            data = self.transform_train_image(img)
            return data, label

        elif self.mode=="valid":
            data = self.transform_test_image(img)
            return data, label

    def __len__(self):
        return len(self.data)
from statics import *
from utils import *
from data import *
def get_data_loader(args):
    return get_voc_reader(args)

def get_voc_reader(args):
    img_dim=args.size
    rgb_means = (104, 117, 123)
    rgb_std = (1, 1, 1)
    p = (0.6, 0.2)[args.version == 'RFB_mobile']
    train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    cfg = (VOC_300, VOC_512)[args.size == '512']

    testset = VOCDetection(
        VOCroot, [('2007', 'test')], None, AnnotationTransform())

    train_dataset = VOCDetection(VOCroot, train_sets, preproc(
        img_dim, rgb_means, rgb_std, p), AnnotationTransform())

    trainloader = torch.utils.data.DataLoader(train_dataset, args.batch_size,
                                              shuffle=True, num_workers=args.num_workers,
                                              collate_fn=detection_collate)
    num_classes=len(args.classes.split(","))
    detector = Detect(num_classes, 0, cfg)

    return (trainloader, (testset,detector))


def test():
    trainloader, valloader = get_data_loader(100)
    for idx, (inputs, targets) in enumerate(valloader):
        print(inputs.shape)

"""
all the ignore parts of image will be zero.
"""
from file_utils import *

def get_ignore_parts_for_train():
    annotations=[]
    for line in read_text_file(train_bbx_ignore_file):
        line_list = line.split(" ")
        # print(len(line_list))
        file_name = line_list[0]
        for idx in range(1, len(line_list) - 1, 4):
            left = line_list[idx + 1]
            top = line_list[idx + 2]
            w = line_list[idx + 3]
            h = line_list[idx + 4]
            annotations[file_name].append([ left, top, w, h])
    return annotations


if __name__ == '__main__':
    read_train_gt()

