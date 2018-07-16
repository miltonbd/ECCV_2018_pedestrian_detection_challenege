import os
from PIL import Image
from imgaug import augmenters as iaa
import numpy as np
import imageio
from utils.utils import *
import threading
from data.voc0712 import *

import argparse
import pickle
import time
from utils.utils import *
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

"""
5Dtensor
todo save some random tensor.
generate augmentation data in pascal voc format and later append them in mscoco files in training.
"""
data_dir="/media/milton/ssd1/research/competitions/data_wider_pedestrian"
create_dir_if_not_exists(data_dir)
voc_format_data_dir=os.path.join(data_dir,'VOC_Wider_pedestrian')
seq = iaa.Sequential([
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
                  iaa.GaussianBlur(sigma=(0, 0.5))
                  ),

    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        # translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        # rotate=(-25, 25),
        shear=(-4, 4)
    ),
    iaa.Grayscale(alpha=(0.0, 1.0))
], random_order=False)  # apply augmenters in random order
batch_size=4
train_sets = [('', 'trainval')]
rgb_means = (104, 117, 123)
rgb_std = (1, 1, 1)
p=.6
train_dataset = VOCDetection(voc_format_data_dir, train_sets, preproc(
    512, rgb_means, rgb_std, p), AnnotationTransform())
train_loader=data.DataLoader(train_dataset, batch_size,
                                                  shuffle=True, num_workers=4,
                                                  collate_fn=detection_collate)


print("Total items:{}".format(len(train_dataset)))
def read_batch(image_paths):
    images=[]
    for source_img_path in image_paths:
        img = imageio.imread(source_img_path).astype(np.float32)
        images.append(img)
    return images

augmented_data=[]

def write_file(aug_save_path,image_aug,label):

    imageio.imwrite(aug_save_path, image_aug)
    augmented_data.append([aug_save_path, label])


def augment_images(images):
    images=images.transpose((0,2,3,1))
    image_augs = seq.augment_images(images)
    return  image_augs.transpose((0,3,1,2))
