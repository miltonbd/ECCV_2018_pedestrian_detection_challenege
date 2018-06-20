import os
from PIL import Image
from imgaug import augmenters as iaa
import numpy as np
import imageio
from utils import *
import threading

"""
5Dtensor
todo save some random tensor.
"""

aug_save_dir="/media/milton/ssd1/research/competitions/EmotiW_2018/Train_aug"

create_dir_if_not_exists(aug_save_dir)

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

aug_batch_size=100

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
