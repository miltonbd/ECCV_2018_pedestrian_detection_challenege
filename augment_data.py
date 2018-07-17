from imgaug import augmenters as iaa
import imageio
from data.voc0712 import *

from utils.utils import *
import numpy as np
import os
import torch
import threading
"""
5Dtensor
todo save some random tensor.
generate augmentation data in pascal voc format and later append them in mscoco files in training.
"""
data_dir="/media/milton/ssd1/research/competitions/data_wider_pedestrian"
create_dir_if_not_exists(data_dir)
voc_format_data_dir=os.path.join(data_dir,'VOC_Wider_pedestrian')
batch_size=4
trainval_txt=os.path.join(voc_format_data_dir,'ImageSets','Main','trainval.txt')
from utils.file_utils import read_text_file
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from utils.pascal_utils import read_pascal_annotation
from torchvision.transforms import Compose
from skimage import io, color
from skimage.transform import resize
trainvals=read_text_file(trainval_txt)
train_dataset = [img.strip() for img in trainvals]
import copy
import imgaug
from imgaug import augmenters as iaa
import imgaug as ia
from utils.pascal_utils import *
from utils.utils import progress_bar
import time

ia.seed(1)
JPEG_dir='/media/milton/ssd1/research/competitions/data_wider_pedestrian/VOC_Wider_pedestrian/JPEGImages_aug'
# anno_dir="/media/milton/ssd1/research/competitions/data_wider_pedestrian/VOC_Wider_pedestrian/Annotations_512"
anno_dir="/media/milton/ssd1/research/competitions/data_wider_pedestrian/annotations_train"
JPEG_aug_dir='/media/milton/ssd1/research/competitions/data_wider_pedestrian/VOC_Wider_pedestrian/JPEGImages_aug'
anno_aug_dir="/media/milton/ssd1/research/competitions/data_wider_pedestrian/VOC_Wider_pedestrian/Annotations_aug"

create_dir_if_not_exists(JPEG_dir)
create_dir_if_not_exists(anno_dir)
# create_dir_if_not_exists(JPEG_aug_dir)
# create_dir_if_not_exists(anno_aug_dir)

class VocDataset(Dataset):
    def __init__(self, dataset, anno_dir):
        self.dataset=dataset
        self.anno_dir=anno_dir

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        xml_file=os.path.join(voc_format_data_dir,self.anno_dir,self.dataset[index]+".xml")
        annotations=read_pascal_annotation(xml_file)
        boxes=np.asarray(annotations['objects'])
        image_path=annotations['filename']
        if not os.path.exists(image_path):
            print("{} does not exists".format(image_path))
        image=io.imread(image_path)
        res={
            'img':image,
            'annot':boxes.astype(np.float64)
        }
        return res

def collate_aug(batch):
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        imgs.append(sample['img'])
        targets.append(sample['annot'])
    return (np.asarray(imgs), targets)

def augment(images, boxes, batch_idx, size=556):
    from imgaug import parameters as iap
    boxes_augs = []
    for box1 in boxes:
        for box in box1:
            boxes_augs.append(ia.BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))

    bbs = ia.BoundingBoxesOnImage(boxes_augs, shape=images[0].shape)
    seq = iaa.Sequential([
        iaa.OneOf([
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Flipud(0.5),  # horizontal flips
            iaa.CropAndPad(percent=(-0.15, 0.15)),  # random crops
        ]),

        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.Add((-40, 40)),
                      iaa.GaussianBlur(sigma=(0, 0.5)),
                      # Invert each image's chanell with 5% probability.
                      # This sets each pixel value v to 255-v.
                      iaa.Invert(0.05, per_channel=True),  # invert color channels

                      # Add a value of -10 to 10 to each pixel.
                      iaa.Add((-10, 10), per_channel=0.5),

                      # Change brightness of images (50-150% of original value).
                      # iaa.Multiply((0.5, 1.5), per_channel=0.5),
                      # iaa.ContrastNormalization((0.75, 1.5)),


                      # Improve or worsen the contrast of images.
                      # Convert each image to grayscale and then overlay the
                      # result with the original with random alpha. I.e. remove
                      # colors with varying strengths.

                      ),
        # Strengthen or weaken the contrast in each image.
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.OneOf([iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-10, 10),
            shear=(-4, 4)
        ),
        iaa.Multiply((0.5, 1.5)),
        iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
        iaa.Sequential([
            iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
            iaa.WithChannels(0, iaa.Add((50, 100))),
            iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
        ]),
        iaa.Superpixels(n_segments=100),
        iaa.Invert(0.2),
        iaa.CoarseSaltAndPepper(size_percent=0.05),
        iaa.ElasticTransformation(2),
        iaa.SimplexNoiseAlpha(
            first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True)),
        iaa.FrequencyNoiseAlpha(
            first=iaa.Multiply(iap.Choice([0.5, 1.5]), per_channel=True)
        ),
        iaa.Grayscale(alpha=(0.0, 1.0)),
        # iaa.PiecewiseAffine(scale=(0.01, 0.05))
        ]),


    ], random_order=True)  # apply augmenters in random order

    seq_det = seq.to_deterministic()  # Call this once PER BATCH, otherwise you will always get the to get random
    images_data=images
    # for i,img in enumerate(images):
    #     images_data[i,:,:,:]=img[:,:,:]
    aug_images = seq_det.augment_images(images_data)
    bbs_augs = seq_det.augment_bounding_boxes([bbs])
    for i, aug_bb in enumerate(bbs_augs):
        idx_i = batch_idx + i
        imgid = train_dataset[idx_i]
        save_augs(JPEG_dir, anno_dir, idx_i, aug_images[i], aug_bb, imgid + "_" + str(idx_i))
    return

print("Total items:{}".format(len(train_dataset)))

voc_dataset_resize=VocDataset(train_dataset, 'Annotations')

train_dataloader_resize = DataLoader(voc_dataset_resize, num_workers=4, batch_size=1, collate_fn=collate_aug)

def augment_images(images,boxes):
    boxes_augs = []
    for box1 in boxes:
        for box in box1:
            boxes_augs.append(ia.BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))

    bbs = ia.BoundingBoxesOnImage(boxes_augs, shape=images[0].shape)

    seq = iaa.Sequential([
        iaa.Scale(400),
    ])

    seq_det = seq.to_deterministic()  # Call this once PER BATCH, otherwise you will always get the to get random
    images_data = images
    # for i,img in enumerate(images):
    #     images_data[i,:,:,:]=img[:,:,:]
    image_augs = seq_det.augment_images(images_data)
    bbs_augs = seq_det.augment_bounding_boxes([bbs])
    return (image_augs, bbs_augs)

def save_augs(JPEG_dir,anno_dir, idx_i,aug_img,aug_bb,imgid):
    progress_bar(idx_i,len(train_dataset)," Augmenting.........")
    save_path_img = os.path.join(JPEG_dir, "{}.jpg".format(imgid))
    imageio.imwrite(save_path_img, aug_img)
    save_path_xml = os.path.join(anno_dir, "{}.xml".format(imgid))
    bbox = []
    for bb in aug_bb.bounding_boxes:
        bbox.append([bb.x1, bb.y1, bb.x2, bb.y2, 1])
    write_pascal_annotation_aug(save_path_img, bbox, save_path_xml)


def image_aug():
    train_dataset_aug=train_dataset
    train_dataset_aug.extend(train_dataset)
    train_dataset_aug.extend(train_dataset)
    voc_dataset_aug = VocDataset(train_dataset_aug, 'Annotations')
    train_dataloader_resize = DataLoader(voc_dataset_aug, num_workers=4, batch_size=1, collate_fn=collate_aug)
    threads=[]
    for batch_idx, data in enumerate(train_dataloader_resize):
        # save_augs(JPEG_dir,anno_dir,idx_i,aug_images,aug_bb)
        t = threading.Thread(target=augment, args=(data[0], data[1], batch_idx))
        threads.append(t)
        t.start()
        time.sleep(.00001)

    for t in threads:
        t.join()
#
# voc_dataset_aug=VocDataset(train_dataset, 'Annotations_512')
#
# train_dataloader_aug = DataLoader(voc_dataset_aug, num_workers=4, batch_size=1, collate_fn=collate_aug)
#
# def aug_train_images_from_512():
#     for batch_idx, data in enumerate(train_dataloader_aug):
#         aug_images, aug_bbs = augment_images(data[0], data[1])
#         for i, aug_bb in enumerate(aug_bbs):
#             idx_i = batch_idx + i
#             imgid=train_dataset[idx_i]
#             save_augs(JPEG_aug_dir, anno_aug_dir, idx_i, aug_images[i], aug_bb,imgid+"_"+str(idx_i))
#             # t = threading.Thread(target=save_augs, args=(JPEG_aug_dir, anno_aug_dir, idx_i, aug_images[i], aug_bb,i))
#             # t.start()
#             break
#         break

# resize_to_512() run this first to resize the images to 512 keeping the bounding boxes.

# aug_train_images_from_512()
image_aug()
