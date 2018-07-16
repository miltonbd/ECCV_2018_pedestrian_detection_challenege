from imgaug import augmenters as iaa
import imageio
from data.voc0712 import *

from utils.utils import *
import numpy as np
import os
import torch

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

class VocDataset(Dataset):
    def __init__(self, dataset):
        self.dataset=dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        ia.seed(1)
        xml_file=os.path.join(voc_format_data_dir,'Annotations',self.dataset[index]+".xml")
        annotations=read_pascal_annotation(xml_file)
        boxes=np.asarray(annotations['objects'])
        image_path=os.path.join(voc_format_data_dir,'JPEGImages',self.dataset[index]+".jpg")
        image=io.imread(image_path)

        boxes_aug=[]
        for box in boxes:
            boxes_aug.append(ia.BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]))

        # image = ia.quokka(size=(220, 220))
        bbs = ia.BoundingBoxesOnImage(boxes_aug, shape=image.shape)

        seq = iaa.Sequential([
            iaa.Scale(220)
        ])

        # Make our sequence deterministic.
        # We can now apply it to the image and then to the BBs and it will
        # lead to the same augmentations.
        # IMPORTANT: Call this once PER BATCH, otherwise you will always get the
        # exactly same augmentations for every batch!
        seq_det = seq.to_deterministic()

        # Augment BBs and images.
        # As we only have one image and list of BBs, we use
        # [image] and [bbs] to turn both into lists (batches) for the
        # functions and then [0] to reverse that. In a real experiment, your
        # variables would likely already be lists.
        image_aug = seq_det.augment_images([image])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]

        # print coordinates before/after augmentation (see below)
        # use .x1_int, .y_int, ... to get integer coordinates
        for i in range(len(bbs.bounding_boxes)):
            before = bbs.bounding_boxes[i]
            after = bbs_aug.bounding_boxes[i]
            print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
                i,
                before.x1, before.y1, before.x2, before.y2,
                after.x1, after.y1, after.x2, after.y2)
                  )

        # image with BBs before/after augmentation (shown below)
        image_before = bbs.draw_on_image(image, thickness=2)
        image_after = bbs_aug.draw_on_image(image_aug, thickness=2, color=[0, 0, 255])

        img_resized=resize(img,(target_w,target_h),preserve_range=True)
        res={
            'img':img_resized,
            'annot':boxes.astype(np.float64)
        }
        return res

def collate_aug(batch):
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        imgs.append(sample['img'])
        targets.append(sample['annot'])
    return (imgs, targets)

print("Total items:{}".format(len(train_dataset)))

voc_dataset=VocDataset(train_dataset)

train_dataloader = DataLoader(voc_dataset, num_workers=4, batch_size=batch_size, collate_fn=collate_aug)

for batch_idx,data in enumerate(train_dataloader):
    print(data)

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
