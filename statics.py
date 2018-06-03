import os

data_dir="../data_wider_pedestrian"

train_bbx_gt_file=os.path.join(data_dir,'train_annotations.txt')
train_img_dir=os.path.join(data_dir,'train')

val_bbx_gt_file=os.path.join(data_dir,'val_annotations.txt')
val_img_dir=os.path.join(data_dir,'val')