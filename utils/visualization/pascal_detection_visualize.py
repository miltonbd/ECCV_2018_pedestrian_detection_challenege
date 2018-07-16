from utils.file_utils import read_text_file
import os
import cv2
from utils.pascal_utils import read_pascal_annotation
imageid='19946'
image_path='/media/milton/ssd1/research/competitions/data_wider_pedestrian/VOC_Wider_pedestrian/JPEGImages/img{}.jpg'.format(imageid)
anno_path='/media/milton/ssd1/research/competitions/data_wider_pedestrian/VOC_Wider_pedestrian/Annotations/img{}.xml'.format(imageid)
objects=read_pascal_annotation(anno_path)['objects']

img_demo_detect = cv2.imread(image_path)
save_path=os.path.join('pascal_images', os.path.basename(image_path))
for object in objects:
    x1, y1, x2,y2=object[:4]
    cv2.rectangle(img_demo_detect, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
cv2.imwrite(save_path, img_demo_detect)

