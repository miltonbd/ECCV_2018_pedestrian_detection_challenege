from utils.file_utils import read_text_file
from utils.utils import create_dir_if_not_exists
import os
import cv2
from utils.pascal_utils import read_pascal_annotation

def draw_bbox_pascal(annopath,image_dir=None):
    annotation = read_pascal_annotation(anno_path)
    image_path = annotation['filename']
    if image_dir!=None:
        image_path=os.path.join(image_dir,image_path)
    print(image_path)
    objects = annotation['objects']
    # objects=[[100,100,200,200,1]]
    create_dir_if_not_exists('pascal_images')
    img_demo_detect = cv2.imread(image_path)
    save_path = os.path.join('pascal_images', os.path.basename(image_path))
    for object in objects:
        x1, y1, x2, y2 = object[:4]
        cv2.rectangle(img_demo_detect, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    cv2.imwrite(save_path, img_demo_detect)

#anno_path='/media/milton/ssd1/research/competitions/data_wider_pedestrian/VOC_Wider_pedestrian/Annotations/img{}.xml'.format(imageid)
anno_path='/media/milton/ssd1/dataset/pascal/VOCdevkit/VOC2007/Annotations/000247.xml'

draw_bbox_pascal(anno_path,'/media/milton/ssd1/dataset/pascal/VOCdevkit/VOC2007/JPEGImages')

