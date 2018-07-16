import xml.etree.ElementTree as ET
from PIL import  Image
from xml.dom import minidom
from statics import *
from data_reader import *
from utils.utils import create_dir_if_not_exists
from utils.pascal_utils import *


def convert_wider_pedestrian_to_pascal():
    data=read_train_gt()
    trainvalids=[]
    for row in data:
        obj_list = row[1]
        image_name = row[0]
        annodir='/media/milton/ssd1/research/competitions/data_wider_pedestrian/VOC_Wider_pedestrian/Annotations'
        create_dir_if_not_exists(annodir)
        create_dir_if_not_exists('/media/milton/ssd1/research/competitions/data_wider_pedestrian/VOC_Wider_pedestrian/JPEGImages')
        xml_file_name=image_name.split('.')[0]+".xml"
        xml_file=os.path.join(annodir, xml_file_name)
        image_path=os.path.abspath(os.path.join(data_dir,"train", image_name))
        write_pascal_annotation(image_path,obj_list,xml_file)

        voc_anno_train_dir="/media/milton/ssd1/research/competitions/data_wider_pedestrian/annotations_train"
        if not os.path.exists(voc_anno_train_dir):
            os.makedirs(voc_anno_train_dir)
        anno_path=os.path.join(voc_anno_train_dir,xml_file_name)
        write_pascal_annotation(image_path,obj_list,anno_path)

        trainvalids.append(image_name.split('.')[0])
        # break
    with open('/media/milton/ssd1/research/competitions/data_wider_pedestrian/VOC_Wider_pedestrian/ImageSets/Main/trainval.txt', mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(trainvalids))
    testids=[]
    for row in read_val_gt():
        obj_list = row[1]
        image_name = row[0]
        annodir='/media/milton/ssd1/research/competitions/data_wider_pedestrian/VOC_Wider_pedestrian/Annotations'
        xml_file_name=image_name.split('.')[0]+".xml"
        xml_file=os.path.join(annodir, xml_file_name)
        image_path=os.path.abspath(os.path.join(data_dir,"val", image_name))
        write_pascal_annotation(image_path,obj_list,xml_file)
        testids.append(image_name.split('.')[0])

        voc_anno_train_dir = "/media/milton/ssd1/research/competitions/data_wider_pedestrian/annotations_valid"
        if not os.path.exists(voc_anno_train_dir):
            os.makedirs(voc_anno_train_dir)
        anno_path = os.path.join(voc_anno_train_dir, xml_file_name)
        write_pascal_annotation(image_path, obj_list, anno_path)
        testids.append(image_name.split('.')[0])

        # break
    with open('/media/milton/ssd1/research/competitions/data_wider_pedestrian/VOC_Wider_pedestrian/ImageSets/Main/test.txt', mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(testids))



if __name__ == '__main__':
    convert_wider_pedestrian_to_pascal()