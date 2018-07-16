import glob
import os

from utils.pascal_utils import write_pascal_annotation_aug
from utils.file_utils import read_text_file
train_anno_dir='/media/milton/ssd1/dataset/pedestrian/INRIAPerson/Train/annotations'
test_anno_dir='/media/milton/ssd1/dataset/pedestrian/INRIAPerson/Test/annotations'


def inria_person_to_pascal(train_anno_dir):
    anno_files = glob.glob(os.path.join(train_anno_dir, '**.txt'))
    for anno_file in anno_files:
        filename = ''
        obj_list = []

        for line in read_text_file(anno_file):
            # xml_file=os.path.join(annodir, xml_file_name)
            # image_path=os.path.abspath(os.path.join(data_dir,"train", image_name))
            # write_pascal_annotation(image_path,obj_list,xml_file)

            if 'Image filename' in line:
                filename = line.split(':')[1].strip()[1:-1]
            if 'Bounding box for object' in line:
                bounds = line.split(':')[1].split('-')
                xmin, ymin = bounds[0].strip()[1:-1].split(',')
                xmax, ymax = bounds[1].strip()[1:-1].split(',')
                xmin = int(xmin.strip())
                ymin = int(ymin.strip())
                xmax = int(xmax.strip())
                ymax = int(ymax.strip())
                obj_list.append([xmin, ymin, xmax, ymax, 1])
        image_path = os.path.join('/media/milton/ssd1/dataset/pedestrian/INRIAPerson', filename)
        xml_file = os.path.join('/media/milton/ssd1/research/competitions/data_wider_pedestrian/annotations_train',
                                os.path.basename(image_path).split('.')[0] + ".xml")
        write_pascal_annotation_aug(image_path, obj_list, xml_file)


inria_person_to_pascal(train_anno_dir)
inria_person_to_pascal(test_anno_dir)




