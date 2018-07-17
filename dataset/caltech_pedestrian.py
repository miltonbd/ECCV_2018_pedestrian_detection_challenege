import glob
import os

from utils.pascal_utils import write_pascal_annotation_aug
from utils.json_utils import read_json_file

data_dir='/media/milton/ssd1/dataset/pedestrian/caltech_pedestrian/caltech-pedestrian-dataset-converter/data'
images_dir=os.path.join(data_dir,'images')
json_file=os.path.join(data_dir,'annotations.json')

data=read_json_file(json_file)
for set_key in data.keys():
    set_data=data[set_key]
    for v_key in set_data.keys():
        v_data=set_data[v_key].frames:
        



    break