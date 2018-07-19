import glob
import os
from PIL import Image
from utils.pascal_utils import write_pascal_annotation_aug
from utils.json_utils import read_json_file

data_dir='/media/milton/ssd1/dataset/pedestrian/caltech_pedestrian/caltech-pedestrian-dataset-converter/data'
images_dir=os.path.join(data_dir,'images')
json_file=os.path.join(data_dir,'annotations.json')

data=read_json_file(json_file)
for set_key in data.keys():
    set_data=data[set_key]
    for v_key in set_data.keys():
        frames=set_data[v_key]['frames']
        for frame_key in frames.keys():
            for frame_anno  in frames[frame_key]:

                filename="{}_{}_{}.png".format(set_key.lower(),v_key,frame_key)
                file_path=os.path.join(images_dir, filename)
                # if not os.path.exists(file_path):
                #     print("{} not found".format(file_path))
                try:
                    img=Image.open(file_path)
                except Exception as e:
                    continue
                pass


