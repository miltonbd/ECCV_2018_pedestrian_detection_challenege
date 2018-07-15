import xml.etree.ElementTree as ET
from PIL import  Image
from xml.dom import minidom
from statics import *
from data_reader import *

def write_pascal_annotation(file_name,obj_list,xml_file):
    annotation=ET.Element('annotation')
    filename=ET.SubElement(annotation,'filename')
    filename.text=file_name
    size = ET.SubElement(annotation, 'size')
    img=Image.open(file_name)
    width, height = img.size
    height_elem=ET.SubElement(size,'height')
    width_elem=ET.SubElement(size,'width')
    height_elem.text=str(height)
    width_elem.text=str(width)
    # print(obj_list)
    for i in range(0, len(obj_list), 5):
        class_index = obj_list[i]
        obj_cord = obj_list[i + 1:i + 5]
        obj_cord[2] = int(obj_cord[2]) + int(obj_cord[0])
        obj_cord[3] = int(obj_cord[3]) + int(obj_cord[1])
        object = ET.SubElement(annotation, 'object')
        get_object(object, obj_cord)

    # print(ET.dump(annotation))
    anno_txt=minidom.parseString(ET.tostring(annotation)).toprettyxml()
    text_file = open(xml_file, "w")
    text_file.write(anno_txt)
    text_file.close()
    return

def get_object(object, obj_cord):
    name = ET.SubElement(object, 'name')
    name.text = 'pedestrian'
    bndbox = ET.SubElement(object, 'bndbox')
    difficult=ET.SubElement(object,'difficult')
    difficult.text=str(0)
    xmin = ET.SubElement(bndbox, 'xmin')
    ymin = ET.SubElement(bndbox, 'ymin')
    xmax = ET.SubElement(bndbox, 'xmax')
    ymax = ET.SubElement(bndbox, 'ymax')

    xmin.text=str(obj_cord[0])
    ymin.text=str(obj_cord[1])
    xmax.text=str(obj_cord[2])
    ymax.text=str(obj_cord[3])


    return

def convert_wider_pedestrian_to_pascal():
    data=read_train_gt()
    trainvalids=[]
    for row in data:
        obj_list = row[1]
        image_name = row[0]
        annodir='/media/milton/ssd1/research/competitions/data_wider_pedestrian/VOC_Wider_pedestrian/Annotations'
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