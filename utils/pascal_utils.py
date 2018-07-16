from PIL import Image
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


def write_pascal_annotation_aug(file_name,obj_list,xml_file):
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
    for i,obj in enumerate(obj_list):
        class_index = obj[4]
        obj_cord = obj[0:4]
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


def read_pascal_annotation(anno_file):
    """

    :param anno_file:
    :return:

    """
    tree = ET.parse(anno_file)
    root = tree.getroot()
    filename=root.find('filename').text
    height=int(root.find('size/height').text)
    width=int(root.find('size/width').text)
    objs=root.findall('object')
    objects=[]
    for obj in objs:
        class_label=obj.find('name').text
        xmin=int(float(obj.find('bndbox/xmin').text))
        xmax=int(float(obj.find('bndbox/xmax').text))
        ymin=int(float(obj.find('bndbox/ymin').text))
        ymax=int(float(obj.find('bndbox/ymax').text))
        objects.append([xmin,ymin,xmax,ymax,1])
    res={
        'filename':filename,
        'height':height,
         'width':width,
         'objects':objects
         }
    return res