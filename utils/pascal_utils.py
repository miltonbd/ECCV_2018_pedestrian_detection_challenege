import xml.etree.ElementTree as ET

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
        xmin=int(obj.find('bndbox/xmin').text)
        xmax=int(obj.find('bndbox/xmax').text)
        ymin=int(obj.find('bndbox/ymin').text)
        ymax=int(obj.find('bndbox/ymax').text)
        objects.append([xmin,ymin,xmax,ymax,1])
    res={
        'filename':filename,
        'height':height,
         'width':width,
         'objects':objects
         }
    return res