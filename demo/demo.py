from utils.file_utils import read_text_file
import os
import cv2

val_dir='/media/milton/ssd1/research/competitions/data_wider_pedestrian/val'
for line in read_text_file('scores.txt'):
    line_arr=line.split(' ')
    image_name=line_arr[0]
    image_path=os.path.join(val_dir,image_name)
    save_path=os.path.join('out',image_name)
    if os.path.exists(save_path):
        image_path=save_path
    print(image_path)
    img_face_detect = cv2.imread(image_path)
    print(line_arr)
    x1, y1, w, h = line_arr[2:]
    x1=float(x1)
    y1=float(y1)
    w=float(w)
    h=float(h.strip())
    x2=int(x1)+int(w)
    y2=int(y1)+int(h)
    cv2.rectangle(img_face_detect, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    print(save_path)
    print(img_face_detect.shape)
    cv2.imwrite(save_path, img_face_detect)
    