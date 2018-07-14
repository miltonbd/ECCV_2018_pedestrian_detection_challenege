import sys
import os
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import os
import cv2
import os
"""
face=[[x1,x2,x2,y2]]
"""


def draw_rectangle_w_h_box(img_path, faces, save_dir='./detected_face'):
    create_dir_if_not_exists(save_dir)
    img_face_detect = cv2.imread(img_path)
    for face in faces:
        x1, y1, x2, y2 = face
        cv2.rectangle(img_face_detect, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), img_face_detect)

def draw_rectangle(img_path, faces, save_dir='./detected_face'):
    create_dir_if_not_exists(save_dir)
    img_face_detect = cv2.imread(img_path)
    for face in faces:
        x1, y1, x2, y2 = face
        cv2.rectangle(img_face_detect, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), img_face_detect)

def drawbbox(file_name,bbox,save_dir):
    source_img = Image.open(file_name).convert("RGBA")

    draw = ImageDraw.Draw(source_img)
    # draw.rectangle(((0, 00), (100, 100)), fill="black")
    # draw.text((20, 70), "something123", font=ImageFont.truetype("font_path123"))

    create_dir_if_not_exists(save_dir)
    save_file=os.path.join(save_dir,os.path.basename(file_name))
    source_img.save(save_file, "JPEG")

def get_total_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return  params

def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def check_if_exists(dir):
    return os.path.exists(dir)

def progress_bar(progress, count ,message):
    sys.stdout.write('\r' + "{} of {}: {}".format(progress, count, message))
