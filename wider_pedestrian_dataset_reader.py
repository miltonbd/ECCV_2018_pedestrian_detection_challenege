from statics import *
import os
from scipy.io import loadmat


"""
 	    Train 	Val 	Test
Images 	11500 	5000 	3500
Labels 	46513 	19696 	
"""

def read_train_gt():
    annotations={}
    with open(train_bbx_gt_file,'r') as train_bbx_file:
        content=train_bbx_file.readlines();
        for line in content:
            line_list=line.split(" ")
            # print(len(line_list))
            file_name=line_list[0]
            annotations[file_name]=[]
            for idx in range(1,len(line_list)-1,5):
                class_num=line_list[idx]
                left=line_list[idx+1]
                top=line_list[idx+2]
                w=line_list[idx+3]
                h=line_list[idx+4]
                annotations[file_name].append([class_num,left,top,w,h])
    return annotations


def read_val_gt():
    annotations={}
    with open(val_bbx_gt_file,'r') as train_bbx_file:
        content=train_bbx_file.readlines();
        for line in content:
            line_list=line.split(" ")
            # print(len(line_list))
            file_name=line_list[0]
            annotations[file_name]=[]
            for idx in range(1,len(line_list)-1,5):
                class_num=line_list[idx]
                left=line_list[idx+1]
                top=line_list[idx+2]
                w=line_list[idx+3]
                h=line_list[idx+4]
                annotations[file_name].append([class_num,left,top,w,h])
    return annotations


# annotations=read_train_gt()
# print(len(annotations))
#
# count=0
# for anno in annotations:
#     count+=len(annotations[anno])
# print(count)
# # annos= read_train_gt()
# # for anno in annos:
# #     print(annos[anno])