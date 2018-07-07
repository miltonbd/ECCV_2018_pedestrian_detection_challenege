from statics import voc
import argparse
import pickle
import time

import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

from data import VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, \
    COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
from layers.functions import Detect, PriorBox
from layers.modules import MultiBoxLoss
from utils.nms_wrapper import nms
from utils.timer import Timer

def get_ssd_model(args):
    save_folder = os.path.join(args.save_folder, args.version + '_' + str(args.size), args.date)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    test_save_dir = os.path.join(save_folder, 'ss_predict')
    if not os.path.exists(test_save_dir):
        os.makedirs(test_save_dir)
    gpu=args.gpu
    img_dim = args.size
    num_classes=args.num_classes
    print("==>Loading SSD model...")
    if args.version == 'RFB_vgg':
        from models.RFB_Net_vgg import build_net
    elif args.version == 'RFB_E_vgg':
        from models.RFB_Net_E_vgg import build_net
    elif args.version == 'RFB_mobile':
        from models.RFB_Net_mobile import build_net

        cfg = COCO_mobile_300
    elif args.version == 'SSD_vgg':
        from models.SSD_vgg import build_net
    elif args.version == 'FSSD_vgg':
        from models.FSSD_vgg import build_net
    elif args.version == 'FRFBSSD_vgg':
        from models.FRFBSSD_vgg import build_net
    else:
        print('Unkown version!')
    net = build_net(int(img_dim), num_classes)
    # model(model.cuda(), (3, height, width))
    if not args.resume_net:
        base_weights = torch.load(args.basenet)
        print('Loading base network...')
        net.base.load_state_dict(base_weights)

        def xavier(param):
            init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        init.kaiming_normal(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0

        print('Initializing weights...')
        # initialize newly added layers' weights with kaiming_normal method
        net.extras.apply(weights_init)
        net.loc.apply(weights_init)
        net.conf.apply(weights_init)
        if args.version == 'FSSD_vgg' or args.version == 'FRFBSSD_vgg':
            net.ft_module.apply(weights_init)
            net.pyramid_ext.apply(weights_init)
        if 'RFB' in args.version:
            net.Norm.apply(weights_init)
        if args.version == 'RFB_E_vgg':
            net.reduce.apply(weights_init)
            net.up_reduce.apply(weights_init)

    else:
        # load resume network
        resume_net_path = os.path.join(save_folder, args.version + '_' + args.dataset + '_epoches_' + \
                                       str(args.resume_epoch) + '.pth')
        print('Loading resume network', resume_net_path)
        state_dict = torch.load(resume_net_path)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    return net,"ssd_{}_adam".format(gpu)
