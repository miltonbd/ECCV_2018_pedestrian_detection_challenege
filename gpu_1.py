import  os
gpu=1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
from detector import Detector
from torch import optim
from augment_data import augment_images
from model_loader import *
from loss_loader import *
from data_reader_pascal_voc import VOC_CLASSES

"""
sudo nvidia-smi -pl 180
sudo nvidia-smi --gpu-reset -i 0
use command line to run the training.

todo download more images using image_utils and isic-arhive. Also, use more online resources for data. 

"""

from ssd.layers.modules.multibox_loss import MultiBoxLoss

from statics import *
from ssd import *
def get_loss_function(classifier):
    return  MultiBoxLoss(voc['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, True)

def get_model(gpu):
    return get_ssd_model(gpu, .8)

def get_optimizer(model_trainer):
    epsilon=1e-8
    momentum = 0.9
    weight_decay=5e-4
    # model_trainer.writer.add_scalar("leanring rate", learning_rate)
    # model_trainer.writer.add_scalar("epsilon", epsilon)
    # optimizer=optim.SGD(filter(lambda p: p.requires_grad, model_trainer.model.parameters()),
    #                      lr=0.001,momentum=momentum,weight_decay=weight_decay)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_trainer.model.parameters()),lr=0.01)
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model_trainer.model.parameters()), lr=0.001, momentum=0.9,
    #                       weight_decay=weight_decay)
    return optimizer

class ModelDetails(object):
    def __init__(self,gpu):
        self.model,self.model_name_str = get_model(gpu)
        self.batch_size=32
        self.epochs = 200
        self.logs_dir  = "logs/{}/{}".format(gpu,self.model_name_str)
        self.augment_images = augment_images
        self.dataset_loader=get_data_loader(self.batch_size)
        self.get_loss_function = get_loss_function
        self.get_optimizer = get_optimizer
        self.dataset=data_set_name
        self.class_names=VOC_CLASSES

def start_training(gpu):
    model_details=ModelDetails(gpu)
    detector=Detector(model_details)
    detector.load_data()
    detector.load_model()
    for epoch in range(detector.start_epoch, detector.start_epoch + model_details.epochs):
        try:
          detector.train(epoch)
          # detector.test(epoch)
        except KeyboardInterrupt:
          detector.test(epoch)
          break;
        detector.load_data()

start_training(gpu)