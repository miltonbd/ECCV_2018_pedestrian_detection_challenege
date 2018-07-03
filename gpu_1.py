import  os
gpu=1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
from detector import Detector
from torch import optim
from augment_data import augment_images
from model_loader import *
from loss_loader import *

"""
sudo nvidia-smi -pl 180

use command line to run the training.

todo download more images using image_utils and isic-arhive. Also, use more online resources for data. 

"""

from ssd.layers.modules.multibox_loss import MultiBoxLoss

from statics import *
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
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_trainer.model.parameters()),
                            lr=0.01)
    return optimizer

class ModelDetails(object):
    def __init__(self,gpu):
        self.model,self.model_name_str = get_model(gpu)
        self.batch_size=20
        self.epochs = 200
        self.logs_dir  = "logs/{}/{}".format(gpu,self.model_name_str)
        self.augment_images = augment_images
        self.dataset_loader=get_data_loader(self.batch_size)
        self.get_loss_function = get_loss_function
        self.get_optimizer = get_optimizer
        self.dataset=data_set_name

def start_training(gpu):
    model_details=ModelDetails(gpu)
    clasifier=Detector(model_details)
    clasifier.load_data()
    clasifier.load_model()
    for epoch in range(clasifier.start_epoch, clasifier.start_epoch + model_details.epochs):
        try:
          clasifier.train(epoch)
          clasifier.test(epoch)
        except KeyboardInterrupt:
          clasifier.test(epoch)
          break;
        clasifier.load_data()

start_training(gpu)