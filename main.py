import os
from torch import nn,optim
import torchvision
from classifier import Classifier
from models.vgg import vgg19_bn
from augment_data import augment_images
import  os
import time
import torchvision.models as models
import torch

from torch.nn import functional
import torch.nn.functional as F
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
"""

sudo nvidia-smi -i 0 -pl 180
"""

def load_data(self):

    print('==> Preparing data..')
    self.trainloader, self.testloader = get_data_sets(self.model_details.batch_size_train,self.model_details.batch_size_test )
    train_count = len(self.trainloader) * self.model_details.batch_size_train
    test_count = len(self.testloader) * self.model_details.batch_size_test
    print('==> Total examples, train: {}, test:{}'.format(train_count, test_count))

def get_vgg_model():
    model_conv = vgg19_bn(pretrained=True)

    # Number of filters in the bottleneck layer
    num_ftrs = model_conv.classifier[6].in_features

    # convert all the layers to list and remove the last one
    features = list(model_conv.classifier.children())[:-1]

    ## Add the last layer based on the num of classes in our dataset
    features.extend([nn.Linear(num_ftrs, 3)])

    ## convert it into container and add it to our model class.
    model_conv.classifier = nn.Sequential(*features)
    return model_conv

class ModelDetails(object):

    def __init__(self):
        model = get_vgg_model()
        # children=list(model.classifier.children())[-1]
        # classes=torch.nn.Sequential(children)
        # model.classifier = torch.nn.Sequential(classes, torch.nn.Linear(4096, 3))

        # model.avg_pool = nn.AvgPool2d(2, count_include_pad=False)
        # model.last_linear = nn.Linear(1536, 3)

        # todo use ssd and yolo to detct both face and class
        # todo freeze few layers in first
        # todo augement data set and use random crop, pair augment
        # todo mix the emotional images like image pair
        # todo debug the detected face patch.
        # todo add random
        # todo new loss function
        # todo new optimization
        # todo new training proceudres
        # todo augment image after every epoch.
        ## Freezing the first few layers. Here I am freezing the first 7 layers

        # num_layers_freeze=5
        # for name, child in model.named_children():
        #     if name=='features':
        #         for name, chile in child.named_children():
        #             if int(name)<num_layers_freeze:
        #                 for params in chile.parameters():
        #                     params.requires_grad = False

        self.model= model
        self.learning_rate = 0.01
        self.epsilon=1e-8
        self.optimizer = "adam"
        self.model_name_str = "inceptionv4"
        self.batch_size_train=50
        self.batch_size_test=50
        self.epochs = 200
        self.logs_dir  = "logs/inceptionv4/no_aug"
        self.augmentor = augment_images

model_details=ModelDetails()
model_details.model_name= "inceptionv4"

clasifier=Classifier(model_details)
clasifier.load_data()
clasifier.load_model()
for epoch in range(clasifier.start_epoch, clasifier.start_epoch + model_details.epochs):
    try:
      clasifier.train(epoch)
      time.sleep(2)
      clasifier.test(epoch)
    except KeyboardInterrupt:
      clasifier.test(epoch)
      break;
    clasifier.load_data()
