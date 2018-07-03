from focal_loss import FocalLoss
from torch import nn
gamma = 2

def get_focal_loss(classifier):
    print("==> Using Focal Loss.....")
    classifier.writer.add_text('Info', "Using Focal Loss ")
    return FocalLoss(gamma)

def get_cross_entropy(classifier):
    print("==> Using CrossEntropy.....")
    classifier.writer.add_text('Info', "Using Cross Entropy Loss ")
    return nn.CrossEntropyLoss()

def get_vat_cross_entropy(classifier):
    print("==> Using Adversarial Training Cross Entropy.....")
    pass