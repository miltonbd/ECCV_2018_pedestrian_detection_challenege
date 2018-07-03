from torchsummary import summary
from torch import nn
from data_reader import *
from utils.utils import *

from ssd.ssd import *
from statics import voc
def get_ssd_model(gpu, percentage_freeze):
    print("==>Loading SSD model...")
    model = build_ssd('train', voc['min_dim'], voc['num_classes'])

    # summary(model.cuda(), (3, height, width))
    return model,"pnas_large_{}_adam".format(gpu)
