from statics import voc
from ssd import *
def get_ssd_model(gpu, percentage_freeze):
    print("==>Loading SSD model...")
    model = build_ssd('train', voc['min_dim'], voc['num_classes'])

    # summary(model.cuda(), (3, height, width))
    return model,"ssd_{}_adam".format(gpu)
