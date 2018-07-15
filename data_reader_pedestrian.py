import glob

def get_test_loader_for_upload(batch_size):
    test_files=glob.glob("/media/milton/ssd1/research/competitions/data_wider_pedestrian/test_new/test_new/**.jpg")
    return test_files