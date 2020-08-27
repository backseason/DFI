import os
import torch
from torch.utils import data
from torchvision.transforms import functional as F
import cv2
import numpy as np
from PIL import Image

class ImageDataTest(data.Dataset):
    def __init__(self, test_mode=1, sal_mode='e'):
        if test_mode == 0:
            self.image_root = './data/HED-BSDS_PASCAL/HED-BSDS/test/'
            self.image_source = './data/HED-BSDS_PASCAL/HED-BSDS/test.lst'
        elif test_mode == 1:
            if sal_mode == 'e':
                self.image_root = './data/ECSSD/Imgs/'
                self.image_source = './data/ECSSD/test.lst'
            elif sal_mode == 'p':
                self.image_root = './data/PASCALS/Imgs/'
                self.image_source = './data/PASCALS/test.lst'
            elif sal_mode == 'd':
                self.image_root = './data/DUTOMRON/Imgs/'
                self.image_source = './data/DUTOMRON/test.lst'
            elif sal_mode == 'h':
                self.image_root = './data/HKU-IS/Imgs/'
                self.image_source = './data/HKU-IS/test.lst'
            elif sal_mode == 's':
                self.image_root = './data/SOD/Imgs/'
                self.image_source = './data/SOD/test.lst'
            elif sal_mode == 't':
                self.image_root = './data/DUTS-TE/Imgs/'
                self.image_source = './data/DUTS-TE/test.lst'
        elif test_mode == 2:
            self.image_root = './data/SK-LARGE/images/test/'
            self.image_source = './data/SK-LARGE/test.lst'
        elif test_mode == 3:
            self.image_root = './demo/images/'
            self.image_source = './demo/img.lst'

        with open(self.image_source, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size = load_image_test(os.path.join(self.image_root, self.image_list[item]))
        image = torch.Tensor(image)

        return {'image': image, 'name': self.image_list[item%self.image_num], 'size': im_size}

    def __len__(self):
        return self.image_num

def get_loader(test_mode=0, sal_mode='e', pin=False):
    dataset = ImageDataTest(test_mode=test_mode, sal_mode=sal_mode)
    data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1,
                                      pin_memory=pin)
    return data_loader

def load_image_test(pah):
    if not os.path.exists(pah):
        print('File Not Exists')
    im = cv2.imread(pah)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2,0,1))
    return in_, im_size
