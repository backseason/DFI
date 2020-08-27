import torch
from torch.nn import utils, functional as F
from networks.dfi import build_model
import numpy as np
import os
import cv2

class Solver(object):
    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.config = config
        self.net = build_model()
        if self.config.cuda:
            self.net = self.net.cuda()
        print('Loading pre-trained model from %s...' % self.config.model)
        self.net.load_state_dict(torch.load(self.config.model))
        self.net.eval()

    def test(self, test_mode=0):
        mode_name = ['edge', 'sal', 'skel']
        EPSILON = 1e-8
        img_num = len(self.data_loader)
        for i, data_batch in enumerate(self.data_loader):
            images, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            if test_mode == 0: # edge task
                images = images.numpy()[0].transpose((1,2,0))
                scale = [0.5, 1, 1.5, 2] # multi-scale testing as commonly done
                multi_fuse = np.zeros(im_size, np.float32)
                for k in range(0, len(scale)):
                    im_ = cv2.resize(images, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
                    im_ = im_.transpose((2, 0, 1))
                    im_ = torch.Tensor(im_[np.newaxis, ...])

                    with torch.no_grad():
                        if self.config.cuda:
                            im_ = im_.cuda()
                        preds = self.net(im_, mode=test_mode)
                        preds_i = []
                        for p in preds[1]:
                            preds_i.append(np.squeeze(torch.sigmoid(p).cpu().data.numpy()))
                        pred_fuse = np.squeeze(torch.sigmoid(preds[0]).cpu().data.numpy())
                        pred = (pred_fuse + sum(preds_i)) / (1.0 + len(preds_i))

                        pred = (pred - np.min(pred) + EPSILON) / (np.max(pred) - np.min(pred) + EPSILON)

                        pred = cv2.resize(pred, (im_size[1], im_size[0]), interpolation=cv2.INTER_LINEAR)
                        multi_fuse += pred

                multi_fuse /= len(scale)
                multi_fuse = 255 * (1 - multi_fuse)
                cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name[test_mode] + '.png'), multi_fuse)

            elif test_mode == 1: # saliency task
                with torch.no_grad():
                    if self.config.cuda:
                        images = images.cuda()
                    preds = self.net(images, mode=test_mode)
                    pred = np.squeeze(torch.sigmoid(preds[0]).cpu().data.numpy())

                    multi_fuse = 255 * pred
                    cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name[test_mode] + '.png'), multi_fuse)

            elif test_mode == 2: # skeleton task
                images = images.numpy()[0].transpose((1,2,0))
                scale = [0.5, 1, 1.5] # multi-scale testing as commonly done
                multi_fuse = np.zeros(im_size, np.float32)
                for k in range(0, len(scale)):
                    im_ = cv2.resize(images, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
                    im_ = im_.transpose((2, 0, 1))
                    im_ = torch.Tensor(im_[np.newaxis, ...])

                    with torch.no_grad():
                        if self.config.cuda:
                            im_ = im_.cuda()
                        preds = self.net(im_, mode=test_mode)
                        pred_fuse = np.squeeze(torch.sigmoid(preds[0]).cpu().data.numpy())

                        pred = pred_fuse
                        pred = (pred - np.min(pred) + EPSILON) / (np.max(pred) - np.min(pred) + EPSILON)

                        pred = cv2.resize(pred, (im_size[1], im_size[0]), interpolation=cv2.INTER_LINEAR)
                        multi_fuse += pred

                multi_fuse /= len(scale)
                multi_fuse = 255 * (1 - multi_fuse)
                cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name[test_mode] + '.png'), multi_fuse)
            elif test_mode == 3: # all tasks
                with torch.no_grad():
                    if self.config.cuda:
                        images = images.cuda()
                    preds = self.net(images, mode=test_mode)
                    pred_edge = np.squeeze(torch.sigmoid(preds[0][0]).cpu().data.numpy())
                    pred_sal = np.squeeze(torch.sigmoid(preds[1][0]).cpu().data.numpy())
                    pred_skel = np.squeeze(torch.sigmoid(preds[2][0]).cpu().data.numpy())

                    pred_edge = (pred_edge - np.min(pred_edge) + EPSILON) / (np.max(pred_edge) - np.min(pred_edge) + EPSILON)
                    pred_skel = (pred_skel - np.min(pred_skel) + EPSILON) / (np.max(pred_skel) - np.min(pred_skel) + EPSILON)

                    cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name[0] + '.png'), 255 * (1 - pred_edge))
                    cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name[1] + '.png'), 255 * pred_sal)
                    cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name[2] + '.png'), 255 * (1 - pred_skel))

        print('Testing Finished.')

