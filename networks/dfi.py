import torch
from torch import nn
import torch.nn.functional as F

from .resnet import resnet50_ppm

config = {'converter': [[64,256,512,1024,2048,256],[32,64,128,256,256,256]], 
          'dfims': [[[32,64,128,256,256,256], 32, 0], [[32,64,128,256,256,256], 64, 1], [[32,64,128,256,256,256], 128, 2], [[32,64,128,256,256,256], 256, 3]], 
          'dfims_id': [[0,1,2,3,4,5], [0,1,2,3,4,5], [0,1,2,3,4,5], [0,1,2,3,4,5]], 
          'tams': [32, 64, 128, 256],
          'predictors': [[[32, 64, 128, 256], True], [[32, 64, 128, 256], False], [[32, 64, 128, 256], True]], 
          'predictors_id': [0,1,2,3] } 

def gn(planes, channel_per_group=4, max_groups=32):
    groups = planes // channel_per_group
    return nn.GroupNorm(min(groups, max_groups), planes)

class Converter(nn.Module):
    def __init__(self, list_k):
        super(Converter, self).__init__()
        up = []
        for i in range(len(list_k[0])):
            up.append(nn.Sequential(
                nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), 
                gn(list_k[1][i]), 
                nn.ReLU(inplace=True),
                ))
        self.convert = nn.ModuleList(up)

    def forward(self, x):
        out = []
        for i in range(len(x)):
            out.append(self.convert[i](x[i]))
        return out

class DFIM(nn.Module): 
    def __init__(self, list_k, k, size_id, modes=3):
        super(DFIM, self).__init__()
        self.len = len(list_k)
        self.size_id = size_id
        up = []
        for i in range(len(list_k)):
            up.append(nn.Sequential(nn.Conv2d(list_k[i], k, 1, 1, bias=False), gn(k)))
        self.merge = nn.ModuleList(up)
        merge_convs, fcs, convs = [], [], []
        for m in range(modes):
            merge_convs.append(nn.Sequential(
                        nn.Conv2d(k, k//4, 1, 1, bias=False), 
                        gn(k//4), 
                        nn.ReLU(inplace=True),
                        nn.Conv2d(k//4, k, 1, 1, bias=False),
                        gn(k),
                    ))
            fcs.append(nn.Sequential(
                    nn.Linear(k, k // 4, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(k // 4, self.len, bias=False),
                ))
            convs.append(nn.Sequential(nn.Conv2d(k, k, 3, 1, 1, bias=False), gn(k), nn.ReLU(inplace=True)))
        self.merge_convs = nn.ModuleList(merge_convs)
        self.fcs = nn.ModuleList(fcs)
        self.convs = nn.ModuleList(convs)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.relu =nn.ReLU(inplace=True)

    def forward(self, list_x, mode=3):
        x_size = list_x[self.size_id].size()
        feas = []
        for i in range(len(list_x)):
            feas.append(self.merge[i](F.interpolate(list_x[i], x_size[2:], mode='bilinear', align_corners=True)).unsqueeze(dim=1))
        feas = torch.cat(feas, dim=1) # Nx6xCxHxW
        fea_sum = torch.sum(feas, dim=1) # NxCxHxW

        if mode == 3:
            outs = []
            for mode_ in range(3):
                fea_u = self.merge_convs[mode_](fea_sum)
                fea_s = self.gap(fea_u).squeeze(-1).squeeze(-1) # NxC
                fea_z = self.fcs[mode_](fea_s) # Nx6
                selects = self.softmax(fea_z) # Nx6
                feas_f = selects.reshape(x_size[0], self.len, 1, 1, 1).expand_as(feas) * feas # Nx6xCxHxW
                _, index = torch.topk(selects, 3, dim=1) # Nx3
                selected = []
                for i in range(x_size[0]):
                    selected.append(torch.index_select(feas_f, dim=1, index=index[i]))
                selected = torch.cat(selected, dim=0)
                fea_v = selected.sum(dim=1)
                outs.append(self.convs[mode_](self.relu(fea_v)))
            return torch.cat(outs, dim=0)
        else:
            fea_u = self.merge_convs[mode](fea_sum)
            fea_s = self.gap(fea_u).squeeze(-1).squeeze(-1) # NxC
            fea_z = self.fcs[mode](fea_s) # Nx6
            selects = self.softmax(fea_z) # Nx6
            feas_f = selects.reshape(x_size[0], self.len, 1, 1, 1).expand_as(feas) * feas # Nx6xCxHxW
            _, index = torch.topk(selects, 3, dim=1) # Nx3
            selected = []
            for i in range(x_size[0]):
                selected.append(torch.index_select(feas_f, dim=1, index=index[i]))
            selected = torch.cat(selected, dim=0)
            fea_v = selected.sum(dim=1)
            return self.convs[mode](self.relu(fea_v))

class TAM(nn.Module): # TAM
    reduction = 4
    def __init__(self, k):
        super(TAM, self).__init__()
        k_mid = int(k // self.reduction)
        self.attention = nn.Sequential(
            nn.Conv2d(k, k_mid, 1, 1, bias=False),
            gn(k_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(k_mid, k, 1, 1, bias=False),
            gn(k),
            nn.Sigmoid(),
        )
        self.block = nn.Sequential(nn.Conv2d(k, k, 3, 1, 1, bias=False), gn(k), nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.attention(x)
        out = torch.add(x, torch.mul(x, out))
        out = self.block(out)
        return out

class Predictor(nn.Module):
    def __init__(self, list_k, deep_sup):
        super(Predictor, self).__init__()
        self.trans = nn.ModuleList()
        for i in range(len(list_k)):
            self.trans.append(nn.Conv2d(list_k[i], 1, 1, 1))
        self.fuse = nn.Conv2d(len(list_k), 1, 1, 1)
        self.deep_sup = deep_sup

    def forward(self, list_x, x_size=None):
        up_x = []
        for i, i_x in enumerate(list_x):
            up_x.append(F.interpolate(self.trans[i](i_x), x_size[2:], mode='bilinear', align_corners=True))
        fuse = self.fuse(torch.cat(up_x, dim = 1))
        if self.deep_sup:
            return [fuse, up_x]
        else:
            return [fuse]

def extra_layer(base):
    converter, dfims, tams, predictors = [], [], [], []
    converter = Converter(config['converter'])

    for k in config['dfims']:
        dfims += [DFIM(k[0], k[1], k[2])]

    for k in config['tams']:
        tams += [TAM(k)]

    for k in config['predictors']:
        predictors += [Predictor(k[0], k[1])]

    return base, converter, dfims, tams, predictors


class DFI(nn.Module):
    def __init__(self, base, converter, dfims, tams, predictors):
        super(DFI, self).__init__()
        self.dfims_id = config['dfims_id']
        self.predictors_id = config['predictors_id']

        self.base = base
        self.converter = converter
        self.dfims = nn.ModuleList(dfims)
        self.tams = nn.ModuleList(tams)
        self.predictors = nn.ModuleList(predictors)

    def forward(self, x, mode = 3):
        x_size = x.size()
        x = self.converter(self.base(x))

        # DFIM
        dfims = []
        for k in range(len(self.dfims)):
           dfims.append(self.dfims[k]([x[i] for i in self.dfims_id[k]], mode=mode))
    
        # TAM
        tams = []
        for k in range(len(self.tams)):
            if k in self.predictors_id:
                tams.append(self.tams[k](dfims[k]))

        # Prediction
        predictions = []
        if mode == 3:
            for mode_ in range(mode):
                predictions.append(self.predictors[mode_]([tam[mode_:mode_+1] for tam in tams], x_size))
        else:
            predictions = self.predictors[mode](tams, x_size)
        return predictions

def build_model():
    return DFI(*extra_layer(resnet50_ppm()))
