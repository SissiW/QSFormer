import torch.nn as nn
import torch
import torch.nn.functional as F

# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
from torch.autograd import Variable
import math
import numpy as np
from torch.nn.utils.weight_norm import WeightNorm
from torch.distributions import Bernoulli

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(outdim)
        self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(outdim)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:

            self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
            self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size


    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, args,block=BasicBlock, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        return x

class ResNet18(nn.Module):
    maml = False  # Default

    def __init__(self, args, block=SimpleBlock, list_of_num_layers=[2, 2, 2, 2], list_of_out_dims=[64, 128, 256, 512], flatten=False):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet18, self).__init__()
        assert len(list_of_num_layers) == 4, 'Can have only four stages'

        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
        bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)
        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0) and i != 3
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            # self.final_feat_dim = indim

        self.feat_dim = [512, 14, 14]
        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        out = self.trunk(x)
        # out = out.view(out.size(0), -1)
        return out


if __name__=='__main__':
    from thop import profile
    model=ResNet()
    input = torch.FloatTensor(1, 3, 84, 84)
    macs, paras = profile(model, inputs=(input,))
    print('FLOPs=', macs, (macs/1e9))  
    # bs=80: FLOPs= 281840271360.0  281.84027136
    # bs=1:  FLOPs= 3523003392.0    3.523003392
    print('Parameters=', paras, (paras/1e6))  # Parameters= 12424320.0 12.42432
    print(aa)
    out = model(input)
