from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import os
import shutil
import time
import pprint
import numpy as np
import os.path as osp
import random

import torch.nn.functional as F
import torch.nn as nn

def save_list_to_txt(name,input_list):
    f=open(name,mode='w')
    for item in input_list:
        f.write(item+'\n')
    f.close()

def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print ('use gpu:',gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()





def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print ('create folder:',path)
        os.makedirs(path)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()




_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm



def load_model(model,dir):
    model_dict = model.state_dict()
    print('loading model from :', dir)
    pretrained_dict = torch.load(dir)['params']
    if 'encoder' in list(pretrained_dict.keys())[0]:  # load from a parallel meta-trained model
        if 'module' in list(pretrained_dict.keys())[0]:
            pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
        else:
            pretrained_dict = {k: v for k, v in pretrained_dict.items()}
    else:
        pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}  # load from a pretrained model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
    model.load_state_dict(model_dict)

    return model


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def detect_grad_nan(model):
    for param in model.parameters():
        if (param.grad != param.grad).float().sum() != 0:  # nan detected
            param.grad.zero_()

def mixup_data(x, y, alpha):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()  # to(use_cuda)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    # uniform
    cx = np.random.randint(W)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    return bbx1, bbx2

def mixup_criterion(pred, y_a, y_b, lam):
    return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):  # torch.Size([120, 351, 6, 6]) torch.Size([120])
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # torch.Size([120, 351, 36])

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(inputs.size(0), inputs.size(1)).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.unsqueeze(-1)
        targets = targets.cuda()
        loss = (- targets * log_probs).mean(0).sum() 
        return loss / inputs.size(2)


def constractive_loss(args, support, query, label_1hot):
    # support: torch.Size([5, 640])
    # query: torch.Size([75, 640])
    # label_1hot: torch.Size([80，5])
    N_s, C = support.shape
    N_q, C = query.shape
    sample_total = torch.cat([query, support], 0)  # torch.Size([80, 640])
    similarity = torch.matmul(sample_total, sample_total.t()).softmax(-1)

    # positive pair
    pos_label = label_1hot.repeat(1, args.query+1)  # torch.Size([80, 80])
    pos_sim = similarity * pos_label.cuda()  # torch.Size([80, 80])
    pos_sim_exp = torch.exp(torch.div(pos_sim, args.tau)).sum(1)  # torch.Size([80]) 

    # negative pair
    neg_label = 1 - pos_label  # torch.Size([80, 80])
    neg_sim = similarity * neg_label.cuda()  # torch.Size([80, 80])
    neg_sim_exp = torch.exp(torch.div(neg_sim, args.tau)).sum(1)  # torch.Size([80]) 

    loss_i = -torch.log(torch.div(pos_sim_exp, pos_sim_exp+neg_sim_exp+1e-9))  # torch.Size([80])
    loss = torch.mean(loss_i)
              
    return loss

def constractive_loss2(args, similarity, label_1hot):
    # similarity: torch.Size([75, 5])
    # label_1hot: torch.Size([75，5])
    similarity = similarity.softmax(-1)  # torch.Size([75, 5])

    # positive pair
    pos_label = label_1hot  # torch.Size([75, 5])
    pos_sim = similarity * pos_label.cuda()  # torch.Size([75, 5])
    pos_sim_exp = torch.exp(torch.div(pos_sim, args.tau)).sum(1)  # torch.Size([75]) 

    # negative pair
    neg_label = 1 - pos_label  # torch.Size([75, 5])
    neg_sim = similarity * neg_label.cuda()  # torch.Size([75, 5])
    neg_sim_exp = torch.exp(torch.div(neg_sim, args.tau)).sum(1)  # torch.Size([75]) 

    loss_i = -torch.log(torch.div(pos_sim_exp, pos_sim_exp+neg_sim_exp+1e-9))  # torch.Size([75])
    loss = torch.mean(loss_i)
              
    return loss

