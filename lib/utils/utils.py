import os
import shutil
import time
import pprint

import torch
import numpy as np
from scipy import stats


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
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
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2

def copyModel(premodel_dict, model):
    model_dict = model.state_dict()
    premodel_dict = {k: v for k, v in premodel_dict.items() if k in model_dict}
    model_dict.update(premodel_dict)
    model.load_state_dict(model_dict)

    return model

def cosine(prototype, data):
    a = prototype.size()[0]
    b = data.size()[0]
    prototype = prototype.repeat(b, 1)
    data = data.repeat(a, 1)

    cosine = torch.cosine_similarity(prototype, data, dim=1)

    return cosine


def CI(allacc):
    allmeanacc = []

    for i in range(10000):
        sample = torch.randperm(len(allacc))[:600]
        sampleacc = np.array(allacc[sample])
        meanacc = np.mean(sampleacc)
        allmeanacc.append(meanacc)

    allmeanacc = np.array(allmeanacc)
    mean, std = allmeanacc.mean(), allmeanacc.std(ddof=1)
    print(mean, std)

    conf_intveral = stats.norm.interval(0.95, loc=mean, scale=std)
    print((conf_intveral[0] + conf_intveral[1])*50)
    print((conf_intveral[1] - conf_intveral[0])*50)

    return mean, std, conf_intveral
