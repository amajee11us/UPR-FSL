import _init_paths
import yaml
import argparse

import torch
from torch.utils.data import DataLoader

from datasets.mini_imageNet import MiniImageNet
from utils.samplers import CategoriesSampler
from model.convnet import Convnet
from utils.utils import pprint, set_gpu, count_acc, Averager, euclidean_metric, cosine, copyModel, CI, ensure_path

from utils.collections import AttrDict

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default="exps/exp-v1/config.yaml")

    args = parser.parse_args()
    pprint(vars(args))

    with open(args.config, 'r') as f:
        cfg = AttrDict(yaml.load(f))
        cfg = AttrDict(cfg.test)

    set_gpu(cfg.gpu)
    ensure_path(cfg.result)

    dataset = MiniImageNet(cfg.datapath, 'test')
    sampler = CategoriesSampler(dataset.label,
                                cfg.batch, cfg.way, cfg.shot + cfg.query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=8, pin_memory=True)

    trainset = MiniImageNet(cfg.datapath, 'train')
    propagate_loader = DataLoader(dataset=trainset, batch_size=1280, shuffle=True, num_workers=24, pin_memory=True)

    premodel = torch.load(cfg.load)
    # model = Convnet()

    model = copyModel(torch.load(cfg.load), Convnet()).cuda()
    # model = copyModel(Convnet(), torch.load(cfg.oad)).cuda()
    model.eval()

    allacc = []

    ave_acc = Averager()

    allExtraData = []
    with torch.no_grad():
        for i, batch in enumerate(propagate_loader, 1):
            ext_data, _ = [_.cuda() for _ in batch]
            allExtraData.append(ext_data)
    allExtraData = torch.cat(allExtraData)


    if cfg.progalambda > 0:
        with torch.no_grad():
            n = 1280
            allExtraproto = []
            index = torch.randperm(allExtraData.shape[0])[:int(allExtraData.shape[0]/10)]
            extraDatatemp = allExtraData[index]
            for data in [extraDatatemp[i:i + n] for i in range(0, len(extraDatatemp), n)]:
                allExtraproto.append(model(data))
            allExtraproto = torch.cat(allExtraproto)

    for i, batch in enumerate(loader, 1):
        data, _ = [_.cuda() for _ in batch]
        k = cfg.way * cfg.shot
        data_shot, data_query = data[:k], data[k:]

        x = model(data_shot)
        x = x.reshape(cfg.shot, cfg.way, -1).mean(dim=0)
        proto = x

        if cfg.progalambda > 0:
            cossim = cosine(proto, allExtraproto)
            cossim = cossim.reshape(cfg.way, -1)

            topsim, topkindex = torch.topk(cossim, cfg.topk, dim=1, largest=True, sorted=True)

            p = cfg.progalambda * (allExtraproto[topkindex] * topsim.unsqueeze(2).expand(cfg.way, cfg.topk, 1600)).sum(dim=1)/cfg.topk \
                    + (1 - cfg.progalambda) * proto
        else:
            p = proto
        
        logits = euclidean_metric(model(data_query), p)

        label = torch.arange(cfg.way).repeat(cfg.query)
        label = label.type(torch.cuda.LongTensor)

        acc = count_acc(logits, label)
        ave_acc.add(acc)
        print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

        allacc.append(acc)
        
        x = None; p = None; logits = None

    allacc = np.array(allacc)
    torch.save(allacc, cfg.result + '/allacc')

    mean, std, conf_intveral = CI(allacc)

    result = "mean: " + str(mean) + "\nstd: " + str(std) + "\nconfidence intveral: [" + str(conf_intveral[0]) + " : " + str(conf_intveral[1]) + "]"

    with open(cfg.result + '/acc.txt', 'w') as f:
        f.write(result)