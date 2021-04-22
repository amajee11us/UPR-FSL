import _init_paths
import argparse
import yaml
import os.path as osp
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.mini_imageNet import MiniImageNet
from utils.samplers import CategoriesSampler
from model.convnet import Convnet
from utils.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, copyModel, cosine

from utils.collections import AttrDict

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default="exps/exp-v1/config.yaml")


    args = parser.parse_args()
    pprint(vars(args))

    with open(args.config, 'r') as f:
        cfg = AttrDict(yaml.load(f))
        cfg = AttrDict(cfg.train)


    set_gpu(cfg.gpu)
    ensure_path(cfg.save_path)

    trainset = MiniImageNet(cfg.datapath, 'train')
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      cfg.train_way, cfg.shot + cfg.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=16, pin_memory=True)

    valset = MiniImageNet(cfg.datapath, 'val')
    val_sampler = CategoriesSampler(valset.label, 400,
                                    cfg.test_way, cfg.shot + cfg.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=16, pin_memory=True)

    propagate_loader = DataLoader(dataset=trainset, batch_size=1280, shuffle=True, num_workers=24, pin_memory=True)
    


    if not cfg.pretrain:
        model = nn.DataParallel(Convnet()).cuda()
    else:
        model = copyModel(torch.load(cfg.pretrainpath), Convnet()).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        torch.save(model.module.state_dict(), osp.join(cfg.save_path, name + '.pth'))
    trlog = {}
    trlog['args'] = vars(args)
    trlog['cfg'] = vars(cfg)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    assert(cfg.progalambda >= 0), "progalambda must be no-negative."

    if cfg.progalambda > 0:
        allExtraData = []
        with torch.no_grad():
            for i, batch in enumerate(propagate_loader, 1):
                ext_data, _ = [_.cuda() for _ in batch]
                allExtraData.append(ext_data)
        allExtraData = torch.cat(allExtraData)

    timer = Timer()

    for epoch in range(1, cfg.max_epoch + 1):
        # torch.cuda.empty_cache()
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            # time.sleep(100)
            torch.cuda.empty_cache()
            data, _ = [_.cuda() for _ in batch]
            p = cfg.shot * cfg.train_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(cfg.shot, cfg.train_way, -1).mean(dim=0)

            query_proto = model(data_query)

            p = (1 - cfg.progalambda) * proto
            # query_proto = query_proto.reshape(args.query, args.train_way, -1)

            if cfg.progalambda > 0:
                with torch.no_grad():
                    # select randomly data to avoid out of memory.
                    allExtraproto = []
                    index = torch.randperm(allExtraData.shape[0])[:int(allExtraData.shape[0]/10)]
                    extraDatatemp = allExtraData[index]
                    # compute feature
                    n = 500
                    for data in [extraDatatemp[i:i + n] for i in range(0, len(extraDatatemp), n)]:
                        allExtraproto.append(model(data))
                    allExtraproto = torch.cat(allExtraproto)
                    # torch.cuda.empty_cache()

                    cossim = cosine(proto, allExtraproto)
                    cossim = cossim.reshape(cfg.train_way, -1)

                topsim, topkindex = torch.topk(cossim, cfg.topk, dim=1, largest=True, sorted=True)

                p = p + cfg.progalambda * (allExtraproto[topkindex] * topsim.unsqueeze(2).expand(cfg.train_way, cfg.topk, 1600)).sum(dim=1) / cfg.topk


            label = torch.arange(cfg.train_way).repeat(cfg.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            p = None; proto = None; logits = None; loss = None


        tl = tl.item()
        ta = ta.item()

        torch.cuda.empty_cache()
        model.eval()

        vl = Averager()
        va = Averager()

        if cfg.progalambda > 0:
            with torch.no_grad():
                n = 500
                allExtraproto = []
                index = torch.randperm(allExtraData.shape[0])[:int(allExtraData.shape[0]/10)]
                extraDatatemp = allExtraData[index]
                for data in [extraDatatemp[i:i + n] for i in range(0, len(extraDatatemp), n)]:
                    allExtraproto.append(model(data))
                allExtraproto = torch.cat(allExtraproto)

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = cfg.shot * cfg.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(cfg.shot, cfg.test_way, -1).mean(dim=0)

            query_proto = model(data_query)
            # query_proto = query_proto.reshape(args.query, args.test_way, -1)
            p = (1 - cfg.progalambda) * proto

            if cfg.progalambda > 0:
                cossim = cosine(proto, allExtraproto)
                cossim = cossim.reshape(cfg.test_way, -1)

                topsim, topkindex = torch.topk(cossim, cfg.topk, dim=1, largest=True, sorted=True)

                p = p + cfg.progalambda * (allExtraproto[topkindex] * topsim.unsqueeze(2).expand(cfg.test_way, cfg.topk, 1600)).sum(dim=1)/cfg.topk


            label = torch.arange(cfg.test_way).repeat(cfg.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)
            
            p = None; proto = None; logits = None; loss = None


        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(cfg.save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % cfg.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / cfg.max_epoch)))

