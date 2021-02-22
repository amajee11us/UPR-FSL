import _init_paths
import argparse
import yaml
import os.path as osp

import torch
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

    trainset = MiniImageNet('train')
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      cfg.train_way, cfg.shot + cfg.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=16, pin_memory=True)

    valset = MiniImageNet('val')
    val_sampler = CategoriesSampler(valset.label, 400,
                                    cfg.test_way, cfg.shot + cfg.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=16, pin_memory=True)

    propagate_loader = DataLoader(dataset=trainset, batch_size=1280, shuffle=True, num_workers=24, pin_memory=True)

    model = copyModel(torch.load(cfg.pretrain), Convnet()).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(cfg.save_path, name + '.pth'))
    trlog = {}
    trlog['args'] = vars(args)
    trlog['cfg'] = vars(cfg)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    allExtraData = []
    with torch.no_grad():
        for i, batch in enumerate(propagate_loader, 1):
            ext_data, _ = [_.cuda() for _ in batch]
            allExtraData.append(ext_data)
    allExtraData = torch.cat(allExtraData)

    timer = Timer()

    for epoch in range(1, cfg.max_epoch + 1):
        torch.cuda.empty_cache()
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            torch.cuda.empty_cache()
            data, _ = [_.cuda() for _ in batch]
            p = cfg.shot * cfg.train_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(cfg.shot, cfg.train_way, -1).mean(dim=0)

            query_proto = model(data_query)

            with torch.no_grad():
                n = 2000
                allExtraproto = []
                for data in [allExtraData[i:i + n] for i in range(0, len(allExtraData), n)]:
                    allExtraproto.append(model(data))
                allExtraproto = torch.cat(allExtraproto)
                torch.cuda.empty_cache()
            
                proto1, proto2, proto3 = proto[:10], proto[10: 20], proto[20:]

                cos1 = cosine(proto1, allExtraproto)
                cos2 = cosine(proto2, allExtraproto)
                cos3 = cosine(proto3, allExtraproto)

                cossim = torch.cat((cos1, cos2, cos3), dim=0)
                cossim = cossim.reshape(cfg.train_way, -1)

            topsim, topkindex = torch.topk(cossim, cfg.topk, dim=1, largest=True, sorted=True)

            proto = cfg.progalambda * (allExtraproto[topkindex] * topsim.unsqueeze(2).expand(cfg.train_way, cfg.topk, 1600)).sum(dim=1)/cfg.topk \
                 + (1 - cfg.progalambda) * proto

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
            
            proto = None; logits = None; loss = None

        tl = tl.item()
        ta = ta.item()

        torch.cuda.empty_cache()
        model.eval()

        vl = Averager()
        va = Averager()

        with torch.no_grad():
            n = 2000
            allExtraproto = []
            for data in [allExtraData[i:i + n] for i in range(0, len(allExtraData), n)]:
                allExtraproto.append(model(data))
            allExtraproto = torch.cat(allExtraproto)

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = cfg.shot * cfg.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(cfg.shot, cfg.test_way, -1).mean(dim=0)

            query_proto = model(data_query)

            proto1, proto2, proto3 = proto[:10], proto[10:20], proto[20:]
            cos1 = cosine(proto1, allExtraproto)
            cos2 = cosine(proto2, allExtraproto)
            cos3 = cosine(proto3, allExtraproto)

            cossim = torch.cat((cos1, cos2, cos3), dim=0)
            cossim = cossim.reshape(cfg.test_way, -1)

            topsim, topkindex = torch.topk(cossim, cfg.topk, dim=1, largest=True, sorted=True)

            proto = cfg.progalambda * (allExtraproto[topkindex] * topsim.unsqueeze(2).expand(cfg.test_way, cfg.topk, 1600)).sum(dim=1)/cfg.topk \
            + (1 - cfg.progalambda) * proto

            label = torch.arange(cfg.test_way).repeat(cfg.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)
            
            proto = None; logits = None; loss = None

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