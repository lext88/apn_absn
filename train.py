import os
import tqdm
import time
import wandb
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import compute_accuracy, set_seed, restart_from_checkpoint
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.dataloader.aux_dataloader import get_aux_dataloader
from models.renet import DCANet

from test import test_main, evaluate
from utils import rotrate_concat, record_data
from common.utils import pprint, ensure_path, set_gpu
from loss import AdaptivePrototypicalLoss  # Import the new loss class

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ["CUDA_LAUNCH_BLOCKING"] = '2'

def parse_args():
    parser = argparse.ArgumentParser(description='train')

    ''' about dataset '''
    parser.add_argument('-dataset', type=str, default='miniImageNet',
                        choices=['miniImageNet', 'cub', 'tieredImageNet', 'CIFAR-FS', 'FC100'])
    parser.add_argument('-data_root', type=str, default='/home/lxj/new_main/dataset', help='dir of datasets')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')

    ''' about training specs '''
    parser.add_argument('-batch', type=int, default=64, help='auxiliary batch size')
    parser.add_argument('-temperature', type=float, default=0.2, metavar='tau', help='temperature for metric-based loss')
    parser.add_argument('-lamb', type=float, default=0.25, metavar='lambda', help='loss balancing term')
    parser.add_argument('--w_d', type=float, default=0.01, help='weight of distance loss')
    parser.add_argument('--w_p', type=float, default=0.5)

    ''' about training schedules '''
    parser.add_argument('-max_epoch', type=int, default=80, help='max epoch to run')
    parser.add_argument('-lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('-gamma', type=float, default=0.05, help='learning rate decay factor')
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70], help='milestones for MultiStepLR')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('-save_all', action='store_true', help='save models on each epoch')
    parser.add_argument('-use_resume', action='store_true', help='use the result of training before')
    parser.add_argument('--resume_file', type=str, default='epoch_10.pth')

    ''' about few-shot episodes '''
    parser.add_argument('-way', type=int, default=5, metavar='N', help='number of few-shot classes')
    parser.add_argument('-shot', type=int, default=1, metavar='K', help='number of shots')
    parser.add_argument('-query', type=int, default=15, help='number of query image per class')
    parser.add_argument('-val_episode', type=int, default=200, help='number of validation episode')
    parser.add_argument('-test_episode', type=int, default=2000, help='number of testing episodes after training')
    parser.add_argument('-proto_size', type=int, default=100, help='the number of dynamic prototypes')
    parser.add_argument('--crop_size', type=int, default=84)
    parser.add_argument('--trans', type=int, default=4, help='number of transformations')
    parser.add_argument('--hidden_size', type=int, default=320, help='hidden size for cross attention layer')
    parser.add_argument('--feat_dim', type=int, default=640)
    parser.add_argument('--sup_t', type=float, default=0.2)

    ''' about CoDA '''
    parser.add_argument('-temperature_attn', type=float, default=2, metavar='gamma', help='temperature for softmax in computing cross-attention')

    ''' about env '''
    parser.add_argument('-gpu', default='2', help='the GPU ids e.g. \"0\", \"0,1\", \"0,1,2\", etc')
    parser.add_argument('-test_tag', type=str, default='test_wp0.5_wd0.01_lam0.25_t2', help='extra dir name added to checkpoint dir')
    parser.add_argument('-seed', type=int, default=1, help='random seed')
    parser.add_argument('-wandb', action='store_true', help='not plotting learning curve on wandb',
                        )  # train: enable logging / test: disable logging
    args = parser.parse_args()
    pprint(vars(args))

    torch.set_printoptions(linewidth=100)
    args.num_gpu = set_gpu(args)
    args.device_ids = None if args.gpu == '-1' else list(range(args.num_gpu))
    args.save_path = os.path.join(f'checkpoints/{args.dataset}/{args.shot}shot-{args.way}way/', args.test_tag)
    ensure_path(args.save_path)
    if not args.wandb:
        wandb.init(project=f'renet-{args.dataset}-{args.way}w{args.shot}s',
                   config=args,
                   save_code=True,
                   name=args.test_tag)

    if args.dataset == 'miniImageNet':
        args.num_class = 64
    elif args.dataset == 'cub':
        args.num_class = 100
    elif args.dataset == 'FC100':
        args.num_class = 60
    elif args.dataset == 'tieredImageNet':
        args.num_class = 351
    elif args.dataset == 'CIFAR-FS':
        args.num_class = 64
        args.crop_size = 42
    elif args.dataset == 'cars':
        args.num_class = 130
    elif args.dataset == 'dogs':
        args.num_class = 70

    return args

def train(epoch, model, loader, optimizer, criterion, args=None):
    model.train()

    train_loader = loader['train_loader']
    train_loader_aux = loader['train_loader_aux']

    # label for query set, always in the same pattern
    query_label = torch.arange(args.way).repeat(args.query).cuda()  # 012340123401234...

    loss_meter = Meter()
    acc_meter = Meter()

    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(train_loader)

    for i, ((data, train_labels), (data_aux, train_labels_aux)) in enumerate(zip(tqdm_gen, train_loader_aux), 1):

        data, train_labels = data.cuda(), train_labels.cuda()
        data_aux = data_aux.cuda()
        batch_size = data_aux.size(0)
        data_aux = rotrate_concat([data_aux])
        train_labels_aux = train_labels_aux.repeat(args.trans).cuda()

        # Forward images (3, 84, 84) -> (C, H, W)
        model.module.mode = 'encoder'
        data, fea_loss, cst_loss, dis_loss = model(data)
        data_aux = model(data_aux, aux=True)  # I prefer to separate feed-forwarding data and data_aux due to BN

        # loss for batch
        model.module.mode = 'coda'
        data_shot, data_query = data[:k], data[k:]
        logits, absolute_logits = model((data_shot.unsqueeze(0).repeat(1, 1, 1, 1, 1), data_query))

        # Compute the adaptive prototypical loss
        features = model.get_features(data_shot, data_query)  # Assuming a method to extract features
        ce_loss = criterion.cross_entropy_loss(logits, query_label, attention_weights=None)
        sup_clu_loss = criterion.sup_clu_loss(features, labels=None, mask=None, attention_weights=None)
        loss = ce_loss + sup_clu_loss

        # loss for auxiliary batch
        model.module.mode = 'fc'
        logits_global, logits_eq = model(data_aux)
        loss_aux = F.cross_entropy(logits_global, train_labels_aux)

        proxy_labels = torch.zeros(args.trans * batch_size).cuda().long()
        for ii in range(args.trans):
            proxy_labels[ii * batch_size:(ii + 1) * batch_size] = ii
        loss_eq = F.cross_entropy(logits_eq, proxy_labels)

        l_re = fea_loss + dis_loss * args.w_d
        loss_aux = absolute_loss + loss_aux
        loss = args.lamb * (epi_loss) + loss_aux + loss_eq + l_re

        acc = compute_accuracy(logits, query_label)
        loss_meter.update(loss.item())
        acc_meter.update(acc)
        tqdm_gen.set_description(f'[train] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        optimizer.zero_grad()

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence
