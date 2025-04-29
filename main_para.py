import argparse

import os
# import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F


from utils_para import *
# 改动要实验的数据集在这里
from datasets import *
from models import *
# 改动要跑的模型在这里
from models.bi_model_lit import bi_model_lit as bi_model_lit
from models.bi_model_lit_para import bi_model_lit as bi_model_lit_para

def get_dataset(name, data):
    if 'geo' in name or 'Geo' in name or "GASA" in name:
        if data == 'NYU':
            dataset = Geo_NYUDataset
        elif data == 'Middlebury':
            dataset = Geo_MiddleburyDataset
        elif data == 'Lu':
            dataset = Geo_LuDataset
        elif data == 'Luz':
            dataset = Geo_LuzDataset
        else:
            raise NotImplementedError(f'Dataset {data} not found')
    elif 'gidf' in name or 'GIDF' in name:
        if data == 'NYU':
            dataset = LIT_NYUDataset
        elif data == 'Middlebury':
            dataset = LIT_MiddleburyDataset
        elif data == 'Middlebury_noise':
            dataset = LIT_MiddleburyDataset_Noise
        elif data == 'Lu':
            dataset = LIT_LuDataset
        elif data == 'RGBDD':
            dataset = LIT_RGBDDDataset
        else:
            raise NotImplementedError(f'Dataset {data} not found')
    else:
        if data == 'NYU':
            dataset = NYUDataset
        elif data == 'Lu':
            dataset = LuDataset
        elif data == 'Middlebury':
            dataset = MiddleburyDataset
        elif data == 'Luz':
            dataset = LuzDataset
        else:
            raise NotImplementedError(f'Dataset {data} not found')
    return dataset

parser = argparse.ArgumentParser()
parser.add_argument('--para', action='store_true', help='use model para')
parser.add_argument('--para_block', default=0, type=int, help='number of block in cuda:0')
# para config
parser.add_argument('--name', type=str, default='lit_bs1_d2_b3')
parser.add_argument('--epoch', default=200, type=int, help='max epoch')
parser.add_argument('--AR_epoch',  default=200, type=int, help='epochs for the first stage')
parser.add_argument('--train_batch', default=1, type=int)
parser.add_argument('--depth',  default=2, type=int, help='number of SAT layer in each DEF block')
parser.add_argument('--num_bif',  default=3, type=int, help='number of BIF block')
parser.add_argument('--model', default='GIDF', type=str)
parser.add_argument('--noisy',  action='store_true', help='use noisy dataset')
parser.add_argument('--checkpoint',  default='scratch', type=str, help='checkpoint to use')
# Train config
parser.add_argument('--scale',  default=8, type=float, help='eval scale and test scale')
parser.add_argument('--scale_max',  default=16, type=int, help='max scale')
parser.add_argument('--lr',  default=0.0001, type=float, help='learning rate')
parser.add_argument('--lr_step',  default=100, type=float, help='learning rate decay step')
parser.add_argument('--lr_gamma',  default=0.5, type=float, help='learning rate decay gamma')
parser.add_argument('--input_size',  default=256, type=int, help='crop size for hr image')
parser.add_argument('--embed_dim',  default=64, type=int, help='number of channels in the model')
parser.add_argument('--interpolation',  default='bicubic', type=str, help='interpolation method to generate lr depth')
parser.add_argument('--eval_interval',  default=1, type=int, help='eval interval')
parser.add_argument('--sample_q',  default=2304, type=int, help='')
parser.add_argument('--loss', type=str, default='L1')
parser.add_argument('--seed', type=int, default=2341)
parser.add_argument('--dataset', type=str, default='NYU')
parser.add_argument('--data_root', type=str, default="./data/nyu_labeled/")
parser.add_argument('--num_workers', type=int, default=8)
# Test config
parser.add_argument('--test',  action='store_true', help='test mode')
parser.add_argument('--test_batch', default=1, type=int, help='test batch size')
parser.add_argument('--pre_trained_model',  default='', type=str, help='the specific model to load')
parser.add_argument('--report_per_image',  action='store_true', help='report RMSE of each image')
parser.add_argument('--save',  action='store_true', help='save results')
parser.add_argument('--batched_eval',  action='store_true', help='batched evaluation to avoid OOM for large image resolution')
# useless config
parser.add_argument('--test_val',  default=1, type=int, help='jiif test batch size')
args = parser.parse_args()

seed_everything(args.seed)
print("*******************************")
print("*****AR epoch: ", args.AR_epoch, "/",  args.epoch, "*****")
print("*******************************")

# model
if args.model == 'GIDF':
    if not args.para:
        model = bi_model_lit(args, embed_dim=args.embed_dim, 
                            num_bif=args.num_bif, depth=args.depth)
    else:
        model = bi_model_lit_para(args, embed_dim=args.embed_dim, 
                            num_bif=args.num_bif, depth=args.depth)
else:
    raise NotImplementedError(f'Model {args.model} not found')

# loss
if args.loss == 'L1':
    criterion = nn.L1Loss()
elif args.loss == 'L2':
    criterion = nn.MSELoss()
else:
    raise NotImplementedError(f'Loss {args.loss} not found')

# dataset
dataset = get_dataset(args.name, args.dataset)

if args.model in ['gidf', 'GIDF']:
    if not args.test:
        train_dataset = dataset(root=args.data_root, split='train', batch_size=args.train_batch, scale=args.scale, downsample=args.interpolation, augment=True, to_pixel=True, sample_q=args.sample_q, input_size=args.input_size, if_AR=False, noisy=args.noisy)
        train_dataset_vary = dataset(root=args.data_root, split='train', batch_size=args.train_batch, scale=args.scale, scale_max=args.scale_max, downsample=args.interpolation, augment=True, to_pixel=True, sample_q=args.sample_q, input_size=args.input_size, if_AR=True, noisy=args.noisy)
    # eval img size = [256, 256]
    # test_dataset = dataset(root=args.data_root, split='test', batch_size=args.test_batch, scale=args.scale, scale_max=args.scale_max, downsample=args.interpolation, augment=False, to_pixel=True, sample_q=None, input_size=256) 
    # eval img size = Total size
    test_dataset = dataset(root=args.data_root, split='test', batch_size=args.test_batch, scale=args.scale, scale_max=args.scale_max, downsample=args.interpolation, augment=False, to_pixel=True, sample_q=None, noisy=args.noisy)  # full image
else:
    raise NotImplementedError(f'Dataset for model type {args.model} not found')

# dataloader
if not args.test:
    # train_loader_1: lock scale 8
    train_loader_1 = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, pin_memory=True, drop_last=False, shuffle=True, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
    # train_loader_2: random scale in (1, 16]
    train_loader_2 = torch.utils.data.DataLoader(train_dataset_vary, batch_size=args.train_batch, pin_memory=True, drop_last=False, shuffle=True, num_workers=args.num_workers, collate_fn=train_dataset_vary.collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch, pin_memory=True, drop_last=False, shuffle=False, num_workers=args.num_workers, collate_fn=test_dataset.collate_fn)

# trainer
if not args.test:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    trainer = Trainer(args, args.name, model, objective=criterion, optimizer=optimizer, lr_scheduler=scheduler, metrics=[RMSEMeter(args)], device='cuda', use_checkpoint=args.checkpoint, eval_interval=args.eval_interval, pre_trained_model=args.pre_trained_model)
else:
    trainer = Trainer(args, args.name, model, objective=criterion, metrics=[RMSEMeter(args)], device='cuda', use_checkpoint=args.checkpoint, pre_trained_model=args.pre_trained_model)
    # trainer = Trainer(args, args.name, model, objective=criterion, metrics=[RMSEMeter(args)], device='cpu', use_checkpoint=args.checkpoint, pre_trained_model=args.pre_trained_model)

# main
if not args.test:
    trainer.train(train_loader_fix=train_loader_1, train_loader=train_loader_2, valid_loader=test_loader, max_epochs=args.epoch, AR_epoch=args.AR_epoch)

if args.save:
    # save results (doesn't need GT)
    trainer.test(test_loader)
else:
    # evaluate (needs GT)
    trainer.evaluate(test_loader)

if not args.save:
    val_trainer = Trainer(args, args.name, model, objective=criterion, metrics=[RMSEMeter(args)], device='cuda', use_checkpoint="best", pre_trained_model=args.pre_trained_model)
    if not args.test and (args.test_val > 0):
        scale_list = [4, 8, 16]
        data_list = ["NYU","Middlebury","Lu","Luz"]
        root_list = ["./data/nyu_labeled/","./data/depth_enhance/01_Middlebury_Dataset","./data/depth_enhance/03_RGBD_Dataset","./data/depth_enhance/02_RGBZ_Dataset"]
        for num, j in enumerate(data_list):
            dataset = get_dataset(args.name, j)
            rr = root_list[num]
            for i in scale_list:
                val_set = dataset(root=rr, split='test', scale=i, downsample=args.interpolation, augment=False, to_pixel=True, sample_q=None)
                val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.test_batch, pin_memory=False, drop_last=False, shuffle=False, num_workers=args.num_workers)
                print("--------------- scale:", i, "--dataset:", j, "----------------")
                trainer.evaluate(val_loader)
                




        
            

            

