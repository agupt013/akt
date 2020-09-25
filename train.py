import os, sys, numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
#import torchnet as tnt

import multiprocessing
CORES = 2

from loader  import DataLoader
from models.alexnet import Network
from trainer import train, test
from models.disnet import disnet

from Utils.TrainingUtils import adjust_learning_rate


def compute_mAP(labels,outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i],y_pred[i]))
    return np.mean(AP)

def main(args):
    if args.gpu is not None:
        print('Using GPU %d'%args.gpu)
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    else:
        print('CPU mode')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std= [0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    
    val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(227),
            #transforms.RandomResizedCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    # DataLoader initialize
    train_data   = DataLoader(args.pascal_path,'trainval',transform=train_transform)
    t_trainloader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=args.batch, 
                                               shuffle=True,
                                               num_workers=CORES,
                                               pin_memory=True)
    print('[DATA] Target Train loader done!')
    val_data   = DataLoader(args.pascal_path,'test',transform=val_transform,random_crops=args.crops)
    t_testloader = torch.utils.data.DataLoader(dataset=val_data,
                                             batch_size=args.batch, 
                                             shuffle=False,
                                             num_workers=CORES,
                                             pin_memory=True)
    print('[DATA] Target Test loader done!')

    if not args.test :
        s_trainset = torchvision.datasets.ImageFolder(args.imgnet_path, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(227),
            transforms.ToTensor(), normalize]
         ))
        s_trainloader = torch.utils.data.DataLoader(dataset= s_trainset,
                                             batch_size=5*args.batch, 
                                             shuffle=False,
                                             num_workers=CORES,
                                             pin_memory=True)
        print('[DATA] Source Train loader done!')
   
    N = len(train_data.names)
    iter_per_epoch = N/args.batch
  
    model = Network(num_classes = 21)
    g_model = Network(num_classes = 21)
    d_model = disnet()

    if args.gpu is not None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('[MODEL] CUDA DEVICE : {}'.format(device))   
        
        model.to(device)
        g_model.to(device)
        d_model.to(device)    
    

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr,momentum=0.9,weight_decay = 0.0001)
    g_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, g_model.parameters()),
                                lr=args.lr,momentum=0.9,weight_decay = 0.0001)
    d_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, d_model.parameters()),
                                lr=args.lr,momentum=0.9,weight_decay = 0.0001)
    
    if args.model is not None:
        checkpoint = torch.load(args.model)        
        model.load(checkpoint['model'],True)
        g_model.load(checkpoint['g_model'],True)
        d_model.load_state_dict(checkpoint['d_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])



    
    ############## TRAINING ###############
    print('Start training: lr %f, batch size %d'%(args.lr,args.batch))
    print('Checkpoint: '+args.checkpoint)
    
    # Train the Model
    steps = args.iter_start
    best_mAP = 0.0
    best_path = './{}/model-{}_pretrained-{}_lr-0pt001_lmd_s-{}_acc-{}.pth'.format(args.checkpoint,'alexnet','False',args.lmd_s,'{}')

    if args.test:
        args.epochs = 1

    for epoch in range(int(iter_per_epoch*args.iter_start),args.epochs):       
        if not args.test:
            adjust_learning_rate(optimizer, epoch, init_lr=args.lr, step=100, decay=0.1)
            adjust_learning_rate(g_optimizer, epoch, init_lr=args.lr/2, step=100, decay=0.1)
            adjust_learning_rate(d_optimizer, epoch, init_lr=args.lr/1.5, step=100, decay=0.1)
        
       
            done = train(epoch, model, g_model, d_model, optimizer, g_optimizer, 
             d_optimizer, t_trainloader, s_trainloader, args.lmd_s, device)        
        
        best_mAP = test(epoch, model, g_model, d_model, optimizer, g_optimizer, 
         d_optimizer, t_testloader,best_mAP,best_path, device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train network on Pascal VOC 2007')
    parser.add_argument('--pascal_path', type=str, help='Path to Pascal VOC 2007 folder')
    parser.add_argument('--imgnet_path', type=str, help='Path to ImageNet folder')
    parser.add_argument('--model', default=None, type=str, help='Pretrained model')    
    parser.add_argument('--test', default=None, type=int, help='test model')    
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--epochs', default=350, type=int, help='gpu id')
    parser.add_argument('--iter_start', default=0, type=int, help='Starting iteration count')
    parser.add_argument('--lmd_s', default=0.001, type=float, help='source lambda')
    parser.add_argument('--batch', default=20, type=int, help='batch size')
    parser.add_argument('--checkpoint', default='checkpoints/', type=str, help='checkpoint folder')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate for SGD optimizer')
    parser.add_argument('--crops', default=10, type=int, help='number of random crops during testing')
    args = parser.parse_args()

    if args.test and args.model == None:
        sys.exit('You must provide model when trying to test.')
    main(args)

