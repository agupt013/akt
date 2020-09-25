import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys, os, math
import argparse
import numpy as np
#from utils import saveCheckpoint
from torch.autograd import Variable
from sklearn import metrics
from tqdm import tqdm
from sklearn.metrics import average_precision_score

def compute_mAP(labels,outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i].reshape((-1)),y_pred[i].reshape((-1))))
    return np.mean(AP)

def cycle(s_loader):
    while True:
        for item in s_loader:
            yield item

def train(epoch, model, g_model, d_model, optimizer, g_optimizer, 
          d_optimizer, t_trainloader, s_trainloader, lmd_s, device):

    #print('Current lr:{}'.format(lr_sch.get_lr()))
    print_f = 1
    mAP = []
    
    lmd_s = 0.001# * (int(epoch//75) + 1 ) #0.15 if epoch < 52 else 0.0#min(int(epoch/5)*0.1,0.9)
    lmd_fc = 1
    
    
    Tensor = torch.cuda.FloatTensor if device == "cuda:0" else torch.FloatTensor
    #input(model.state_dict())
    #Define losses
    criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)
    #criterion = nn.MultiLabelSoftMarginLoss()
    d_criteria = nn.BCEWithLogitsLoss()
    adversarial_loss = nn.BCEWithLogitsLoss().to(device)#reduction='none')


    model.train()    
    g_model.train()
    d_model.train()

    # load M classifier weight in G classifier
    try:
        #pass
        g_model.state_dict()['classifier.fc8.weight'].data = model.state_dict()['classifier.fc8.weight'].data
        g_model.state_dict()['classifier.fc8.bias'].data = model.state_dict()['classifier.fc8.bias'].data
    except:
        #pass
        g_model.state_dict()['module.classifier.fc8.weight'].data = model.state_dict()['module.classifier.fc8.weight'].data
        g_model.state_dict()['module.classifier.fc8.bias'].data = model.state_dict()['module.classifier.fc8.bias'].data
    '''if epoch%5 == 0:
        g_model.train()
        d_model.eval()
    else:
        g_model.eval()
        d_model.train()'''
    
    # initialize variables
    t_total = 0
    t_correct = 0.0    
    s_total = 0.0
    s_correct = 0.0
    running_loss = 0.0
    running_d_loss = 0.0
    running_g_loss = 0.0
    
    # make info displayer
    tqdm_loader = tqdm(range(len(t_trainloader)),ncols=150)

    # create iterative dataloader to use next() on source data
    
    s_trainloader_ = iter(s_trainloader)
    for data in t_trainloader:

        model.train()
        # get the target data 
        t_inputs, t_labels = data
        t_inputs = t_inputs.to(device)
        t_labels = t_labels.to(device)
       
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward target through M
        with torch.no_grad():
            feats = model(t_inputs)#,out_feat_keys)        
            t_feat, t_outputs = feats
            _, t_preds = t_outputs.max(1)
        
        # compute loss
        #loss = criterion(t_outputs, t_labels)
        
        # get the source batch
        try:
            s_inputs, _ = next(s_trainloader_)
            s_inputs = s_inputs.to(device)
        except:
            #s_trainloader = torch.utils.data.DataLoader(s_trainset, batch_size=24,
            #                              shuffle=True, num_workers=4)
            s_trainloader_ = iter(s_trainloader)
            s_inputs, _ = next(s_trainloader_)
            s_inputs = s_inputs.to(device)
        
        g_model.train()
        # forward source through G
        g_feats = g_model(s_inputs)#,out_feat_keys)
        sp_feat, s_labels = g_feats
        _, sp_pseudo = s_labels.max(1)          
         

        # Adversarial Training
        valid = Variable(Tensor(t_feat.size(0), 1).fill_(0.9), requires_grad=False).to(device)
        fake = Variable(Tensor(sp_feat.size(0), 1).fill_(0.1), requires_grad=False).to(device)

        # -----------------
        #  Train Generator
        # -----------------
        #model.eval()
        #d_model.eval()
        # g_model.train()
        
        # zero the parameter gradients
        g_optimizer.zero_grad()        

        # compute generator loss
        valid = Variable(Tensor(sp_feat.size(0), 1).fill_(0.9), requires_grad=False).to(device)
        g_loss = lmd_fc*adversarial_loss(d_model(sp_feat), valid)#.clamp(1e-8,1-1e-7)         
        #g_loss += lmd_conv*adversarial_loss(d_conv_model(sp_conv), valid)
        #g_loss /= 2.0
 
        # backward pass for M
        g_loss.backward()
        #torch.nn.utils.clip_grad_norm_(g_model.parameters(), 10)
        g_optimizer.step()

        running_g_loss += g_loss.item()
        # ----------------- #


        if np.random.rand() < 0.01:
            #print('in')
            valid = Variable(Tensor(t_feat.size(0), 1).fill_(0.1), requires_grad=False).to(device)
            fake = Variable(Tensor(sp_feat.size(0), 1).fill_(0.9), requires_grad=False).to(device)
        else:        
            valid = Variable(Tensor(t_feat.size(0), 1).fill_(0.9), requires_grad=False).to(device)
            fake = Variable(Tensor(sp_feat.size(0), 1).fill_(0.1), requires_grad=False).to(device)
        # ---------------------
        #  Train Discriminator
        # ---------------------
        #model.eval()
        #d_model.train()        
        #g_model.eval()

        # zero the parameter gradients
        d_optimizer.zero_grad()
        
        # compute discriminator loss
        
        real_loss = adversarial_loss(d_model(t_feat.detach()), valid)
        #import pdb; pdb.set_trace()        
        fake_loss = adversarial_loss(d_model(sp_feat[np.random.choice(sp_feat.size(0), t_feat.size(0)),:].detach()), fake[:t_feat.size(0)])
       
        d_loss = (real_loss + fake_loss) / 2.0

        # backward pass for D
        d_loss.backward()
        #torch.nn.utils.clip_grad_norm_(d_model.parameters(), 10)
        d_optimizer.step()

        running_d_loss += d_loss.item()
        # ----------------- #

               
        # ---------------------
        #  Train Model
        # ---------------------
        #model.train()
        #d_model.eval()        
        #g_model.eval()
        with torch.no_grad():
            g_feats = g_model(s_inputs)#,out_feat_keys)
            sp_feat, s_labels = g_feats
            _, sp_pseudo = s_labels.max(1)             


        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward target through M
        feats = model(t_inputs)#,out_feat_keys)        
        t_feat, t_outputs = feats
        _, t_preds = t_outputs.max(1)       
        
        
        # forward source through M
        feats = model(s_inputs)#,out_feat_keys)        
        s_feat, s_outputs = feats
        _, s_preds = s_outputs.max(1)

        # compute loss
        #loss = lmd_s * adversarial_loss(d_conv_model(t_conv), valid)

        #t_outputs = nn.LogSigmoid()(t_outputs)
      
        t_labels = t_labels.float().cuda()
        
        mask = (t_labels == 255)        
        loss = torch.sum(criterion(t_outputs, t_labels).masked_fill_(mask, 0)) / t_labels.size(0)

        try:
            mAP.append(compute_mAP(t_labels.data,t_outputs.data))
        except:
            #print(t_labels)
            #input('t_labels : NaN value Occurred')
            #print(t_outputs)
            #input('t_outputs : NaN value Occurred')
            tqdm_loader.close()
            print('Training diverged (model). Resetting the weights to previous stable state.')
            
            return False

        #s_outputs = nn.LogSigmoid()(s_outputs)
        s_labels = s_labels.float().cuda()
        #mask = (s_labels == 255)
        #loss_s = lmd_s * torch.sum(criterion(s_outputs, s_labels.detach()).masked_fill_(mask, 0)) / s_labels.size(0)
        loss_s = lmd_s * torch.sum(criterion(s_outputs, nn.Softmax()(s_labels).detach())) / s_labels.size(0)
      
        '''if loss_s < 0.0:
            loss_s = 0.0'''
        loss += loss_s
        
        #loss = criterion(t_outputs, t_labels)
        #loss += lmd_s * criterion(s_outputs, s_labels.detach())
        '''if loss < 0.001:
            tqdm_loader.close()
            print('Training diverged (loss). Resetting the weights to previous stable state.')
            return False'''
        # backward pass for M
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        # statistics
        running_loss += loss.item()
        #t_correct += t_preds.eq(t_labels).sum().item()
        t_total += t_labels.size(0)

        # set display info
        tqdm_loader.set_description('\r[Training] [Ep %3d] loss: %.3f (%.3f) d_loss: %.5f g_loss: %.5f mAP: %.5f'%
                   (epoch + 1, running_loss / t_total,loss_s, running_d_loss / t_total,
                   running_g_loss / t_total,100*np.mean(mAP[-20:]))
                                    )
               
        tqdm_loader.update(print_f)
    
    # close loader to avoid irregularity in display
    tqdm_loader.close()

    # update lr scheduler
    #lr_sch.step()
    return True

def test(epoch,model, g_model, d_model, optimizer, g_optimizer, 
          d_optimizer,t_testloader,best_mAP,best_path,device):
    mAP = []
    model.eval()

    tqdm_test = tqdm(range(len(t_testloader)),ncols=150)
    tqdm_test.refresh()

    
    with torch.no_grad():
        for (images, labels) in t_testloader:
            images = images.view((-1,3,227,227))
            #images = Variable(images, volatile=True)
            if device is not None:
                images = images.cuda()
        
            # Forward + Backward + Optimize
            _,outputs = model(images)
            outputs = outputs.float()
            outputs = outputs.cpu().data
            outputs = outputs.view((-1,10,21))
            outputs = outputs.mean(dim=1).view((-1,21))
            
            #score = tnt.meter.mAPMeter(outputs, labels)
            mAP.append(compute_mAP(labels,outputs))
            tqdm_test.set_description('val mAP: %.3f %% |' % (100*np.mean(mAP[-20:])))
            tqdm_test.update(1)

                
    # close loader to avoid irregularity in display
    tqdm_test.close()    

    epoch_mAP = np.mean(mAP[-20:])
    best_mAP = saveCheckpoint(epoch, model, g_model, d_model, optimizer, g_optimizer, 
          d_optimizer,epoch_mAP,best_mAP,best_path)

    model.train()
    return best_mAP

def saveCheckpoint(epoch,model, g_model, d_model, optimizer, g_optimizer, 
          d_optimizer, epoch_acc,best_acc,best_path):
    if epoch_acc > best_acc:        
        print('Saving Best model...\tTop1: %.2f%%' %(100.*epoch_acc))
        
        if not os.path.isdir('./checkpoint'):
            os.mkdir('./checkpoint')
        if os.path.exists(best_path.format(best_acc)):
            os.remove(best_path.format(best_acc))
        state = {'epoch': epoch,'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'g_model': g_model.state_dict(), 'g_optimizer': g_optimizer.state_dict(),
                 'd_model': d_model.state_dict(), 'd_optimizer': d_optimizer.state_dict()}
        torch.save(state,best_path.format(epoch_acc))
        best_acc = epoch_acc
        
    else:
        print('Best Model \tTop1: %.2f%%' %(100.*best_acc))
        
    return best_acc
