## --------------------------------------------------------------------
##         B0    B1      B2      B3      B4      B5      B6      B7
## ====================================================================
## Input   224   240     260     300     380     456     528     600
## output  1280  1280    1408    1536    1792    2048    2304    2560
## --------------------------------------------------------------------
##

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
import random

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from efficientnet_pytorch import EfficientNet

from dataset import CustomImageFolder
from torchvision import transforms, datasets
from torchvision.utils import save_image

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import yaml
import csv
from datetime import datetime

from pathfilemgr import MPathFileManager
from hyp_data import MHyp, MData

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--volume', help='volume directory', default='moai')
parser.add_argument('--project', help='project directory', default='test_project')
parser.add_argument('--subproject', help='subproject directory', default='test_subproject')
parser.add_argument('--task', help='task directory', default='test_task')
parser.add_argument('--version', help='version', default='v1')

opt = parser.parse_args()

##########################################################################################
### make dir and hyp
##########################################################################################

mpfm = MPathFileManager(opt.volume, opt.project, opt.subproject, opt.task, opt.version)
mhyp = MHyp()
mdata = MData()
mpfm.load_train_hyp(mhyp)
mpfm.load_train_data(mdata)

##########################################################################################
### make results.csv header
##########################################################################################

result_file = open(mpfm.result_csv, mode='a', newline='', encoding='utf-8')
result_csv = csv.writer(result_file)
result_csv.writerow(["epoch", "accuracy", "loss", "time"])

##########################################################################################
### load data
##########################################################################################

class_names_list = [mdata.names[i] for i in sorted(mdata.names.keys())]

print(class_names_list)

# fc 제외하고 freeze
# for n, p in model.named_parameters():
#     if '_fc' not in n:
#         p.requires_grad = False
# model = torch.nn.parallel.DistributedDataParallel(model)

#########################################################################################################
## parameters for dataloader
#########################################################################################################
batch_size  = mhyp.batch_size
random_seed = 555
random.seed(random_seed)
torch.manual_seed(random_seed)

#########################################################################################################
## train, valid dataset
#########################################################################################################

class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names_list)}

datasets_dict = {}

datasets_dict['train'] = CustomImageFolder(
                                mpfm.train_path,
                                transforms.Compose([
                                    transforms.Resize((mhyp.img_size, mhyp.img_size)),
                                    transforms.ToTensor(),
                                    transforms.ColorJitter(brightness=mhyp.brightness, contrast=mhyp.contrast, saturation=mhyp.saturation, hue=mhyp.hue),
                                    transforms.RandomHorizontalFlip(p=mhyp.hflip),
                                    transforms.RandomVerticalFlip(p=mhyp.vflip),
                                    transforms.RandomRotation(degrees=mhyp.rotate),
                                    # transforms.RandomResizedCrop(size=512, scale=(0.08, 1.0), ratio=(0.75, 1.33), interpolation=2),
                                    # transforms.RandomErasing(),
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ]),class_to_idx=class_to_idx)

datasets_dict['valid'] = CustomImageFolder(
                                mpfm.val_path,
                                transforms.Compose([
                                    transforms.Resize((mhyp.img_size, mhyp.img_size)),
                                    transforms.ToTensor(),
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    transforms.ColorJitter(brightness=mhyp.brightness, contrast=mhyp.contrast, saturation=mhyp.saturation, hue=mhyp.hue),
                                    transforms.RandomHorizontalFlip(p=mhyp.hflip),
                                    transforms.RandomVerticalFlip(p=mhyp.vflip),
                                    transforms.RandomRotation(degrees=mhyp.rotate),
                                ]),class_to_idx=class_to_idx)


#########################################################################################################
## define data loader
#########################################################################################################
dataloaders, batch_num = {}, {}
dataloaders['train'] = torch.utils.data.DataLoader(datasets_dict['train'],
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=0)
dataloaders['valid'] = torch.utils.data.DataLoader(datasets_dict['valid'],
                                              batch_size=1, shuffle=False,
                                              num_workers=0)

batch_num['train'], batch_num['valid'] = len(dataloaders['train']), len(dataloaders['valid'])
print('batch_size : %d,  tvt : %d / %d' % (batch_size, batch_num['train'], batch_num['valid']))

# for i, (inputs, labels) in enumerate(dataloaders['train']):
#     sample_fname, class_idx = dataloaders['train'].dataset.samples[i]
#     print(sample_fname, class_idx)

#########################################################################################################
## model
#########################################################################################################
device = torch.device(f"cuda:{mhyp.gpu}" if torch.cuda.is_available() else "cpu")  # set gpu

model_name = f'efficientnet-b{mhyp.model}'
pretrained_model = f'./pretrained_weights/{model_name}.pth'
# image_size = EfficientNet.get_image_size(model_name)
# print('model input size: ', image_size)
# print(model_name)
model = EfficientNet.from_pretrained(model_name, weights_path=pretrained_model, num_classes=mhyp.num_class)
model = model.to(device)

#########################################################################################################
## Optimizer
#########################################################################################################
epoch = mhyp.epoch

if mhyp.loss == 'ce':
    criterion = nn.CrossEntropyLoss()
    
elif mhyp.loss == 'msm':
    criterion = nn.MultiLabelSoftMarginLoss()

if mhyp.optim == 'sgd':
    optimizer_ft = optim.SGD(model.parameters(), 
                             lr = mhyp.lr,
                             momentum=0.9,
                             weight_decay=1e-4)
elif mhyp.optim == 'adam':
    optimizer_ft = optim.Adam(model.parameters(), lr = mhyp.learning_rate, weight_decay=mhyp.weight_decay)

lmbda = lambda epoch: mhyp.lr_lambda
exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lmbda)

##########################################################################################
### tensorboard
##########################################################################################

twriter = SummaryWriter(mpfm.train_result)

##########################################################################################
### save arguments
##########################################################################################

mpfm.save_hyp(mhyp)
mpfm.save_data(mdata)

##########################################################################################
### train
##########################################################################################
softmax = nn.Softmax(dim=1)
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss, running_corrects, num_cnt = 0.0, 0, 0
            

            with tqdm(dataloaders[phase], unit="batch") as tepoch:
                tepoch.set_description(f"[{phase}] Epoch {epoch}")

                # Iterate over data.
                for inputs, labels in tepoch:
                    ## RGB to BGR
                    # inputs = inputs[:,[2,1,0],:]
                    
                    inputs = inputs.to(device)
                    # labels = labels.type(torch.FloatTensor)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        
                        # if opt.nc == 2:
                        # outputs = softmax(outputs)

                        _, preds = torch.max(outputs, 1)
                        
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    num_cnt += len(labels)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = float(running_loss / num_cnt)
                epoch_acc  = float((running_corrects.double() / num_cnt).cpu()*100)
                
                tepoch.set_postfix(loss=loss.item(), accuracy=epoch_acc)

                if phase == 'train':
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_acc)

                    twriter.add_scalar("train/loss", epoch_loss, epoch)
                    twriter.add_scalar("train/acc", epoch_acc, epoch)

                else:
                    valid_loss.append(epoch_loss)
                    valid_acc.append(epoch_acc)

                    twriter.add_scalar("val/loss", epoch_loss, epoch)
                    twriter.add_scalar("val/acc", epoch_acc, epoch)

                    ctime = datetime.now().strftime("%H:%M:%S")
                    
                    result_csv.writerow([epoch, epoch_acc, epoch_loss, ctime])
                    result_file.flush()

                print('{} Loss: {:.4f} Acc: {:.2f}'.format(phase, epoch_loss, epoch_acc))
               
                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_idx = epoch
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), f'{mpfm.weight_path}/best.pt')
                    print('==> best model saved - %d / %.1f'%(best_idx, best_acc))

    result_file.close()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc (Epoch %d): %.1f' %(best_idx, best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), f'{mpfm.weight_path}/last.pt')
    print('model saved')


    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc

if __name__ == "__main__":

    model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = train_model(
        model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epoch)


    

