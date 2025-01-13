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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='model number 0 - 7', default='5')
parser.add_argument('--epoch', help='epoch', type=int, default=100)
parser.add_argument('--train_path', help='train path', type=str, default='./dataset/train')
parser.add_argument('--val_path', help='val path', type=str, default='./dataset/valid')
parser.add_argument('--class_names_list', help='class name dictionary', nargs='*', default=['OK', 'NG'])
parser.add_argument('--bs', help='batch size', type=int, default=8)
parser.add_argument('--nc', help='number of classes', type=int, default=1)
parser.add_argument('--lr', help='learning rate', type=float, default=0.01)
parser.add_argument('--wd', help='weight decay', type=float, default=1e-5)
parser.add_argument('--img_size', help='image size', type=int, default=200)
parser.add_argument('--vrate', help='validation rate', type=float, default=0.1)
parser.add_argument('--optim', help='adam | sgd', type=str, default='adam')
parser.add_argument('--gpu', help='gpu device number', type=int, default=0)
parser.add_argument('--loss', help='loss function (bce | ce | msm)', type=str, default='ce')
parser.add_argument('--lr_lambda', help='learning rate lambda', type=float, default=0.98739)

parser.add_argument('--clr_b', help='ColorJitter brightness', type=float, default=0.2)
parser.add_argument('--clr_c', help='ColorJitter contrast', type=float, default=0.2)
parser.add_argument('--clr_s', help='ColorJitter saturation', type=float, default=0.2)
parser.add_argument('--clr_h', help='ColorJitter hue', type=float, default=0.2)
parser.add_argument('--hflip', help='Random Horizontal Flip', type=float, default=0.5)
parser.add_argument('--vflip', help='Random Vertical Flip', type=float, default=0.5)
parser.add_argument('--rotate', help='Random Rotation (degree)', type=int, default=0)


parser.add_argument('--volume', help='volume directory', default='moai')
parser.add_argument('--project', help='project directory', default='test_project')
parser.add_argument('--subproject', help='subproject directory', default='test_subproject')
parser.add_argument('--task', help='task directory', default='test_task')
parser.add_argument('--version', help='version', default='v1')

opt = parser.parse_args()

##########################################################################################
### make folders
##########################################################################################

project_path = f'/{opt.volume}/{opt.project}'
# project_path = f'../{opt.project}'
if not os.path.exists(project_path):
     os.makedirs(project_path)

subproject_path = f'{project_path}/{opt.subproject}'
if not os.path.exists(subproject_path):
     os.makedirs(subproject_path)

task_path = f'{subproject_path}/{opt.task}'
if not os.path.exists(task_path):
     os.makedirs(task_path)

version_path = f'{task_path}/{opt.version}'
if not os.path.exists(version_path):
     os.makedirs(version_path)

weight_path = f'{version_path}/weights'
if not os.path.exists(weight_path):
     os.makedirs(weight_path)

tresult_path = f'{version_path}/training_result'
if not os.path.exists(tresult_path):
     os.makedirs(tresult_path)

opt.train_path = f'{task_path}/train_dataset/train'
opt.val_path = f'{task_path}/train_dataset/valid'
hyp_path = f'{task_path}/train_dataset/hyp.yaml'
data_path = f'{task_path}/train_dataset/data.yaml'

##########################################################################################
### load hyperparameter
##########################################################################################

with open(hyp_path, 'r') as file:
    params = yaml.safe_load(file)

    opt.model = params.get('model')
    opt.epoch = params.get('epoch')
    opt.bs = params.get('batch_size')
    opt.nc = params.get('num_class')
    opt.lr = params.get('learning_rate')
    opt.wd = params.get('weight_decay')
    opt.img_size = params.get('img_size')
    opt.optim = params.get('optim')
    opt.gpu = params.get('gpu')
    opt.loss = params.get('loss')
    opt.lr_lambda = params.get('lr_lambda')

    opt.clr_b = params.get('brightness')
    opt.clr_c = params.get('contrast')
    opt.clr_s = params.get('saturation')
    opt.clr_h = params.get('hue')
    opt.hflip = params.get('hflip')
    opt.vflip = params.get('vflip')
    opt.rotate = params.get('rotate')

##########################################################################################
### make results.csv header
##########################################################################################

result_file = open(f'{tresult_path}/results.csv', mode='a', newline='', encoding='utf-8')
result_csv = csv.writer(result_file)
result_csv.writerow(["epoch", "accuracy", "loss", "time"])

##########################################################################################
### load data
##########################################################################################

with open(data_path, 'r') as file:
    params = yaml.safe_load(file)

    class_names_dict = params.get('names')

    opt.class_names_list = [class_names_dict[i] for i in sorted(class_names_dict.keys())]


# fc 제외하고 freeze
# for n, p in model.named_parameters():
#     if '_fc' not in n:
#         p.requires_grad = False
# model = torch.nn.parallel.DistributedDataParallel(model)

#########################################################################################################
## parameters for dataloader
#########################################################################################################
batch_size  = opt.bs
random_seed = 555
random.seed(random_seed)
torch.manual_seed(random_seed)

#########################################################################################################
## train, valid dataset
#########################################################################################################

class_to_idx = {class_name: idx for idx, class_name in enumerate(opt.class_names_list)}

datasets_dict = {}

datasets_dict['train'] = CustomImageFolder(
                                opt.train_path,
                                transforms.Compose([
                                    transforms.Resize((opt.img_size, opt.img_size)),
                                    transforms.ToTensor(),
                                    transforms.ColorJitter(brightness=opt.clr_b, contrast=opt.clr_c, saturation=opt.clr_s, hue=opt.clr_h),
                                    transforms.RandomHorizontalFlip(p=opt.hflip),
                                    transforms.RandomVerticalFlip(p=opt.vflip),
                                    transforms.RandomRotation(degrees=opt.rotate),
                                    # transforms.RandomResizedCrop(size=512, scale=(0.08, 1.0), ratio=(0.75, 1.33), interpolation=2),
                                    # transforms.RandomErasing(),
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ]),class_to_idx=class_to_idx)

datasets_dict['valid'] = CustomImageFolder(
                                opt.val_path,
                                transforms.Compose([
                                    transforms.Resize((opt.img_size, opt.img_size)),
                                    transforms.ToTensor(),
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    transforms.ColorJitter(brightness=opt.clr_b, contrast=opt.clr_c, saturation=opt.clr_s, hue=opt.clr_h),
                                    transforms.RandomHorizontalFlip(p=opt.hflip),
                                    transforms.RandomVerticalFlip(p=opt.vflip),
                                    transforms.RandomRotation(degrees=opt.rotate),
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
device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")  # set gpu

model_name = f'efficientnet-b{opt.model}'
pretrained_model = f'./pretrained_weights/{model_name}.pth'
image_size = EfficientNet.get_image_size(model_name)
print('model input size: ', image_size)
print(model_name)
model = EfficientNet.from_pretrained(model_name, weights_path=pretrained_model, num_classes=opt.nc)
model = model.to(device)

#########################################################################################################
## Optimizer
#########################################################################################################
epoch = opt.epoch

if opt.loss == 'ce':
    criterion = nn.CrossEntropyLoss()
    
elif opt.loss == 'msm':
    criterion = nn.MultiLabelSoftMarginLoss()

if opt.optim == 'sgd':
    optimizer_ft = optim.SGD(model.parameters(), 
                             lr = opt.lr,
                             momentum=0.9,
                             weight_decay=1e-4)
elif opt.optim == 'adam':
    optimizer_ft = optim.Adam(model.parameters(), lr = opt.lr, weight_decay=opt.wd)

lmbda = lambda epoch: opt.lr_lambda
exp_lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=lmbda)

##########################################################################################
### make folders
##########################################################################################


twriter = SummaryWriter(tresult_path)

##########################################################################################
### save arguments
##########################################################################################
opt_list = [ f'{key}: {opt.__dict__[key]}' for key in opt.__dict__ ]
with open(f'{tresult_path}/hyp.yaml', 'w') as f:
    [f.write(f'{st}\n') for st in opt_list]

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
                    torch.save(model.state_dict(), f'{weight_path}/best.pt')
                    print('==> best model saved - %d / %.1f'%(best_idx, best_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc (Epoch %d): %.1f' %(best_idx, best_acc))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), f'{weight_path}/last.pt')
    print('model saved')

    result_csv.close()

    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc

if __name__ == "__main__":

    model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc = train_model(
        model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=epoch)


    

