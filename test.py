from efficientnet_pytorch import EfficientNet

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.utils import save_image

from dataset import ImageDataSet

import numpy as np
import os

import argparse

import shutil
import time
import yaml

from pathfilemgr import MPathFileManager
from hyp_data import MHyp, MData

#################################################################################################
### parameters
#################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--volume', help='volume directory', default='moai')
parser.add_argument('--project', help='project directory', default='test_project')
parser.add_argument('--subproject', help='subproject directory', default='test_subproject')
parser.add_argument('--task', help='task directory', default='test_task')
parser.add_argument('--version', help='version', default='v1')
opt = parser.parse_args()

mpfm = MPathFileManager(opt.volume, opt.project, opt.subproject, opt.task, opt.version)
mhyp = MHyp()
mdata = MData()
mpfm.load_test_hyp(mhyp)
mpfm.load_test_data(mdata)

#################################################################################################
### load model
#################################################################################################
weight_path = f'{mpfm.weight_path}/best.pt'

model_name = f'efficientnet-b{mhyp.model}'
pretrained_model = f'./pretrained_weights/{model_name}.pth'
model = EfficientNet.from_pretrained(model_name, weights_path=pretrained_model, num_classes=mhyp.num_class)
model.load_state_dict(torch.load(weight_path))
model.eval()

#################################################################################################
### load data
#################################################################################################
test_dataset = ImageDataSet(
    root = mpfm.test_dataset,
    transform = transforms.Compose([
        transforms.Resize((mhyp.img_size, mhyp.img_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    subdir = False)

if len(test_dataset) == 0: exit()

dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

device = torch.device(f"cuda:{mhyp.gpu}" if torch.cuda.is_available() else "cpu")  # set gpu
model = model.to(device)

############################################################################################
## warm up
############################################################################################
for i, (img, path) in enumerate(dataloader):
    img = img.to(device)
    model(img)
    if i == 5:
        break

############################################################################################
## test and save
############################################################################################+

softmax = nn.Softmax(dim=1)
def test_and_save(model):
    model.eval()

    total = len(test_dataset)
    stime = time.time()
    with torch.no_grad():
        
        for i, (img, path) in enumerate(dataloader):
            fname = path[0].rsplit('.', 1)[0]
            fname = fname.rsplit('/', 1)[-1]
            # fname = fname.rsplit('\\', 1)[-1]
            
            img = img.to(device)

            outputs = model(img)
            outputs = softmax(outputs)
            _, preds = torch.max(outputs, 1)

            outputs = outputs.cpu().detach().numpy()

            str_outputs = ""
            clss_idx = 0
            for score in outputs[0]:
                str_outputs += mdata.names[clss_idx]  + ',' + str(score) + '\n'
                clss_idx += 1

            with open(f'{mpfm.test_result}/{fname}.txt', 'w') as file:
                file.write(str_outputs[:-1])

    etime = time.time()
    ttime = etime - stime
    ttl_time = round(ttime, 2)
    avg_time = round(ttime/(i+1) * 1000, 2)
    print(f"[time] total: {ttl_time}sec, avg: {avg_time}ms")


if __name__ == "__main__":
    test_and_save(model)

