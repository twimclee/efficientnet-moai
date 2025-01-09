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

#################################################################################################
### parameters
#################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='running name', default='twim')
parser.add_argument('--model', help='model number 0 - 7', default='0')
parser.add_argument('--data_path', help='test dataset path', type=str, default='./dataset')
parser.add_argument('--weight_path', help='weight path', type=str, default='./best.pt')
parser.add_argument('--nc', help='number of classes', type=int, default=1)
parser.add_argument('--img_size', help='image size', type=int, default=200)
parser.add_argument('--class_list', nargs='+', default=['defect'], help='class list')
parser.add_argument('--device', type=int, default=0, help='device number')
parser.add_argument('--subdir', type=bool, default=True, help='load all files in subdiretory')

parser.add_argument('--volume', help='volume directory', default='moai')
parser.add_argument('--project', help='project directory', default='test_project')
parser.add_argument('--subproject', help='subproject directory', default='test_subproject')
parser.add_argument('--task', help='task directory', default='test_task')
parser.add_argument('--version', help='version', default='v1')

opt = parser.parse_args()

#################################################################################################
### make directory
#################################################################################################

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

result_path = f'{version_path}/inference_result'
if not os.path.exists(result_path):
     os.makedirs(result_path)

opt.data_path = f'{version_path}/inference_dataset'

#################################################################################################
### set parameters
#################################################################################################
hyp_path = f'{version_path}/training_result/hyp.yaml'
with open(hyp_path, 'r') as file:
    params = yaml.safe_load(file)

    opt.nc = params.get('nc')
    opt.img_size = params.get('img_size')

#################################################################################################
### load model
#################################################################################################
opt.weight_path = f'{version_path}/weights/best.pt'

model_name = f'efficientnet-b{opt.model}'
pretrained_model = f'./pretrained_weights/{model_name}.pth'
model = EfficientNet.from_pretrained(model_name, weights_path=pretrained_model, num_classes=opt.nc)
model.load_state_dict(torch.load(opt.weight_path))
model.eval()

#################################################################################################
### load data
#################################################################################################
test_dataset = ImageDataSet(
    root = opt.data_path,
    transform = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    subdir = opt.subdir)

if len(test_dataset) == 0: exit()

dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")  # set gpu
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
            
            img = img.to(device)

            outputs = model(img)
            outputs = softmax(outputs)
            _, preds = torch.max(outputs, 1)

            outputs = outputs.cpu().detach().numpy()

            str_outputs = ""
            for score in outputs[0]:
                str_outputs += str(score) + ','

            with open(f'{result_path}/{fname}.txt', 'w') as file:
                file.write(str_outputs[:-1])

    etime = time.time()
    ttime = etime - stime
    ttl_time = round(ttime, 2)
    avg_time = round(ttime/(i+1) * 1000, 2)
    print(f"[time] total: {ttl_time}sec, avg: {avg_time}ms")


if __name__ == "__main__":
    test_and_save(model)


