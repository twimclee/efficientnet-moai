from os import listdir, walk
from os.path import isfile, join

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision import datasets, transforms

from PIL import Image
import numpy as np

class ImageDataSet(Dataset):
    def __init__(self, root='test', transform=None, subdir=False):
        self.root = root
        self.transform = transform

        if subdir:
            self.images = self.__subdirectory()
        else:
            self.images = [f for f in listdir(self.root) if isfile(join(self.root, f))]

    def __len__(self):
        return len(self.images)

    def __subdirectory(self):
        dirs = []
        for path, subdirs, files in walk(self.root):
            for name in files:
                dirs.append(join(path, name))

        return dirs

    def __getitem__(self, index):
        # path = join(self.root, self.images[index])
        path = self.images[index]  # for linux
        img = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)

        return img, path

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, class_to_idx=None):
        # 기존 ImageFolder를 초기화
        super().__init__(root, transform=transform, target_transform=target_transform)

        if class_to_idx:
            self.class_to_idx = class_to_idx
            # 클래스 인덱스를 업데이트합니다
            self.idx_to_class = {v: k for k, v in class_to_idx.items()}

            new_samples = []
            for path, label in self.samples:
                new_label = self.class_to_idx[self.classes[label]]
                new_samples.append((path, new_label))
            self.samples = new_samples
