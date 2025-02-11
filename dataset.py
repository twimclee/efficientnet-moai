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

        print(self.root)

        if subdir:
            self.images = self.__subdirectory()
        else:
            self.images = [join(self.root, f) for f in listdir(self.root) if isfile(join(self.root, f))]

    def __len__(self):
        return len(self.images)

    def __subdirectory(self):
        dirs = []
        for path, subdirs, files in walk(self.root):
            for name in files:
                dirs.append(join(path, name))

        return dirs

    def __getitem__(self, index):
        path = self.images[index]

        img = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)

        return img, path

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, class_to_idx=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        if class_to_idx:
            self.class_to_idx = class_to_idx
            self.idx_to_class = {v: k for k, v in class_to_idx.items()}

            new_samples = []
            for path, label in self.samples:
                new_label = self.class_to_idx[self.classes[label]]
                new_samples.append((path, new_label))
            self.samples = new_samples
