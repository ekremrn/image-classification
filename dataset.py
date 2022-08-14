import os
import numpy as np
import pandas as pd
import albumentations as A

from PIL import Image
from albumentations import pytorch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100


#
class CustomDataset(Dataset):


    def __init__(self, data_dir, dataset, transform, train = True):
        
        path = os.path.join(data_dir, dataset)

        csv = 'train.csv' if train else 'test.csv'

        self.data = pd.read_csv( os.path.join(path, csv) )
        self.classes = open(os.path.join(path, "classes.txt"), 'r').read().splitlines()
        
        self.transform = transform


        def __len__(self):
            return len(self.data)
        

        def __getitem__(self, idx):

            img_path = os.path.join(self.data_dir, self.data.iloc[idx, 0])
            label_name = self.data.iloc[idx, 1]
            # X
            img = Image.open(img_path)
            image = self.transform(image = np.array(img))["image"]            
            # Y
            label = self.classes.index(label_name)
            return image, label


#
def get_transforms(size):

    # Different Archs?
    mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_tr = A.Compose(
        [
            #A.Resize(cfg.resize, cfg.resize),
            A.RandomResizedCrop(size, size),
            A.HorizontalFlip(0.5),
            A.ImageCompression(quality_lower = 50, quality_upper = 100),
            A.ShiftScaleRotate(shift_limit = 0.2, scale_limit = 0.2, rotate_limit = 10, border_mode = 0, p = 0.5),
            A.Cutout(max_h_size = int(size * 0.4), max_w_size = int(size * 0.4), num_holes = 1, p = 0.5),
            A.transforms.CoarseDropout(max_holes = 16, max_height = 16, max_width = 16, p = 0.5),
            A.transforms.CLAHE(clip_limit = 4.0, tile_grid_size = (8, 8), p = 0.5),
            A.transforms.HueSaturationValue(hue_shift_limit = 20, sat_shift_limit = 30, val_shift_limit = 20, p = 0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(*mean_std),
            A.pytorch.transforms.ToTensorV2()
        ]
    )

    eval_tr = A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(*mean_std),
            A.pytorch.transforms.ToTensorV2()
        ]
    )

    return {'train' : train_tr, 'test' : eval_tr}


#
def get(root, dataset, transforms, batch_size):
    
    if dataset == "CIFAR10":
        trainset = CIFAR10(root = root, train = True, 
                           download = True, transform = transforms['train'])
        testset = CIFAR10(root = root, train = False, 
                           download = True, transform = transforms['test'])
    elif dataset == "CIFAR100":
        trainset = CIFAR100(root = root, train = True, 
                           download = True, transform = transforms['train'])
        testset = CIFAR100(root = root, train = False, 
                           download = True, transform = transforms['test'])
    else:
        trainset = CustomDataset(root = root, dataset = dataset, 
                                train = True, transform = transforms['train'])
        testset = CustomDataset(root = root, dataset = dataset, 
                                train = False, transform = transforms['test'])
    
    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True)

    testloader = DataLoader(testset, batch_size = batch_size, shuffle = False)

    return {'training' : trainloader, 'validation' : testloader}

    