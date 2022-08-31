import os
import cv2
import pandas as pd
import albumentations as A

from albumentations import pytorch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


#
class CustomDataset(Dataset):

    def __init__(self, root, dataset, transform, train = True):
        
        self.path = os.path.join(root, dataset)

        csv = 'train.csv' if train else 'test.csv'

        self.data = pd.read_csv( os.path.join(self.path, csv) )
        self.classes = open(os.path.join(self.path, "classes.txt"), 'r').read().splitlines()
        
        self.transform = transform
        self.num_classes = len(self.classes)


    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):

        img_path = os.path.join(self.path, self.data.iloc[idx, 0])
        label_name = self.data.iloc[idx, 1]
        # X
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image = image)["image"]            
        # Y
        label = self.classes.index(label_name)
        return image, label


#
class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


#
class CIFAR100(datasets.CIFAR100):
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


#
def get_transforms(size, aug_mode):

    # Different Archs?
    mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_tr = []

    train_tr.extend([
        #A.Resize(size, size),
        A.RandomResizedCrop(size, size),
        A.HorizontalFlip(0.5)])

    if aug_mode == 'special':
        train_tr.extend([
            
        ])

    if aug_mode == 'big':
        train_tr.extend([            
            A.ImageCompression(quality_lower = 50, quality_upper = 100),
            A.ShiftScaleRotate(shift_limit = 0.2, scale_limit = 0.2, rotate_limit = 10, border_mode = 0, p = 0.5),
            A.Cutout(max_h_size = int(size * 0.4), max_w_size = int(size * 0.4), num_holes = 1, p = 0.5),
            A.transforms.CoarseDropout(max_holes = 16, max_height = 16, max_width = 16, p = 0.5),
            A.transforms.CLAHE(clip_limit = 4.0, tile_grid_size = (8, 8), p = 0.5),
            A.transforms.HueSaturationValue(hue_shift_limit = 20, sat_shift_limit = 30, val_shift_limit = 20, p = 0.5),
            A.RandomBrightnessContrast(p=0.3)])

    train_tr.extend([A.Normalize(*mean_std), A.pytorch.transforms.ToTensorV2()])

    eval_tr = [
        A.Resize(size, size),
        A.Normalize(*mean_std),
        A.pytorch.transforms.ToTensorV2()]

    return {'train' : A.Compose(train_tr), 'test' : A.Compose(eval_tr)}


#
def get(root, dataset, transforms, batch_size):
    
    if dataset == "CIFAR10":
        trainset = CIFAR10(root = root, train = True, 
                           download = True, transform = transforms['train'])
        testset = CIFAR10(root = root, train = False, 
                           download = True, transform = transforms['test'])
        num_classes = 10
    elif dataset == "CIFAR100":
        trainset = CIFAR100(root = root, train = True, 
                           download = True, transform = transforms['train'])
        testset = CIFAR100(root = root, train = False, 
                           download = True, transform = transforms['test'])
        num_classes = 100
    else:
        trainset = CustomDataset(root = root, dataset = dataset, 
                                train = True, transform = transforms['train'])
        testset = CustomDataset(root = root, dataset = dataset, 
                                train = False, transform = transforms['test'])
        num_classes = trainset.num_classes
    
    trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True)

    testloader = DataLoader(testset, batch_size = batch_size, shuffle = False)

    return {'training' : trainloader, 'validation' : testloader, 'num_classes' : num_classes}

    