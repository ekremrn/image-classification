import os
import numpy as np
import pandas as pd
import albumentations as A

from PIL import Image, ImageFile
from albumentations import pytorch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


#
class CustomDataset(Dataset):
    def __init__(self, root, dataset, transform, train=True):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        self.path = os.path.join(root, dataset)

        csv = "train.csv" if train else "test.csv"

        self.data = pd.read_csv(os.path.join(self.path, csv))
        self.classes = (
            open(os.path.join(self.path, "classes.txt"), "r").read().splitlines()
        )

        self.transform = transform
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = os.path.join(self.path, self.data.iloc[idx, 0])
        label_name = self.data.iloc[idx, 1]
        # X
        img = Image.open(img_path)
        image = img.convert("RGB") if len(img.size) == 2 else img
        image = self.transform(image=np.array(image))["image"]
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

    if aug_mode == "min":
        train_tr = [A.Resize(size, size), A.HorizontalFlip(0.5)]

    elif aug_mode == "standart":
        train_tr = [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.RandomCrop(size, size, p=0.2),
                    A.ShiftScaleRotate(p=0.2),
                    A.RandomBrightnessContrast(p=0.2),
                ],
                p=0.2,
            ),
            # Noise
            A.OneOf(
                [
                    A.IAAAdditiveGaussianNoise(),
                    A.GaussNoise(),
                ],
                p=0.3,
            ),
            # Blur
            A.OneOf(
                [
                    A.MotionBlur(p=0.3),
                    A.MedianBlur(blur_limit=3, p=0.3),
                    A.Blur(blur_limit=3, p=0.3),
                ],
                p=0.3,
            ),
            # Distortion
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.3),
                    A.IAAPiecewiseAffine(p=0.3),
                ],
                p=0.3,
            ),
            # Color
            A.OneOf(
                [
                    A.CLAHE(clip_limit=2),
                    A.IAASharpen(),
                    A.IAAEmboss(),
                    A.RandomBrightnessContrast(),
                ],
                p=0.3,
            ),
            # Weather
            A.OneOf(
                [
                    A.RandomSunFlare(p=0.3),
                    A.RandomRain(p=0.3),
                ],
                p=0.3,
            ),
            A.HueSaturationValue(p=0.3),
        ]

    elif aug_mode == "special":
        train_tr = [
            A.Resize(size, size),
            # Weather situations
            A.RandomSunFlare(p=0.25),
            A.RandomRain(p=0.25),
            # A.RandomShadow(p=0.25),
            # A.RandomFog(p=0.25),
            # Rotation
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.25),
            A.ShiftScaleRotate(p=0.25),
            # Color
            A.ChannelShuffle(p=0.25),
            A.RandomBrightnessContrast(p=0.25),
        ]

    train_tr.extend([A.Normalize(*mean_std), A.pytorch.transforms.ToTensorV2()])

    eval_tr = [
        A.Resize(size, size),
        A.Normalize(*mean_std),
        A.pytorch.transforms.ToTensorV2(),
    ]

    return {"train": A.Compose(train_tr), "test": A.Compose(eval_tr)}


#
def get(root, dataset, transforms, batch_size):

    if dataset == "CIFAR10":
        trainset = CIFAR10(
            root=root, train=True, download=True, transform=transforms["train"]
        )
        testset = CIFAR10(
            root=root, train=False, download=True, transform=transforms["test"]
        )
        num_classes = 10
    elif dataset == "CIFAR100":
        trainset = CIFAR100(
            root=root, train=True, download=True, transform=transforms["train"]
        )
        testset = CIFAR100(
            root=root, train=False, download=True, transform=transforms["test"]
        )
        num_classes = 100
    else:
        trainset = CustomDataset(
            root=root, dataset=dataset, train=True, transform=transforms["train"]
        )
        testset = CustomDataset(
            root=root, dataset=dataset, train=False, transform=transforms["test"]
        )
        num_classes = trainset.num_classes

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return {
        "training": trainloader,
        "validation": testloader,
        "num_classes": num_classes,
    }
