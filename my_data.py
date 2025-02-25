import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])
    ])
    return transform

def get_cifar_dataset(train_split = True):
    if train_split:
        return torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=get_transform())
    else:
        return torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=get_transform())  


def get_animales_dataset(train_split = True):
    base_dir  = '/kaggle/input/animals-detection-images-dataset/'
    os.listdir(base_dir)

    if train_split:
        datadir = os.path.join(base_dir, 'train')
    else:
        datadir = os.path.join(base_dir, 'test')
    return torchvision.datasets.DatasetFolder(root = datadir, transform = get_transform())    

    # Number of classes
    #classes = os.listdir(train_dir)
    #return torchvision.datasets.DatasetFolder(root = train_dir) #find_classes(directory: Union[str, Path]) → Tuple[List[str], Dict[str, int]]


def get_dataloader(batch_size = 64, shuffle=False, train = True, cifar = True):
    if cifar:
        dataset = get_cifar_dataset(train_split = train)
    else:
        dataset = get_animales_dataset()
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
