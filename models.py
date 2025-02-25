import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import detectors
import timm

model = timm.create_model("resnet50_cifar10", pretrained=True)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 16 * 16, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        return x


def get_model(use_pretrained = True):
    if use_pretrained:
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for CIFAR-10 classes
        print("Using pretrained ResNet-18 model.")
    
    else:
        model = SimpleCNN()
        print("Using SimpleCNN model.")

    return model



