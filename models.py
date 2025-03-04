import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import detectors
import timm
import torchvision.models as models
import torch.nn.functional as F
import torch.nn.functional as F
from functorch import vmap, grad


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


class AnimalClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AnimalClassifier, self).__init__()
        # Load a pre-trained ResNet50 model
        resnet = models.resnet50(pretrained=True)
        # Remove the final fully-connected layer (i.e. include_top=False)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze the ResNet50 feature extractor parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Define the classification head
        self.flatten = nn.Flatten()  # Alternatively, use x.view(x.size(0), -1) in forward
        self.fc1 = nn.Linear(2048, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)  # For producing probabilities
        
    def forward(self, x):
        x = self.features(x)        # Output shape: (batch, 2048, 1, 1)
        x = x.view(x.size(0), -1)     # Flatten to (batch, 2048)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

    def model_softmax(self, x):
        return F.softmax(self(x), dim=1)



