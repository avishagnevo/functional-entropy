import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from regularization import Perturbation, Regularization, RegParameters

# --- Assume these variables are already defined from your earlier data-loading blocks ---
# train_dir, test_dir, labels, X, Y, X_valid, Y_valid

# For demonstration, we'll use the validation set.
# Convert validation images from (N, 224, 224, 3) to (N, 3, 224, 224) and normalize pixel values to [0,1].
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
Y_valid_tensor = torch.tensor(Y_valid, dtype=torch.long)

# Create a DataLoader for the Animal validation data.
valid_dataset = TensorDataset(X_valid_tensor, Y_valid_tensor)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

# --- Define your AnimalClassifier if not already imported ---
class AnimalClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AnimalClassifier, self).__init__()
        # Load a pre-trained ResNet50 and remove its final fully-connected layer
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Freeze feature extractor parameters
        for param in self.features.parameters():
            param.requires_grad = False
        # Define the classifier head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, num_classes)
        # Note: We omit softmax since CrossEntropyLoss expects raw logits
        
    def forward(self, x):
        x = self.features(x)         # (batch, 2048, 1, 1)
        x = x.view(x.size(0), -1)      # Flatten to (batch, 2048)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# --- Load your Animal model from saved weights ---
num_classes = len(labels)  # e.g., 29 classes
model = AnimalClassifier(num_classes=num_classes)
model.load_state_dict(torch.load('ResNet50_DEL.pth'))
model.eval()
print("Animal model loaded and set to evaluation mode.")

# --- Initialize Regularization Parameters ---
reg_params = RegParameters()
reg_params.estimation = 'var'  # Use variance-based estimation
print("Regularization parameters initialized. Using variance-based estimation.")

# --- Regularization Term Computation Loop on the first batch ---
for batch_idx, (images, labels) in enumerate(valid_loader):
    print(f"Processing batch {batch_idx+1}/{len(valid_loader)}...")
    print(f"Input images shape: {images.shape}")  # Expected: (8, 3, 224, 224)
    
    # Forward pass: compute logits
    logits = model(images)
    print(f"Logits shape: {logits.shape}")  # Expected: (8, num_classes)
    
    # Compute expanded logits (replicates logits for n_samples)
    expanded_logits = Perturbation.get_expanded_logits(logits, reg_params.n_samples)
    print(f"Expanded logits shape: {expanded_logits.shape}")  # Expected: (8 * n_samples, num_classes)
    
    # Perturb input images using the helper function
    inf_images = Perturbation.perturb_tensor(images, reg_params.n_samples)
    print(f"Perturbed images shape: {inf_images.shape}")  # Expected: (8 * n_samples, 3, 224, 224)
    
    # Forward pass on the perturbed images
    inf_output = model(inf_images)
    print(f"Inference output shape: {inf_output.shape}")  # Expected: (8 * n_samples, num_classes)
    
    # Compute binary cross entropy loss with logits (targets: expanded_logits)
    inf_loss = nn.functional.binary_cross_entropy_with_logits(inf_output, expanded_logits)
    print(f"Computed loss: {inf_loss.item():.4f}")
    
    # Compute gradients of the loss with respect to the perturbed images
    gradients = torch.autograd.grad(inf_loss, [inf_images], create_graph=True)
    print(f"Gradients computed. Shape: {gradients[0].shape}")  # Expected: (8 * n_samples, 3, 224, 224)
    
    # Process gradients with the batch normalization helper
    grads = [Regularization.get_batch_norm(gradients[0], loss=inf_loss, estimation='var')]
    print(f"Gradient batch norm shape: {grads[0].shape}")
    
    # Stack gradients and compute the regularization term
    inf_scores = torch.stack(grads)
    print(f"Stacked gradients shape: {inf_scores.shape}")
    
    reg_term = Regularization.get_regularization_term(inf_scores, norm=reg_params.norm,
                                                      optim_method=reg_params.optim_method)
    print(f"Regularization Term (Variance-based): {reg_term.item():.4f}")
    
    break  # Process only the first batch for demonstration

print("Regularization term calculation complete.")
