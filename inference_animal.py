import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from regularization import Perturbation, Regularization, RegParameters
from models import AnimalClassifier
import torch.optim as optim
from data import get_train_data, get_valid_data

# --- Load data-loading blocks ---
#train_dir, labels, X, Y = get_train_data()
text_dir, labels, X_valid, Y_valid = get_valid_data()

# For demonstration, we'll use the validation set.
# Convert validation images from (N, 224, 224, 3) to (N, 3, 224, 224) and normalize pixel values to [0,1].
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
Y_valid_tensor = torch.tensor(Y_valid, dtype=torch.long)

# Create a DataLoader for the Animal validation data.
valid_dataset = TensorDataset(X_valid_tensor, Y_valid_tensor)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

# --- Load your Animal model from saved weights ---
epoch = 0 
PATH = f"checkpoints/checkpoint_{epoch}epoch.pth"
print(f"Checkpoint saved to {PATH}")

# --- Loading Checkpoint ---
# Determine the device to load the model to (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading checkpoint on device:", device)

# Load checkpoint with map_location=device
checkpoint = torch.load(PATH, map_location=device)

#labels = os.listdir(train_dir)
num_classes = len(labels)  

# Create your model and optimizer (using your model class and optimizer class)
model = AnimalClassifier(num_classes=num_classes)          
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the state dictionaries
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
epoch_loss = checkpoint['epoch_loss']
val_loss = checkpoint['val_loss']
epoch_acc = checkpoint['epoch_acc']
val_acc = checkpoint['val_acc']

# Move model to the device and set mode
model.to(device)
model.eval()  # or model.train() if continuing training
print("Animal model loaded and set to evaluation mode.")
print("Animal model loaded has the following attributes:",
      " epoch:", epoch, " epoch_loss:", epoch_loss, 
      " val_loss:", val_loss, " epoch_acc:", epoch_acc, " val_acc:", val_acc)

# --- Initialize Regularization Parameters ---
reg_params = RegParameters()
reg_params.estimation = 'var'  # Use variance-based estimation
print("Regularization parameters initialized. Using variance-based estimation.")

# --- Regularization Term Computation Loop ---
images_pixels_importance = []
for batch_idx, (images, batch_labels) in enumerate(valid_loader):
    print(f"Processing batch {batch_idx+1}/{len(valid_loader)}...")
    #pixels_importance = []
    
    # Move the batch to the device (GPU if available)
    images = images.to(device)
    batch_labels = batch_labels.to(device)
    #print(f"Input images shape: {images.shape}")  # Expected: (batch, 3, 224, 224)
    
    # Forward pass: compute logits
    logits = model(images)
    print(f"Logits shape: {logits.shape}")  # Expected: (8, num_classes)
    print(f"Batch labels: {batch_labels}")
    
    # Compute expanded logits (replicates logits for n_samples)
    expanded_logits = Perturbation.get_expanded_logits(logits, reg_params.n_samples)
    print(f"Expanded logits shape: {expanded_logits.shape}")  # Expected: (batch * n_samples, num_classes)
    
    pixels_subsets = [[0]]
    subsets_importance = []
    images_subsets_importance = []

    for pixels in pixels_subsets:
        # Perturb input images using the helper function
        inf_images = Perturbation.perturb_tensor_subset(images, pixels , reg_params.n_samples)
        #print(f"Perturbed images shape: {inf_images.shape}")  # Expected: (batch * n_samples, 3, 224, 224)
        
        # Forward pass on the perturbed images
        inf_output = model(inf_images)
        #print(f"Inference output shape: {inf_output.shape}")  # Expected: (batch * n_samples, num_classes)
        
        # Compute binary cross entropy loss with logits (targets: expanded_logits)
        inf_loss = nn.functional.binary_cross_entropy_with_logits(inf_output, expanded_logits)
        print(f"Computed loss: {inf_loss.item():.4f}")
        
        # Compute gradients of the loss with respect to the perturbed images
        gradients = torch.autograd.grad(inf_loss, [inf_images], create_graph=True)
        print(f"Gradients computed. Shape: {gradients[0].shape}")  # Expected: (batch * n_samples, 3, 224, 224)
        
        # Process gradients with the batch normalization helper
        grads = [Regularization.get_batch_norm(gradients[0], loss=inf_loss, estimation='var')]
        print(f"Gradient batch norm shape: {grads[0].shape}")
        print(f"Gradient batch norm: {grads[0]}")
        
        # Stack gradients and compute the regularization term
        inf_scores = torch.stack(grads)
        print(f"Stacked gradients shape: {inf_scores.shape}")
        print(f"Stacked gradients: {inf_scores}")
        
        reg_term = Regularization.get_regularization_term(inf_scores, norm=reg_params.norm,
                                                        optim_method=reg_params.optim_method)
        print(f"Regularization Term (Variance-based): {reg_term.item():.4f}")

        subsets_importance.append(reg_term.item())
    
    images_subsets_importance.append(subsets_importance)

    break  # Process only the first image for demonstration

print("Regularization term calculation complete.")


'''
    images_dim = list(images.shape)
    max_pixel = images_dim[-2]*images_dim[-1]

    for pixel in range(max_pixel):
        # Perturb input images using the helper function
        inf_images = Perturbation.perturb_tensor_exept_pixel(images,pixel , reg_params.n_samples)
        #print(f"Perturbed images shape: {inf_images.shape}")  # Expected: (batch * n_samples, 3, 224, 224)
        
        # Forward pass on the perturbed images
        inf_output = model(inf_images)
        #print(f"Inference output shape: {inf_output.shape}")  # Expected: (batch * n_samples, num_classes)
        
        # Compute binary cross entropy loss with logits (targets: expanded_logits)
        inf_loss = nn.functional.binary_cross_entropy_with_logits(inf_output, expanded_logits)
        print(f"Computed loss: {inf_loss.item():.4f}")
        
        # Compute gradients of the loss with respect to the perturbed images
        gradients = torch.autograd.grad(inf_loss, [inf_images], create_graph=True)
        print(f"Gradients computed. Shape: {gradients[0].shape}")  # Expected: (batch * n_samples, 3, 224, 224)
        
        # Process gradients with the batch normalization helper
        grads = [Regularization.get_batch_norm(gradients[0], loss=inf_loss, estimation='var')]
        print(f"Gradient batch norm shape: {grads[0].shape}")
        print(f"Gradient batch norm: {grads[0]}")
        
        # Stack gradients and compute the regularization term
        inf_scores = torch.stack(grads)
        print(f"Stacked gradients shape: {inf_scores.shape}")
        print(f"Stacked gradients: {inf_scores}")
        
        reg_term = Regularization.get_regularization_term(inf_scores, norm=reg_params.norm,
                                                        optim_method=reg_params.optim_method)
        print(f"Regularization Term (Variance-based): {reg_term.item():.4f}")
        
        pixels_importance.append(reg_term.item())
    
        break  # Process only the first pixel for demonstration
    '''