import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
#import torchvision.transforms as transforms
from regularization import Perturbation, Regularization, RegParameters
from models import AnimalClassifier
import torch.optim as optim
from data import get_train_data, get_valid_data
import torch.nn.functional as F
from functorch import vmap, grad


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
images_subsets_importance[batch_idx]['ground_truth'] = 0
images_subsets_importance[batch_idx]['predicted'] = 0

for batch_idx, (images, batch_labels) in enumerate(valid_loader):
    print(f"Processing batch {batch_idx+1}/{len(valid_loader)}...")
    #pixels_importance = []
    
    # Move the batch to the device (GPU if available)
    images = images.to(device)
    batch_labels = batch_labels.to(device)
    #print(f"Input images shape: {images.shape}")  # Expected: (batch, 3, 224, 224)
    print(f"Batch labels shape: {batch_labels.shape}")  # Expected: (batch)
    ground_truth_label = batch_labels.item()
    
    # Forward pass: compute logits
    logits = model(images)
    print(f"Logits shape: {logits.shape}")  # Expected: (batch, num_classes)

    # Compute the predicted labels for each image in the batch
    predicted_labels = torch.argmax(logits, dim=1)
    print(f"Predicted labels shape: {predicted_labels.shape}") # Expected: (batch)
    predicted_label = predicted_labels.item()
    
    # Compute expanded logits (replicates logits for n_samples)
    #expanded_logits = Perturbation.get_expanded_logits(logits, reg_params.n_samples)
    #print(f"Expanded logits shape: {expanded_logits.shape}")  # Expected: (batch * n_samples, num_classes)
    
    pixels_subsets = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
    subsets_importance = {}
    images_subsets_importance = {}

    for pixels in pixels_subsets:

        # Perturb input images using the helper function
        inf_images = Perturbation.perturb_tensor_subset(images, pixels , reg_params.n_samples)
        #print(f"Perturbed images shape: {inf_images.shape}")  # Expected: (batch * n_samples, 3, 224, 224)
        
        # Forward pass on the perturbed images
        inf_output = model(inf_images)
        #print(f"Inference output shape: {inf_output.shape}")  # Expected: (batch * n_samples, num_classes)

        importance_wrt_labels = {}
        subsets_importance['ground_truth'] = 0
        subsets_importance['predicted'] = 0
        for label in [ground_truth_label, predicted_label]:

            # Define a function that, for a single image (shape: (C, H, W)), computes the softmax probability for the chosen label.
            def f(x):
                label_idx = label
                # x has shape (C, H, W)
                # Model expects a batch dimension, so add one
                out = model(x.unsqueeze(0))            # shape: (1, num_classes)
                out_sm = F.softmax(out, dim=1)           # shape: (1, num_classes)
                return out_sm[0, label_idx]              # return the probability for label_idx
                
            # Create a function to compute the gradient of f with respect to its input.
            grad_f = torch.func.grad(f)

            # Use vmap to apply grad_f to each image in the batch.
            gradients = torch.vmap(grad_f)(inf_images)  # Expected shape: (B, C, H, W)
            print("Per-sample gradient shape:", gradients.shape) # ([10, 3, 224, 224])
            
            # Calculate the expectation of the batch gradient
            importance = Regularization.get_batch_norm(gradients, estimation='var')
            print(f"Gradient batch norm shape: {importance.shape}") # shape 0
            print(f"Gradient batch norm: {importance}")

            #reg_term = Regularization.get_regularization_term(inf_scores, norm=reg_params.norm, optim_method=reg_params.optim_method)
            print(f"Importance Term (Variance-based): {importance:.4f}")

            if label == ground_truth_label:
                importance_wrt_labels['ground_truth'] = importance
            else:
                importance_wrt_labels['predicted'] = importance    
            #break # Process only the first label for demonstration

        subsets_importance['ground_truth'] += importance_wrt_labels['ground_truth']
        subsets_importance['predicted'] += importance_wrt_labels['predicted']
        #break # Process only the first pixels subset for demonstration    
    
    images_subsets_importance[batch_idx]['ground_truth'] += subsets_importance['ground_truth']
    images_subsets_importance[batch_idx]['predicted'] += subsets_importance['predicted']

    print(images_subsets_importance[batch_idx]['ground_truth'])
    print(images_subsets_importance[batch_idx]['predicted'])
    break  # Process only the first image for demonstration

print("Imporatance calculation complete.")