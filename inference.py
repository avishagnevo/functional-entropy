import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from regularization import Perturbation, Regularization, RegParameters
import models
import data
from cifar10_models.vgg import vgg13_bn

# Load pretrained model
print("Loading pretrained VGG-13 model...")
model = vgg13_bn(pretrained=True)
model.eval()  # Set to evaluation mode
print("Model loaded and set to evaluation mode.")

# Load CIFAR-10 dataloader
print("Loading CIFAR-10 training data...")
trainloader = data.get_dataloader(batch_size=8, shuffle=False, train=True, cifar=True)
print("Data loaded. Number of batches:", len(trainloader))

# Initialize Regularization
print("Initializing regularization parameters...")
reg_params = RegParameters()
reg_params.estimation = 'var'  # Set the estimation method to variance
print("Regularization parameters initialized. Using variance-based estimation.")

# Regularization term computation loop
for batch_idx, (images, labels) in enumerate(trainloader):
    print(f"Processing batch {batch_idx+1}/{len(trainloader)}...")
    print(f"Input images shape: {images.shape}")  # Expected: (8, 3, 32, 32)
    
    logits = model(images)  # Forward pass
    print(f"Logits shape: {logits.shape}")  # Expected: (8, 10)
    
    # Regularization process
    expanded_logits = Perturbation.get_expanded_logits(logits, reg_params.n_samples)
    print(f"Expanded logits shape: {expanded_logits.shape}")  # Expected: (8 * n_samples, 10)
    
    inf_images = Perturbation.perturb_tensor(images, reg_params.n_samples)
    print(f"Perturbed images shape: {inf_images.shape}")  # Expected: (8 * n_samples, 3, 32, 32)
    
    inf_output = model(inf_images)
    print(f"Inference output shape: {inf_output.shape}")  # Expected: (8 * n_samples, 10)
    
    inf_loss = nn.functional.binary_cross_entropy_with_logits(inf_output, expanded_logits)
    print(f"Computed loss: {inf_loss.item():.4f}")
    
    gradients = torch.autograd.grad(inf_loss, [inf_images], create_graph=True)
    print(f"Gradients computed. Shape: {gradients[0].shape}")  # Expected: (8 * n_samples, 3, 32, 32)
    
    grads = [Regularization.get_batch_norm(gradients[0], loss=inf_loss, estimation='var')]  # Use variance-based estimation
    print(f"Gradient batch norm shape: {grads[0].shape}")
    
    inf_scores = torch.stack(grads)
    print(f"Stacked gradients shape: {inf_scores.shape}")
    
    reg_term = Regularization.get_regularization_term(inf_scores, norm=reg_params.norm, optim_method=reg_params.optim_method)
    print(f"Regularization Term (Variance-based): {reg_term.item():.4f}")

    break  # Break loop after first batch

print("Regularization term calculation complete.")
