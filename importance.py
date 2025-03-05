import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from regularization import Perturbation, Regularization, RegParameters
from models import AnimalClassifier
from data import get_train_data, get_valid_data
from functorch import vmap, grad

# Automatically set the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Utility Functions
# =============================================================================
def compute_per_sample_gradient(model: nn.Module, images: torch.Tensor, label_idx: int) -> torch.Tensor:
    """
    Computes the gradient of the softmax probability for a specified label with respect to each input image.
    
    :param model: The neural network model.
    :param images: A batch of images of shape (B, C, H, W). Must have requires_grad=True.
    :param label_idx: The index of the label for which to compute the gradient.
    :return: A tensor of shape (B, C, H, W) with the gradient for each image.
    """
    def f(x: torch.Tensor) -> torch.Tensor:
        # x has shape (C, H, W); add batch dimension
        out = model(x.unsqueeze(0))              # shape: (1, num_classes)
        out_sm = F.softmax(out, dim=1)             # shape: (1, num_classes)
        return out_sm[0, label_idx]                # return the probability for the specified label

    # Compute the gradient function and apply it to each image in the batch via vmap
    grad_f = torch.func.grad(f)
    per_sample_grad = torch.vmap(grad_f)(images)  # shape: (B, C, H, W)
    return per_sample_grad


def compute_importance(gradients: torch.Tensor, estimation: str = 'var') -> float:
    """
    Computes an importance measure from the gradients using the Regularization helper.
    
    :param gradients: Tensor of gradients of shape (B, C, H, W).
    :param estimation: Estimation method to use ('var' or 'ent').
    :return: A scalar importance measure.
    """
    importance = Regularization.get_batch_norm(gradients, estimation=estimation)
    return importance


def compute_subset_importance(model: nn.Module, images: torch.Tensor, pixels: list[int], 
                              label_idx: int, reg_params: RegParameters) -> float:
    """
    Computes the importance measure for a subset of pixels for a given label by perturbing the images on that subset,
    computing the softmax probability, and then calculating the gradient with respect to the inputs.
    
    :param model: The neural network model.
    :param images: A batch of images of shape (B, C, H, W).
    :param pixels: List of pixel indices to include in the perturbation.
    :param label_idx: The index of the label for which to compute the gradient.
    :param reg_params: Regularization parameters (e.g., number of samples).
    :return: A scalar importance measure for the pixel subset.
    """
    # Perturb the images on the specified pixel subset
    inf_images = Perturbation.perturb_tensor_subset(images, pixels, reg_params.n_samples)
    # Ensure perturbed images require gradients
    inf_images.requires_grad_(True)
    # Compute per-sample gradients of the softmax probability (for label label_idx) w.r.t. perturbed images
    per_sample_grad = compute_per_sample_gradient(model, inf_images, label_idx)
    # Compute an importance measure from these gradients
    importance = compute_importance(per_sample_grad, estimation=reg_params.estimation)
    return importance

def aggregate_importance_scores(subsets_importance: dict) -> dict:
    """
    Aggregates importance scores across pixel subsets for each category.
    
    :param subsets_importance: A dictionary where each key is a pixel subset identifier (str)
                               and each value is a dictionary with keys 'ground_truth' and 'predicted'
                               containing the importance scores (floats) for that pixel subset.
    :return: A dictionary with keys 'ground_truth_total' and 'predicted_total' representing the aggregated
             importance scores for the ground truth and predicted categories, respectively.
    """
    ground_truth_total = sum(score_dict['ground_truth'] for score_dict in subsets_importance.values())
    predicted_total = sum(score_dict['predicted'] for score_dict in subsets_importance.values())
    return {'ground_truth_total': ground_truth_total, 'predicted_total': predicted_total}


# =============================================================================
# Main Block
# =============================================================================
def main():
    # --- Load Data ---
    # Here, we load validation data using your helper function.
    # (Assumes get_valid_data returns (text_dir, labels, X_valid, Y_valid))
    text_dir, labels, X_valid, Y_valid = get_valid_data()
    # Convert validation images from (N, 224, 224, 3) to (N, 3, 224, 224) and normalize pixel values to [0,1].
    X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    Y_valid_tensor = torch.tensor(Y_valid, dtype=torch.long)
    valid_dataset = TensorDataset(X_valid_tensor, Y_valid_tensor)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    # --- Load Model Checkpoint ---
    epoch = 0 
    PATH = f"checkpoints/checkpoint_{epoch}epoch.pth"
    print(f"Checkpoint saved to {PATH}")
    checkpoint = torch.load(PATH, map_location=device)
    
    num_classes = len(labels)
    model = AnimalClassifier(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    epoch_loss = checkpoint['epoch_loss']
    val_loss = checkpoint['val_loss']
    epoch_acc = checkpoint['epoch_acc']
    val_acc = checkpoint['val_acc']
    model.to(device)
    model.eval()
    print("Animal model loaded and set to evaluation mode.")
    print("Attributes:", f"epoch: {epoch}, epoch_loss: {epoch_loss}, val_loss: {val_loss}, epoch_acc: {epoch_acc}, val_acc: {val_acc}")
    
    # --- Initialize Regularization Parameters ---
    reg_params = RegParameters()
    reg_params.estimation = 'var'  # Use variance-based estimation
    print("Regularization parameters initialized. Using variance-based estimation.")
    
    # --- Regularization Term Computation ---
    # We'll compute importance for each pixel subset for both ground-truth and predicted labels.
    batch_importance = {}
    
    # Process only one batch for demonstration.
    for batch_idx, (images, batch_labels) in enumerate(valid_loader):
        # Move data to device
        images = images.to(device)
        batch_labels = batch_labels.to(device)
        ground_truth_label = batch_labels.item()
        logits = model(images)
        predicted_label = torch.argmax(logits, dim=1).item()
        
        # Define pixel subsets to evaluate
        pixel_subsets = [[0], [1]]
        subsets_importance = {}
        
        for pixels in pixel_subsets:
            importance_ground_truth = compute_subset_importance(model, images, pixels, ground_truth_label, reg_params)
            importance_predicted = compute_subset_importance(model, images, pixels, predicted_label, reg_params)
            subsets_importance[str(pixels)] = {
                'ground_truth': importance_ground_truth,
                'predicted': importance_predicted
            }
        batch_importance[batch_idx] = subsets_importance
        # Example usage within your main loop:
        # Assuming 'subsets_importance' is the dictionary computed for one batch:
        aggregated = aggregate_importance_scores(subsets_importance)
        print("Aggregated Ground Truth Importance:", aggregated['ground_truth_total'])
        print("Aggregated Predicted Importance:", aggregated['predicted_total'])
        break  # Process only the first batch for demonstration
    
    print("Importance calculation complete.")
    #print("Batch importance:", batch_importance)

if __name__ == "__main__":
    main()
