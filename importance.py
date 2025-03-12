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
from typing import List  
import json
import matplotlib.pyplot as plt
import functorch

#from functorch import vmap, grad

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
    if not torch.cuda.is_available():
        grad_f = torch.func.grad(f)
        per_sample_grad = torch.vmap(grad_f)(images)  # shape: (B, C, H, W)
    else:
        grad_f = functorch.grad(f)
        per_sample_grad = functorch.vmap(grad_f)(images)  # shape: (B, C, H, W)    
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


def generate_saliency_map(image_path: str, model: torch.nn.Module, labels: list, reg_params: RegParameters) -> None:
    """
    Loads an image from the given path, extracts ground-truth labels from the filename 
    (expected format: "Label1_Label2.png"), computes the per-pixel gradient (saliency map) 
    for each label, and displays the original image with the saliency maps.
    
    The saliency map is computed by calculating the gradient of the softmax probability 
    (for the chosen label) with respect to the input image. The gradient is then reduced 
    by taking the maximum absolute value across channels.
    
    :param image_path: Path to the image file.
    :param model: The classification model.
    :param labels: List of valid labels. The image filename must contain two labels 
                   (separated by an underscore) that are present in this list.
    :param reg_params: Regularization parameters (used to set the estimation method and number of samples).
    :return: None. Displays the original image and its saliency maps.
    """
    # Load and preprocess the image
    img = cv.imread(image_path)
    if img is None:
        print(f"Error loading image from {image_path}")
        return
    # Convert from BGR to RGB and resize to (224, 224)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (224, 224))
    
    # Convert image to tensor (shape: (1, 3, 224, 224)) and normalize to [0,1]
    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    # Parse the filename to extract ground truth labels
    base_name = os.path.basename(image_path)
    name_no_ext, _ = os.path.splitext(base_name)
    gt_labels = name_no_ext.split('_')
    if len(gt_labels) < 2:
        print("Warning: Less than 2 labels found in the filename. Expected format: Label1_Label2.png")
    
    saliency_maps = {}
    for label in gt_labels:
        try:
            label_idx = labels.index(label)
        except ValueError:
            print(f"Label '{label}' not found in the provided labels list.")
            continue
        # Ensure the image tensor requires gradients
        img_tensor.requires_grad_(True)
        
        # Compute the per-sample gradient of the softmax probability for the given label
        grad_tensor = compute_per_sample_gradient(model, img_tensor, label_idx)  # shape: (1, 3, 224, 224)
        
        # Compute saliency map: take the max absolute gradient across channels, resulting in (1, 224, 224)
        sal_map = torch.max(torch.abs(grad_tensor), dim=1)[0]  
        saliency_maps[label] = sal_map.squeeze().detach().cpu().numpy()
    
    # Plot the original image and saliency maps side-by-side
    num_plots = len(saliency_maps) + 1
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    i = 1
    for label, sal_map in saliency_maps.items():
        axes[i].imshow(sal_map, cmap='hot')
        axes[i].set_title(f"Saliency: {label}")
        axes[i].axis("off")
        i += 1
    plt.tight_layout()
    plt.show()
    


def load_config(config_path: str) -> dict:
    """
    Loads the configuration from a JSON file.
    
    :param config_path: Path to the JSON config file.
    :return: A dictionary with configuration parameters.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def compute_subset_importance(model: nn.Module, images: torch.Tensor, pixels: list[int], 
                              label_idx: int, reg_params: RegParameters) -> float:
    """
    Computes the importance measure for a subset of pixels for a given label by perturbing the images
    on that subset, computing the softmax probability, and then calculating the gradient with respect
    to the inputs.
    
    :param model: The neural network model.
    :param images: A batch of images of shape (B, C, H, W).
    :param pixels: List of pixel indices (flattened) to include in the perturbation.
    :param label_idx: The index of the label for which to compute the gradient.
    :param reg_params: Regularization parameters.
    :return: A scalar importance measure for the pixel subset.
    """
    inf_images = Perturbation.perturb_tensor_subset(images, pixels, reg_params.n_samples)
    inf_images.requires_grad_(True)
    per_sample_grad = compute_per_sample_gradient(model, inf_images, label_idx)
    importance = compute_importance(per_sample_grad, estimation=reg_params.estimation)
    return importance

def compute_pixel_level_importance(model: nn.Module, image: torch.Tensor, label_idx: int, 
                                   reg_params: RegParameters) -> torch.Tensor:
    """
    Computes an information map for a single image for a specific label using the developed method.
    For each pixel in the image, computes the importance measure via perturbations.
    
    :param model: The neural network model.
    :param image: A single image tensor of shape (1, 3, H, W) with requires_grad=True.
    :param label_idx: The index of the label for which to compute the importance.
    :param reg_params: Regularization parameters.
    :return: A tensor of shape (H, W) with the importance score for each pixel.
    """
    _, C, H, W = image.shape
    num_pixels = H * W
    saliency_map = torch.zeros((H * W,), device=image.device)
    for idx in range(num_pixels):
        imp = compute_subset_importance(model, image, [idx], label_idx, reg_params)
        saliency_map[idx] = imp
        if idx % 100 == 0:
            print(f"Processed {idx} pixels.")
    return saliency_map.view(H, W)

def generate_information_map(image_path: str, model: nn.Module, labels: list, 
                             reg_params: RegParameters) -> None:
    """
    Loads an image from IMAGE_PATH, extracts the two ground truth labels from its filename (expected format: 
    "Label1_Label2.png"), computes a pixel-level information map for each label using the developed method, and 
    displays the original image alongside the maps.
    
    :param image_path: Path to the image file.
    :param model: The classification model.
    :param labels: List of valid labels. The image filename should contain two labels separated by an underscore.
    :param reg_params: Regularization parameters.
    :return: None. Displays the original image and its information maps.
    """
    # Load and preprocess the image
    img = cv.imread(image_path)
    if img is None:
        print(f"Error loading image from {image_path}")
        return
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (224, 224))
    
    # Convert to tensor: shape (1, 3, 224, 224)
    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    img_tensor.requires_grad_(True)
    
    # Extract ground truth labels from filename
    base_name = os.path.basename(image_path)
    name_no_ext, _ = os.path.splitext(base_name)
    gt_labels = name_no_ext.split('_')
    if len(gt_labels) < 2:
        print("Warning: Expected at least 2 labels in filename (e.g., Giraffe_Lion.png)")
    
    info_maps = {}
    for label in gt_labels:
        print(f"Computing information map for label: {label}")
        try:
            label_idx = labels.index(label)
        except ValueError:
            print(f"Label '{label}' not found in the provided labels list.")
            continue
        sal_map = compute_pixel_level_importance(model, img_tensor, label_idx, reg_params)
        info_maps[label] = sal_map.detach().cpu().numpy()
    
    # Display the original image and information maps
    num_plots = len(info_maps) + 1
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    for i, (label, info_map) in enumerate(info_maps.items(), start=1):
        axes[i].imshow(info_map, cmap='hot')
        axes[i].set_title(f"Info Map: {label}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()
    
    info_map_path = "info_map.png"
    fig.savefig(info_map_path)
    print(f"Information map saved to {info_map_path}")


# =============================================================================
# Main Block
# =============================================================================
def main():
    # --- Load Data ---
    #text_dir, labels, X_valid, Y_valid = get_valid_data()
    #text_dir, labels, X_valid, Y_valid = get_valid_data()

    #X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    #Y_valid_tensor = torch.tensor(Y_valid, dtype=torch.long)
    #valid_dataset = TensorDataset(X_valid_tensor, Y_valid_tensor)
    #valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    #X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    #Y_valid_tensor = torch.tensor(Y_valid, dtype=torch.long)
    #valid_dataset = TensorDataset(X_valid_tensor, Y_valid_tensor)
    #valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    config = load_config('config.json')
    labels = config.get('labels', [])
    config = load_config('config.json')
    labels = config.get('labels', [])
    # --- Load Model Checkpoint ---
    epoch = 0 
    PATH = f"checkpoints/checkpoint_59epoch_0.9599acc_0.9446valacc_18c.pth"
    print(f"Checkpoint saved to {PATH}")
    checkpoint = torch.load(PATH, map_location=device)
    
    num_classes = len(labels)
    model = AnimalClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Animal model loaded and set to evaluation mode.")
    print("Attributes:", f"epoch: {checkpoint['epoch']}, epoch_loss: {checkpoint['epoch_loss']}, val_loss: {checkpoint['val_loss']}, epoch_acc: {checkpoint['epoch_acc']}, val_acc: {checkpoint['val_acc']}")
    
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # --- Initialize Regularization Parameters ---
    reg_params = RegParameters()
    reg_params.estimation = 'var'  # Use variance-based estimation
    print("Regularization parameters initialized. Using variance-based estimation.")
    
    # Example usage:
    IMAGE_PATH = "images2explain/Giraffe_Lion.png"
    generate_information_map(IMAGE_PATH, model, labels, reg_params)
    #generate_saliency_map(IMAGE_PATH, model, labels, reg_params)
    generate_information_map(IMAGE_PATH, model, labels, reg_params)
    #generate_saliency_map(IMAGE_PATH, model, labels, reg_params)

    stop
    
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
