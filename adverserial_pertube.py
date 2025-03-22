# %%
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
import gc
from importance import compute_per_sample_gradient, load_config, compute_importance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
def prepare_image(image_path: str) -> torch.Tensor:
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
    
    return img_tensor

def get_labels(image_path: str) -> List[str]:
    # Parse the filename to extract ground truth labels
    base_name = os.path.basename(image_path)
    name_no_ext, _ = os.path.splitext(base_name)
    gt_labels = name_no_ext.split('_')[:2]
    if len(gt_labels) < 2:
        print("Warning: Less than 2 labels found in the filename. Expected format: Label1_Label2.png")

    return gt_labels

# %%
def get_model_labels() -> List[str]:
    config = load_config('config.json')
    labels = config.get('labels', [])
    return labels

def load_model(model_path: str, labels: List[str]) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location=device)
    model = AnimalClassifier(num_classes=len(labels))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

# %%
def get_model_prediction(image_path: str, model: torch.nn.Module, labels: list, noise: torch.Tensor, epsilon: float) -> None:
    img_tensor = prepare_image(image_path)
    noised_img = img_tensor + epsilon * noise
    logits = model(noised_img)
    probs = F.softmax(logits, dim=1)
    _, pred_idx = torch.max(probs, 1)
    pred_label = labels[pred_idx.item()]
    print(f"Probabilities: {probs}")
    print(f"Predicted label: {pred_label}")

def get_saliency_map(image_path: str, model: torch.nn.Module, labels: list, reg_params: RegParameters) -> None:
    """
    The saliency map is computed by calculating the gradient of the softmax probability 
    (for the chosen label) with respect to the input image. The gradient is then reduced 
    by taking the maximum absolute value across channels.
    
    :param image_path: Path to the image file.
    :param model: The classification model.
    :param labels: List of valid labels. The image filename must contain two labels 
                   (separated by an underscore) that are present in this list.
    :param reg_params: Regularization parameters (used to set the estimation method and number of samples).
    :return: saliency maps for each label in the image filename.
    """
    img_tensor = prepare_image(image_path)
    gt_labels = get_labels(image_path)
    print(f"Ground truth labels: {gt_labels}")

    saliency_maps = {}
    for label in gt_labels:
        try:
            label_idx = labels.index(label)
        except ValueError:
            print(f"Label '{label}' not found in the provided labels list.")
            continue

        img_tensor.requires_grad_(True)
        print("Model Logits:", model(img_tensor))
        
        # Compute the per-sample gradient of the softmax probability for the given label
        grad_tensor = compute_per_sample_gradient(model, img_tensor, label_idx)  # shape: (1, 3, 224, 224)
        
        # Compute saliency map: take the max absolute gradient across channels, resulting in (1, 224, 224)
        sal_map = torch.max(torch.abs(grad_tensor), dim=1)[0]  
        sal_map = torch.max(grad_tensor, dim=1)[0]
        saliency_maps[label] = sal_map.squeeze().detach().cpu().numpy()

    return saliency_maps, gt_labels

# %%
def compute_subset_importance_adverserial(model: nn.Module, label_idx: int, reg_params: RegParameters, noise: torch.Tensor, epsilon: float, image_path: str, n_pertube: int =10) -> float:
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
    img_tensor = prepare_image(image_path)
    pertube_images = torch.cat([img_tensor + torch.randn_like(img_tensor) for _ in range(n_pertube)], dim=0)
    print(f"Perturbed images shape: {pertube_images.shape}")
    noise_signs = torch.sign(torch.Tensor(noise))
    flag = torch.all(noise_signs == 1)
    print(f"Flag: {flag}")
    print(f"Noise signs: {noise_signs}")
    noised_pertube_images = torch.cat([img_tensor + torch.randn_like(img_tensor) * noise_signs for _ in range(n_pertube)], dim=0)
    print(f"Noised perturbed images shape: {noised_pertube_images.shape}")
    batch = torch.cat([pertube_images, noised_pertube_images], dim=0)
    print("Batch shape:", batch.shape)
    #print('images shape:', images.shape)
    #pertub_images = Perturbation.perturb_tensor_subset(images, pixels, reg_params.n_samples)
    #print(f"Perturbed images shape: {pertub_images.shape}")
    per_sample_grad = compute_per_sample_gradient(model, batch, label_idx)
    print(f"Per-sample gradient shape: {per_sample_grad.shape}")
    batch.requires_grad_(True)
    importance = compute_importance(per_sample_grad, n_pertube, estimation=reg_params.estimation)
    print(f"Importance: {importance}")    
    #print_memory_usage()
    batch = batch.detach()#.clone()  # Stop tracking grads to avoid OOM
    per_sample_grad = per_sample_grad.detach()#.clone()
    #print_memory_usage()

    del pertub_images, per_sample_grad  # Delete the intermediate tensors to free up memory
    #torch.cuda.empty_cache()  # Clear GPU memory
    #gc.collect()  # Run garbage collection
    return importance

# %%
def main():
    PATH = f"checkpoints/checkpoint_59epoch_0.9599acc_0.9446valacc_18c.pth"

    labels = get_model_labels()
    model = load_model(PATH, labels)

    reg_params = RegParameters()
    reg_params.estimation = 'var'  # Use variance-based estimation

    # Example usage:
    #IMAGE_PATH = "images2explain/Giraffe_Lion.png" 
    IMAGE_PATH =  "images2explain/Horse_Zebra.png"
    #IMAGE_PATH =  "images2explain/Zebra_Lion_3.png"
    #IMAGE_PATH =  "images2explain/Lizard_Lizard_0.png"
    #IMAGE_PATH =  "images2explain/Eagle_Deer_0.png"
    #generate_information_map(IMAGE_PATH, model, labels, reg_params)
    saliency_maps, keys = get_saliency_map(IMAGE_PATH, model, labels, reg_params)

    epsilon = 20000
    label_idx = 14

    get_model_prediction(IMAGE_PATH, model, labels, saliency_maps[keys[0]], epsilon)
    get_model_prediction(IMAGE_PATH, model, labels, saliency_maps[keys[0]], 0)

    compute_subset_importance_adverserial(model, label_idx, reg_params, saliency_maps[keys[0]] ,epsilon, IMAGE_PATH) 


if __name__ == "__main__":
    main()