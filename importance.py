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
from typing import List , Optional 
import json
import matplotlib.pyplot as plt
import functorch
import psutil
import gc
import time
import torch
import torch.nn.functional as F

#from functorch import vmap, grad

# Automatically set the device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Utility Functions
# =============================================================================
# approx_importance.py

import torch
import torch.nn.functional as F
import functorch
from importance import *  # import existing functions if needed

def compute_per_sample_gradient_approx(model: torch.nn.Module, original_image: torch.Tensor, 
                                       perturbed_images: torch.Tensor, label_idx: int) -> torch.Tensor:
    """
    Approximates the gradient of the softmax probability for a specified label with respect to each 
    perturbed image, using a first-order Taylor expansion. The perturbed images are assumed to be of the form:
        x_perturbed = original_image + δ
    where δ is small and nonzero only at the targeted pixel(s).
    
    The approximation is:
        grad f(x + δ) ≈ grad f(x) + H(x)·δ
    where H(x)·δ is computed efficiently using functorch.jvp.

    :param model: The neural network model.
    :param original_image: The unperturbed image tensor of shape (C, H, W) with requires_grad=True.
    :param perturbed_images: A batch of perturbed images of shape (B, C, H, W). These should be produced 
                             by adding small noise only at the pixels of interest.
    :param label_idx: The index of the label for which to compute the gradient.
    :return: A tensor of shape (B, C, H, W) with the approximated gradient for each perturbed image.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # ensure all GPU work is done before timing
    
    start = time.perf_counter()
    # Define f on a single image: returns the softmax probability for the given label.
    def f(x: torch.Tensor) -> torch.Tensor:
        out = model(x.unsqueeze(0))              # shape: (1, num_classes)
        out_sm = F.softmax(out, dim=1)             # shape: (1, num_classes)
        return out_sm[0, label_idx]                # scalar

    # Compute the gradient at the original image (g_orig).
    grad_f = functorch.grad(f)
    g_orig = grad_f(original_image)  # shape: (C, H, W)

    # For each perturbed image, compute δ = (x_perturbed - original_image).
    # original_image has shape (C, H, W); we unsqueeze to (1, C, H, W) to broadcast.
    delta = perturbed_images - original_image.unsqueeze(0)  # shape: (B, C, H, W)

    # Define a helper that computes the Hessian-vector product (H(x)·δ) for a single δ.
    def hvp(delta_single: torch.Tensor) -> torch.Tensor:
        # functorch.jvp computes (f(x), directional derivative of f at x in direction δ)
        # Here we compute it for grad_f, so that the directional derivative equals H(x)·δ.
        _, jvp_val = functorch.jvp(grad_f, (original_image,), (delta_single,))
        return jvp_val  # shape: (C, H, W)
    
    # Vectorize the hvp function over the batch dimension.
    vectorized_hvp = functorch.vmap(hvp)

    end = time.perf_counter()
    print(f"1st Computation took {end - start:.4f} seconds")
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # ensure all GPU work is done before timing
    
    start = time.perf_counter()


    # Vectorize hvp over the batch dimension.
    approx_hvp = vectorized_hvp(delta)  # shape: (B, C, H, W)

    # The approximated gradient for each perturbed image:
    approx_grad = g_orig.unsqueeze(0) + approx_hvp  # shape: (B, C, H, W)
    
    end = time.perf_counter()
    print(f"2nd Computation took {end - start:.4f} seconds")
    
    return approx_grad

# You might then override your original compute_per_sample_gradient with the approximate version.
# For example, if you want to switch based on a flag, you could do:

def _compute_per_sample_gradient(model: torch.nn.Module, images: torch.Tensor, label_idx: int, 
                                approx: bool = True) -> torch.Tensor:
    """
    Computes the gradient of the softmax probability for a specified label with respect to each input image.
    
    If approx==True, it uses a first-order Taylor expansion to approximate the gradient for each perturbed image
    by reusing the gradient computed at the original image.
    
    :param model: The neural network model.
    :param images: A batch of perturbed images of shape (B, C, H, W). Must have requires_grad=True.
    :param label_idx: The index of the label for which to compute the gradient.
    :param approx: Whether to use the approximation.
    :return: A tensor of shape (B, C, H, W) with the gradient for each image.
    """
    # If no approximation, fall back on the original method.
    if not approx:
        if not torch.cuda.is_available():
            grad_f = torch.func.grad(lambda x: F.softmax(model(x.unsqueeze(0)), dim=1)[0, label_idx])
            return torch.vmap(grad_f)(images)
        else:
            grad_f = functorch.grad(lambda x: F.softmax(model(x.unsqueeze(0)), dim=1)[0, label_idx])
            return functorch.vmap(grad_f)(images)
    
    # Otherwise, assume that all perturbed images in 'images' were generated from the same original image.
    # Here, we extract the original image from the first sample.
    original_image = images[0].detach()
    # Compute approximate gradients.
    per_sample_grad = compute_per_sample_gradient_approx(model, original_image, images, label_idx)
    
    return per_sample_grad


def compute_per_sample_gradient(model: torch.nn.Module, images: torch.Tensor, label_idx: int) -> torch.Tensor:
    """
    Computes the gradient of the softmax probability for a specified label with respect 
    to each input image and prints the computation time.
    
    :param model: The neural network model.
    :param images: A batch of images of shape (B, C, H, W). Must have requires_grad=True.
    :param label_idx: The index of the label for which to compute the gradient.
    :return: A tensor of shape (B, C, H, W) with the gradient for each image.
    """
    #if torch.cuda.is_available():
    #    torch.cuda.synchronize()  # ensure all GPU work is done before timing
    
    #start = time.perf_counter()

    def f(x: torch.Tensor) -> torch.Tensor:
        # x has shape (C, H, W); add batch dimension
        out = model(x.unsqueeze(0))              # shape: (1, num_classes)
        out_sm = F.softmax(out, dim=1)             # shape: (1, num_classes)
        return out_sm[0, label_idx]                # return the probability for the specified label

    if not torch.cuda.is_available():
        grad_f = torch.func.grad(f)
        per_sample_grad = torch.vmap(grad_f)(images)  # shape: (B, C, H, W)
    else:
        grad_f = functorch.grad(f)
        per_sample_grad = functorch.vmap(grad_f)(images)  # shape: (B, C, H, W)
    
    #if torch.cuda.is_available():
    #    torch.cuda.synchronize()  # wait for GPU work to finish
    
    #end = time.perf_counter()
    #print(f"Batch size {images.shape[0]}: Computation took {end - start:.4f} seconds")
    
    return per_sample_grad

def compute_softmax_prob(model: torch.nn.Module, images: torch.Tensor, label_idx: int) -> torch.Tensor:
    """
    Computes the softmax probability for a specified label for each input image.
    
    :param model: The neural network model.
    :param
    :param label_idx: The index of the label for which to compute the probability.
    :return: A tensor of shape (B,) with the softmax probability for each image.
    """
    # Compute the output logits for each image
    logits = model(images)
    
    # Compute the softmax probabilities for the specified label
    softmax_probs = F.softmax(logits, dim=1)[:, label_idx]
    
    return softmax_probs



def compute_importance(gradients: torch.Tensor,n_samples: int ,estimation: str = 'var', softmax_prob : Optional[torch.Tensor] = None) -> float:
    """
    Computes an importance measure from the gradients using the Regularization helper.
    
    :param gradients: Tensor of gradients of shape (B, C, H, W).
    :param estimation: Estimation method to use ('var' or 'ent').
    :param n_samples: Number of samples used for the perturbation.
    :param softmax_prob: The softmax probability for the specified label for each input image.
    :return: A scalar importance measure.
    """
    importance = Regularization.get_importance_by_estimation(gradients, n_samples, estimation, softmax_prob)
    importance = importance.detach().clone()  # Ensure no graph connection
    
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
    Loads an image, extracts ground-truth labels from its filename (format: "Label1_Label2.png"),
    computes the per-pixel gradient (saliency map) for each label, and displays the original image 
    with the saliency maps overlayed.

    The saliency map is computed by taking the gradient of the softmax probability for the label 
    with respect to the input image and then reducing it by taking the maximum absolute value 
    across channels.
    
    :param image_path: Path to the image file.
    :param model: The classification model.
    :param labels: List of valid labels.
    :param reg_params: Regularization parameters.
    :return: None. Displays the images.
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
    gt_labels = name_no_ext.split('_')[:2]
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

        output = model(img_tensor)
        print("The model prediction is:", output)
        
        # Compute per-sample gradient for the given label
        grad_tensor = compute_per_sample_gradient(model, img_tensor, label_idx)  # shape: (1, 3, 224, 224)
        
        # Compute saliency map: take the maximum absolute gradient across channels
        sal_map = torch.max(torch.abs(grad_tensor), dim=1)[0]   # shape: (1, 224, 224)
        #sal_map = torch.max(grad_tensor, dim=1)[0]
        
        # Remove the extra batch dimension: shape becomes (224, 224)
        sal_map = sal_map.squeeze(0)
        
        # Normalize the saliency map to [0, 1]
        sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)
        
        # Convert to uint8 (required for cv.applyColorMap)
        sal_map_uint8 = (sal_map.detach().cpu().numpy() * 255).astype(np.uint8)
        
        # Apply the JET colormap (expects a single-channel 8-bit image)
        colored_sal_map = cv.applyColorMap(sal_map_uint8, cv.COLORMAP_JET)
        
        # Overlay the saliency map on the original image
        alpha = 0.5  # Transparency factor
        overlay = cv.addWeighted(img, alpha, colored_sal_map, 1 - alpha, 0)
        
        saliency_maps[label] = overlay
    
    # Plot the original image and saliency maps side-by-side
    num_plots = len(saliency_maps) + 1
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    i = 1
    for label, overlay in saliency_maps.items():
        axes[i].imshow(overlay)
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
    pertub_images = Perturbation.perturb_tensor_subset(images, pixels, reg_params.n_samples)
    per_sample_grad = compute_per_sample_gradient(model, pertub_images, label_idx)
    pertub_images.requires_grad_(True)

    if reg_params.estimation == 'ent':
        softmax_prob = compute_softmax_prob(model, pertub_images, label_idx)
        importance = compute_importance(per_sample_grad, reg_params.n_samples, estimation=reg_params.estimation, softmax_prob=softmax_prob)
    else:
        importance = compute_importance(per_sample_grad, reg_params.n_samples, estimation=reg_params.estimation)
    
    pertub_images = pertub_images.detach()#.clone()  # Stop tracking grads to avoid OOM
    per_sample_grad = per_sample_grad.detach()#.clone()

    del pertub_images, per_sample_grad  # Delete the intermediate tensors to free up memory
    return importance

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory Usage: {process.memory_info().rss / 1e6} MB")  # Convert to MB
    torch.cuda.empty_cache()

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
    saliency_map = torch.zeros((num_pixels,), device=image.device)
    with open('pixel_importance.txt', 'w') as f:
        for idx in range(num_pixels):
            imp = compute_subset_importance(model, image, [idx], label_idx, reg_params)
            saliency_map[idx] = imp
            f.write(f'idx: {idx}, pixel importance: {imp}\n')
            
            del imp  # Delete the importance tensor to free up memory
            torch.cuda.empty_cache()
            gc.collect()
            
            if idx % 100 == 0 and idx > 0:
                print(f"Processed {idx} pixels.")
                print_memory_usage()
        
    return saliency_map.view(H, W)


def load_pixel_info(path: str, saliency_map: torch.Tensor):
    """
    Loads already computed pixel information scores from the file and updates the saliency map.
    Returns the updated saliency map and the starting index for new computations.
    
    :param path: File path containing previously computed pixel importance scores.
    :param saliency_map: A tensor to update with loaded values.
    :return: Updated saliency map and the starting index for further computations.
    """
    start_idx = 0
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split(', ')
                if len(parts) == 2:
                    idx = int(parts[0].split(': ')[1])
                    imp = float(parts[1].split(': ')[1])
                    saliency_map[idx] = imp
                    start_idx = max(start_idx, idx + 1)
    return saliency_map, start_idx


def compute_pixel_level_importance_batch(model: nn.Module, image: torch.Tensor, label_idx: int, 
                                           reg_params: RegParameters, save_path: str = 'pixel_info.txt', batch_size: int = 64) -> torch.Tensor:
    """
    Computes an information map for a single image for a specific label using batched perturbations.
    
    Processes pixels in batches and appends the computed importance values to the file at save_path.
    
    :param model: The neural network model.
    :param image: A single image tensor of shape (1, 3, H, W) with requires_grad=True.
    :param label_idx: The index of the label for which to compute the importance.
    :param reg_params: Regularization parameters.
    :param save_path: File path to which pixel importance results will be appended.
    :param batch_size: Number of pixels to process in one batch.
    :return: A tensor of shape (H, W) with the importance score for each pixel.
    """
    # image shape: (1, 3, H, W)
    _, C, H, W = image.shape
    num_pixels = H * W
    saliency_map = torch.zeros((num_pixels,), device=image.device)
    
    # Load existing pixel information
    saliency_map, start_idx = load_pixel_info(save_path, saliency_map)

    for batch_start in range(start_idx, num_pixels, batch_size):
        batch_indices = list(range(batch_start, min(batch_start + batch_size, num_pixels)))
        
        # Compute the importance values for the current batch.
        importance_batch = compute_subset_importance(model, image, batch_indices, label_idx, reg_params)  # expected shape: (batch_size,)
        saliency_map[batch_start:batch_start + len(batch_indices)] = importance_batch
        
        # Create lines for this batch.
        pixel_importance_lines = []
        for idx, imp in zip(batch_indices, importance_batch):
            pixel_importance_lines.append(f'idx: {idx}, pixel importance: {imp.item()}\n')
        
        # Append current batch's results to the file.
        with open(save_path, 'a') as f:
            f.writelines(pixel_importance_lines)
        
        # Optionally print progress and perform cleanup.
        if (batch_start + len(batch_indices)) % (batch_size) == 0:
            print(f"Processed {batch_start + len(batch_indices)} / {num_pixels} pixels.")
            print_memory_usage()
            torch.cuda.empty_cache()
            gc.collect()

    return saliency_map.view(H, W)




def generate_information_map(image_path: str, model: nn.Module, labels: list, 
                             reg_params: RegParameters, batch_flag: bool=True) -> None:
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
    img, img_tensor = prepare_image(image_path)
    img_tensor.requires_grad_(True)
    
    # 2) Parse labels from filename
    gt_labels = get_labels(image_path)
    name_no_ext = gt_labels[0] + "_" + gt_labels[1]

    info_maps = {}
    for label in gt_labels:
        print(f"Computing information map for label: {label}")
        PIXEL_INFO_PATH = f"pixel_info_{name_no_ext}_" 
        PIXEL_INFO_PATH = PIXEL_INFO_PATH + label + ".txt"   
    
        try:
            label_idx = labels.index(label)
        except ValueError:
            print(f"Label '{label}' not found in the provided labels list.")
            continue
        if batch_flag:
            sal_map = compute_pixel_level_importance_batch(model, img_tensor, label_idx, reg_params, PIXEL_INFO_PATH)
        else:
            sal_map = compute_pixel_level_importance(model, img_tensor, label_idx, reg_params)
        info_maps[label] = sal_map.detach().cpu().numpy()
        
        del sal_map
        torch.cuda.empty_cache()
        gc.collect()
    
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
    
    del info_maps, img_tensor
    torch.cuda.empty_cache()
    gc.collect()


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
    
    return img, img_tensor

def get_labels(image_path: str) -> List[str]:
    # Parse the filename to extract ground truth labels
    base_name = os.path.basename(image_path)
    name_no_ext, _ = os.path.splitext(base_name)
    gt_labels = name_no_ext.split('_')[:2]
    if len(gt_labels) < 2:
        print("Warning: Less than 2 labels found in the filename. Expected format: Label1_Label2.png")

    return gt_labels

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

# =============================================================================
# Main Block
# =============================================================================
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
    generate_saliency_map(IMAGE_PATH, model, labels, reg_params)

    stop

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
