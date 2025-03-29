import os
import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from captum.attr import IntegratedGradients, NoiseTunnel
from importance import prepare_image, get_model_labels, load_model, get_labels
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients, NoiseTunnel, Saliency


def generate_importance_map_vargrad_overlay(
    image_path: str,
    model: nn.Module,
    target_label: int,
    target_label_name: str,
    nt_samples: int = 4,
    stdevs: float = 0.02
) -> np.ndarray:
    """
    Uses Captum's IntegratedGradients with NoiseTunnel (nt_type='vargrad') to compute an importance
    (attribution) map for the input image.
    
    :param image_path: Path to the image file.
    :param model: The model (should be in eval mode) for which to compute attributions.
    :param target_label: The target class index for which to compute attributions.
    :param target_label_name: A string used to name the saved image.
    :param nt_samples: Number of noisy samples to average over.
    :param stdevs: Standard deviation of the Gaussian noise added.
    :return: An overlay image (numpy array) showing the original image blended with the heatmap.
    """
    # 1) Load & preprocess the image.
    orig_img, img_tensor = prepare_image(image_path)
    img_tensor.requires_grad = True

    # 2) Create a baseline (here a zero tensor of the same shape as the input).
    baseline = img_tensor # torch.zeros_like(img_tensor)

    # 3) Instantiate IntegratedGradients and wrap with NoiseTunnel.
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    
    # 4) Compute attributions using NoiseTunnel with vargrad smoothing.
    attributions, delta = nt.attribute(
        img_tensor,
        nt_type='vargrad',
        stdevs=stdevs,
        nt_samples=nt_samples,
        baselines=baseline,
        target=target_label,
        return_convergence_delta=True
    )
    
    # 5) Process the attributions:
    # Convert to numpy, average over channels, and normalize.
    attr_map = attributions.detach().cpu().numpy()[0]  # shape: (C, H, W)
    attr_map = np.mean(attr_map, axis=0)                # shape: (H, W)
    attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
    attr_map_uint8 = (attr_map * 255).astype(np.uint8)
    
    # 6) Generate a colored heatmap using OpenCV.
    colored_attr_map = cv.applyColorMap(attr_map_uint8, cv.COLORMAP_JET)
    colored_attr_map = cv.resize(colored_attr_map, (orig_img.shape[1], orig_img.shape[0]))
    
    # 7) Overlay the heatmap on the original image.
    overlay = cv.addWeighted(orig_img, 0.3, colored_attr_map, 0.7, 0) 
    
    # 8) Create a figure, display and save the result.
    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(cv.cvtColor(overlay, cv.COLOR_BGR2RGB))
    ax.set_title(f"VarGrad Attribution Map {target_label_name}")
    ax.axis("off")
    
    # Ensure the save directory exists.
    save_dir = 'vargrad'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    info_map_path = os.path.join(save_dir, f"{target_label_name}.png")
    fig.savefig(info_map_path)
    print(f"Attribution map saved to {info_map_path}")
    plt.show()
    
    return overlay

def _generate_importance_map_vargrad(
    image_path: str,
    model: nn.Module,
    target_label: int,
    target_label_name: str,
    nt_samples: int = 4,
    stdevs: float = 0.02
) -> np.ndarray:
    """
    Uses Captum's IntegratedGradients with NoiseTunnel (nt_type='vargrad') to compute an attribution
    map for the input image, then applies the 'twilight_shifted' colormap to display and save only the 
    colored attribution map.
    
    :param image_path: Path to the image file.
    :param model: The model (should be in eval mode) for which to compute attributions.
    :param target_label: The target class index for which to compute attributions.
    :param target_label_name: A string used to name the saved image.
    :param nt_samples: Number of noisy samples to average over.
    :param stdevs: Standard deviation of the Gaussian noise added.
    :return: A colored attribution map (numpy array) in RGB format.
    """
    # 1) Load & preprocess the image.
    orig_img, img_tensor = prepare_image(image_path)
    img_tensor.requires_grad = True

    # 2) Create a baseline (a zero tensor of the same shape).
    baseline = img_tensor #torch.zeros_like(img_tensor)

    # 3) Instantiate IntegratedGradients and wrap with NoiseTunnel.
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    
    # 4) Compute attributions using NoiseTunnel with vargrad smoothing.
    attributions, delta = nt.attribute(
        img_tensor,
        nt_type='vargrad',
        stdevs=stdevs,
        nt_samples=nt_samples,
        baselines=baseline,
        target=target_label,
        return_convergence_delta=True
    )
    
    # 5) Process the attributions:
    # Convert to numpy, average over channels, and normalize.
    attr_map = attributions.detach().cpu().numpy()[0]  # shape: (C, H, W)
    attr_map = np.mean(attr_map, axis=0)                # shape: (H, W)
    attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
    
    # 6) Apply matplotlib's colormap 'twilight_shifted'
    cmap = plt.get_cmap('twilight_shifted')
    # cmap returns an RGBA image; scale to [0, 255]
    colored_attr_map = cmap(attr_map)                  # shape: (H, W, 4)
    # Remove the alpha channel.
    colored_attr_map = np.delete(colored_attr_map, 3, axis=2)  
    colored_attr_map = (colored_attr_map * 255).astype(np.uint8)
    # Resize to match the original image.
    colored_attr_map = cv.resize(colored_attr_map, (orig_img.shape[1], orig_img.shape[0]))
    
    # 7) Display and save only the attribution map.
    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(colored_attr_map)
    ax.set_title(f"VarGrad Attribution Map {target_label_name} ")
    ax.axis("off")
    
    # Ensure the save directory exists.
    save_dir = 'vargrad'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    info_map_path = os.path.join(save_dir, f"{target_label_name}_attr.png")
    fig.savefig(info_map_path)
    print(f"Attribution map saved to {info_map_path}")
    
    plt.show()
    
    return colored_attr_map


def generate_importance_map_vargrad(
    image_path: str,
    model: torch.nn.Module,
    labels: list,
    nt_samples: int = 20,
    stdevs: float = 0.02
) -> None:
    """
    Computes vargrad-based attributions for each ground-truth label found in the image filename
    and plots a grid of subplots showing the original image alongside the vargrad overlays.
    
    :param image_path: Path to the image file.
    :param model: The model (in eval mode) for which to compute attributions.
    :param labels: List of valid class labels.
    :param nt_samples: Number of noisy samples for vargrad.
    :param stdevs: Standard deviation for the Gaussian noise added.
    """
    # 1) Load & preprocess the image.
    # Assume prepare_image returns the original image (BGR format) and a preprocessed tensor.
    orig_img, img_tensor = prepare_image(image_path)
    img_tensor.requires_grad_(True)
    
    # 2) Parse ground-truth labels from the filename.
    gt_labels = get_labels(image_path)  # e.g., returns ['Giraffe', 'Lion']
    
    info_maps = {}
    # 3) For each label, compute vargrad overlay using IntegratedGradients wrapped in NoiseTunnel.
    for label in gt_labels:
        try:
            label_idx = labels.index(label)
        except ValueError:
            print(f"Label '{label}' not found in the provided labels list.")
            continue

        # Use the input itself as the baseline (alternatively, you might use zeros)
        baseline = img_tensor
        
        '''
        #ig = IntegratedGradients(model)
        #nt = NoiseTunnel(ig)
        attributions, _ = nt.attribute(
            img_tensor,
            nt_type='vargrad',
            stdevs=stdevs,
            nt_samples=nt_samples,
            baselines=baseline,
            target=label_idx,
            return_convergence_delta=True
        )
        '''
        saliency = Saliency(model)
        nt = NoiseTunnel(saliency)
        attributions= nt.attribute(
            img_tensor,
            nt_type='smoothgrad_sq',
            stdevs=stdevs,
            nt_samples=nt_samples,
            target=label_idx
        )
        
        # 4) Process attributions: average over channels and normalize.
        attr_map = attributions.detach().cpu().numpy()[0]  # shape: (C, H, W)
        attr_map = np.mean(attr_map, axis=0)                # shape: (H, W)
        attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
        attr_map_uint8 = (attr_map * 255).astype(np.uint8)
        colored_attr_map = cv.applyColorMap(attr_map_uint8, cv.COLORMAP_JET)
        colored_attr_map = cv.resize(colored_attr_map, (orig_img.shape[1], orig_img.shape[0]))
        
        # 5) Overlay the heatmap on the original image.
        overlay = cv.addWeighted(orig_img, 0.3, colored_attr_map, 0.7, 0)
        info_maps[label] = overlay

    # 6) Plot the results: original image plus one overlay per label.
    num_plots = len(info_maps) + 1
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    # Plot original image.
    #axes[0].imshow(cv.cvtColor(orig_img, cv.COLOR_BGR2RGB))
    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Plot each vargrad overlay.
    for i, (label, overlay) in enumerate(info_maps.items(), start=1):
        axes[i].imshow(cv.cvtColor(overlay, cv.COLOR_BGR2RGB))
        axes[i].set_title(f"VarGrad Map: {label}")
        axes[i].axis("off")
    
    plt.tight_layout()
    
    # 7) Save the figure.
    save_dir = "vargrad_sal_maps"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "vargrad_info_map.png")
    fig.savefig(save_path)
    print(f"VarGrad-based information map saved to {save_path}")
    plt.show()

# Example main usage:
def main():
    #PATH = "checkpoints/checkpoint_59epoch_0.9599acc_0.9446valacc_18c.pth"
    #PATH = "checkpoints/checkpoint_47epoch_0.9576acc_0.9529valacc_4c.pth"
    PATH = "checkpoints/checkpoint_86epoch_0.9327trainF1_0.9344valF1_4c.pth"

    #IMAGE_PATH = "images2explain/Horse_Zebra.png"
    IMAGE_PATH = "images2explain/Giraffe_Lion.png" 
    #IMAGE_PATH = "images2explain/Zebra_Lion.png"
    #IMAGE_PATH = "images2explain/Lion_Horse.png"
    
    labels = get_model_labels()  
    model = load_model(PATH, labels) 
    
    generate_importance_map_vargrad(
        image_path=IMAGE_PATH,
        model=model,
        labels=labels,
        nt_samples=20,
        stdevs=1.0
    )


if __name__ == "__main__":
    main()
