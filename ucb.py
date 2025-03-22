import numpy as np
import torch
import torch.nn as nn
import csv
from importance import prepare_image, get_model_labels, load_model, get_labels, compute_subset_importance
from regularization import RegParameters
from typing import List, Optional
import cv2 as cv
import os
import gc
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_importance_batch(
    model: nn.Module,
    image: torch.Tensor,
    label_idx: int,
    reg_params: RegParameters,
    save_path: str = 'pixel_info_sublist.txt',
    batch_size: int = 64,
    pixel_list: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Computes an importance score for a *subset of pixels* in a single image for a specific label,
    using batched perturbations. If no pixel_list is provided, it processes all pixels.

    :param model: The neural network model (eval mode).
    :param image: A single image tensor of shape (1, 3, H, W) with requires_grad=True.
    :param label_idx: The class index for which to compute pixel importance.
    :param reg_params: Regularization or sampling parameters (e.g., n_samples).
    :param save_path: File path to which the computed per-pixel results will be appended.
    :param batch_size: Number of pixels to process in one batch for GPU efficiency.
    :param pixel_list: A list of pixel indices to process. If None, we default to all pixels.
    :return: A 1D tensor of size len(pixel_list) or size (H*W) if pixel_list=None,
             containing the computed importance for each pixel in the requested subset.
    """
    # 1) Infer shape and number of pixels
    _, C, H, W = image.shape
    total_pixels = H * W
    flag=False

    # 2) If no pixel_list provided, default to all pixels in [0..H*W-1]
    if pixel_list is None:
        pixel_list = list(range(total_pixels))
        flag = True

    # 3) Prepare an output tensor for the subset, shape [len(pixel_list)]
    device_ = image.device
    subset_size = len(pixel_list)
    saliency_subset = torch.zeros((subset_size,), device=device_)

    # 4) Process the subset in batches
    for start_idx in range(0, subset_size, batch_size):
        end_idx = min(start_idx + batch_size, subset_size)
        batch_indices = pixel_list[start_idx:end_idx]

        # 4a) Pass this batch of pixels to the "compute_subset_importance" function
        #     which should uniformly sample each pixel in 'batch_indices' and return
        #     a 1D tensor of shape (len(batch_indices),)
        importance_batch = compute_subset_importance(
            model=model,
            image=image,
            pixel_indices=batch_indices,
            label_idx=label_idx,
            reg_params=reg_params,
            full_pertube=flag
        )  
        # e.g., shape: (len(batch_indices),)

        saliency_subset[start_idx : end_idx] = importance_batch

        # 4c) Append lines to the file (like the original function)
        #lines_to_write = []
        #for local_i, imp_val in zip(batch_indices, importance_batch):
        #    lines_to_write.append(f'idx: {local_i}, pixel importance: {imp_val.item()}\n')
        #with open(save_path, 'a') as f:
        #    f.writelines(lines_to_write)

    # 5) Return the importance for the requested subset.
    #    The caller can reshape if needed (e.g., (H, W) if the subset is all pixels).
    return saliency_subset


def log_csv_entry(csv_file, row):
    """
    Appends a single row (dict with known keys) to a CSV file.
    Keys should match the columns: 
      'ucb_iteration', 'idx', 'ucb_value', 'importance_value', 'n_idx', 'n_total'.
    """
    fieldnames = ['ucb_iteration', 'idx', 'ucb_value', 'importance_value', 'n_idx', 'n_total']
    write_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def do_initial_pass_uniform(
    model, image, label_idx, reg_params,
    global_batch_size_for_perturbations,
    n_init, save_path, num_pixels
):
    """
    1) Samples each pixel exactly n_init times (uniform).
    2) Returns an array of shape (num_pixels,) with the initial importance estimate.
    """
    # We'll assume there's a function `compute_importance_batch`
    # that can handle a 'pixel_list=None' to mean "all pixels".
    
    old_n_samples = reg_params.n_samples
    reg_params.n_samples = n_init
    
    # We pass pixel_list=None (or the entire range) if your function defaults to all pixels
    uniform_init_map = compute_importance_batch(
        model=model,
        image=image,
        label_idx=label_idx,
        reg_params=reg_params,
        save_path=save_path,
        batch_size=global_batch_size_for_perturbations,
        pixel_list=None  # or range(num_pixels)
    )
    
    # Restore original n_samples
    reg_params.n_samples = old_n_samples
    return uniform_init_map

def do_ucb_iteration_partialsort(
    iteration, top_percent, c, 
    importance_est, n_i, sum_n_i,
    compute_importance_batch,
    model, image, label_idx, reg_params, global_batch_size_for_perturbations,
    csv_path
):
    """
    Executes one iteration of partial-sort UCB:
      1) Compute UCB scores for all pixels.
      2) Partial-sort to pick top k = top_percent * total
      3) Sample those pixels (uniform approach).
      4) Update rolling average + sample counts
      5) Log to CSV.
    Returns: updated importance_est, n_i
    """
    num_pixels = len(n_i)
    k = max(1, int(top_percent * num_pixels))
    
    # 1) Compute UCB score
    ucb_scores = np.zeros(num_pixels, dtype=np.float32)
    for i in range(num_pixels):
        if n_i[i] == 0:
            # Large bonus to ensure exploration
            bonus = 1e6
        else:
            bonus = c * np.sqrt(np.log(sum_n_i) / n_i[i])
        ucb_scores[i] = importance_est[i] + bonus
    
    # 2) Partial-sort => top-k
    top_k_indices = np.argpartition(ucb_scores, -k)[-k:]
    
    # 3) Sample these top-k pixels with your uniform function
    importance_batch = compute_importance_batch(
        model=model,
        image=image,
        label_idx=label_idx,
        reg_params=reg_params,
        save_path="tmp_UCB_partial_sort.txt",  # or pass a unique name
        batch_size=global_batch_size_for_perturbations,
        pixel_list=top_k_indices
    )
    
    # 4) Weighted average update
    new_samples_each = reg_params.n_samples
    for local_i, idx in enumerate(top_k_indices):
        old_count = n_i[idx]
        new_count = old_count + new_samples_each
        old_imp = importance_est[idx]
        new_imp = importance_batch[local_i].item()
        # Weighted average
        importance_est[idx] = (old_imp * old_count + new_imp * new_samples_each) / new_count
        n_i[idx] = new_count
    
    # 5) Log results to CSV
    new_sum_n_i = np.sum(n_i)
    for idx in top_k_indices:
        row = {
            'ucb_iteration': iteration,
            'idx': int(idx),
            'ucb_value': float(ucb_scores[idx]),
            'importance_value': float(importance_est[idx]),
            'n_idx': int(n_i[idx]),
            'n_total': int(new_sum_n_i),
        }
        log_csv_entry(csv_path, row)
    
    return importance_est, n_i

def compute_pixel_importance_ucb_wrapper_partialsort(
    model: torch.nn.Module,
    image: torch.Tensor,
    label_idx: int,
    reg_params,
    ucb_iterations: int,
    top_percent: float,
    global_batch_size_for_perturbations: int,
    n_init: int,
    csv_path: str = "ucb_log.csv",
    uniform_log_path: str = "pixel_info_init.txt"
) -> torch.Tensor:
    """
      1) Do an initial uniform pass over all pixels with n_init samples each.
      2) Then run 'ucb_iterations' steps, partial-sorting to pick top-percent, 
         using the existing uniform-sampling function for each subset.
      3) Log each iteration's chosen pixel in a CSV with columns:
         [ucb_iteration, idx, ucb_value, importance_value, n_idx, n_total].
    
    :return: final importance map (H,W)
    """

    # Basic shape
    _, C, H, W = image.shape
    num_pixels = H * W
    
    # We track sample counts + importance
    n_i = np.zeros(num_pixels, dtype=np.int32)
    importance_est = np.zeros(num_pixels, dtype=np.float32)

    # (1) Initial pass (uniform)
    # Let the uniform function do its normal logic for all pixels:
    # We'll override n_samples in do_initial_pass_uniform to n_init for now.
    uniform_init_map = do_initial_pass_uniform(
        model=model,
        image=image,
        label_idx=label_idx,
        reg_params=reg_params,
        global_batch_size_for_perturbations=global_batch_size_for_perturbations,
        n_init=n_init,
        save_path=uniform_log_path,
        num_pixels=num_pixels
    )
    # uniform_init_map => (num_pixels,)
    # Fill n_i & importance_est
    for i in range(num_pixels):
        n_i[i] = n_init
        importance_est[i] = uniform_init_map[i].item()
    
    # (2) Main UCB loop
    for t in range(1, ucb_iterations + 1):
        sum_n_i = max(1, np.sum(n_i))
        importance_est, n_i = do_ucb_iteration_partialsort(
            iteration=t,
            top_percent=top_percent,
            c=reg_params.c,  # or similar param
            importance_est=importance_est,
            n_i=n_i,
            sum_n_i=sum_n_i,
            compute_pixel_level_importance_batch=compute_importance_batch,
            model=model,
            image=image,
            label_idx=label_idx,
            reg_params=reg_params,
            global_batch_size_for_perturbations=global_batch_size_for_perturbations,
            csv_path=csv_path
        )
        print(f"UCB iteration {t}/{ucb_iterations} done.")

    # Convert final to shape (H, W)
    final_map = torch.from_numpy(importance_est.reshape(H, W))
    return final_map


def generate_importance_map_ucb(
    image_path: str, 
    model: nn.Module, 
    labels: list, 
    reg_params,
    ucb_iterations: int = 5,
    top_percent: float = 0.1,
    batch_size_for_perturbations: int = 64,
    n_init: int = 3,
    csv_path: str = "ucb_log.csv"
) -> None:
    """
    Similar to the old `generate_information_map`, but uses the new UCB-based partial sort approach.
    For each label found in the filename, we compute a pixel-level importance map with:
      1) an initial uniform pass of n_init,
      2) a UCB outer loop for ucb_iterations steps, each partial-sorting top_percent of pixels.
    Finally, display and save the original image plus these saliency maps.

    :param image_path: Path to the image file (filename must contain labels separated by '_').
    :param model: The classification model in eval mode.
    :param labels: List of valid class labels.
    :param reg_params: A RegParameters instance containing e.g. c, n_samples, estimation, etc.
    :param ucb_iterations: Number of times to repeat the partial-sort UCB steps.
    :param top_percent: Fraction of pixels to pick each iteration.
    :param batch_size_for_perturbations: Batch size for calls to `compute_pixel_level_importance_batch`.
    :param n_init: Number of uniform perturbations per pixel in the initial pass.
    :param csv_path: Where to log each iteration's chosen pixels (in CSV format).
    """

    # 1) Load & preprocess the image
    img, img_tensor = prepare_image(image_path)
    img_tensor.requires_grad_(True)

    # 2) Parse labels from filename
    gt_labels = get_labels(image_path)
    name_no_ext = gt_labels[0] + "_" + gt_labels[1]

    # 3) For each ground-truth label in the filename, do the UCB approach
    info_maps = {}
    for label in gt_labels:
        print(f"[UCB] Computing information map for label: {label}")
        try:
            label_idx = labels.index(label)
        except ValueError:
            print(f"Label '{label}' not found in the provided labels list.")
            continue

        # We choose a separate text file for the initial pass log
        uniform_log_path = f"pixel_info_init_{name_no_ext}_{label}.txt"

        # 4) Call the new modular UCB-based function
        sal_map = compute_pixel_importance_ucb_wrapper_partialsort(
            model=model,
            image=img_tensor,
            label_idx=label_idx,
            reg_params=reg_params,
            ucb_iterations=ucb_iterations,
            top_percent=top_percent,
            global_batch_size_for_perturbations=batch_size_for_perturbations,
            n_init=n_init,
            csv_path=csv_path,
            uniform_log_path=uniform_log_path
        )
        info_maps[label] = sal_map.detach().cpu().numpy()

        # optional memory cleanup
        del sal_map
        torch.cuda.empty_cache()
        gc.collect()

    # 5) Display results
    num_plots = len(info_maps) + 1
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    for i, (label, info_map) in enumerate(info_maps.items(), start=1):
        axes[i].imshow(info_map, cmap='hot')
        axes[i].set_title(f"UCB Info Map: {label}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

    info_map_path = f"ucb_info_map_{name_no_ext}.png"
    fig.savefig(info_map_path)
    print(f"UCB-based information map saved to {info_map_path}")

    del info_maps, img_tensor
    torch.cuda.empty_cache()
    gc.collect()


def main():
    # Example usage
    PATH = f"checkpoints/checkpoint_59epoch_0.9599acc_0.9446valacc_18c.pth"

    labels = get_model_labels()
    model = load_model(PATH, labels)

    reg_params = RegParameters()
    reg_params.estimation = 'var'  # e.g. variance-based
    reg_params.n_samples = 2
    # you might also specify reg_params.c, reg_params.n_samples, etc.

    IMAGE_PATH = "images2explain/Horse_Zebra.png"

    # We'll run 3 UCB iterations, picking top 10%, with an initial pass of 2 samples/pixel
    generate_importance_map_ucb(
        image_path=IMAGE_PATH,
        model=model,
        labels=labels,
        reg_params=reg_params,
        ucb_iterations=1,
        top_percent=0.1,
        batch_size_for_perturbations=64,
        n_init=2,
        csv_path="ucb_log.csv"
    )
