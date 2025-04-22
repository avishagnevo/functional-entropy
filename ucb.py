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
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights


device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def ensure_dir_exists(file_path):
    """
    Ensures that the directory for the given file path exists.
    If it does not exist, it creates the directory.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
 
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
            writer.writerow(row)

def log_csv_header(csv_file):
    """
    Writes the header row to a CSV file.
    """
    ensure_dir_exists(csv_file)
    fieldnames = ['ucb_iteration', 'idx', 'ucb_value', 'importance_value', 'n_idx', 'n_total']
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

def log_csv_batch(csv_file, iteration, indices, ucb_scores, importance_est, n_i, sum_n_i):
    """
    Appends a batch of rows to a CSV file.
    """
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        for idx in indices:
            row = [iteration, idx, ucb_scores[idx], importance_est[idx], n_i[idx], sum_n_i]
            writer.writerow(row)   

def load_csv_initial_pass(csv_file:str, ):
    """
    Loads the initial pass log from a CSV file.
    """
    with open(csv_file, mode='r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader if int(row[0]) == 0]
        indices = [int(row[1]) for row in rows]
        values = [float(row[3]) for row in rows]

    return indices, values                 


def compute_importance_batch(
    model: nn.Module,
    image: torch.Tensor,
    label_idx: int,
    reg_params: RegParameters,
    batch_size: int = 64,
    pixel_list: Optional[List[int]] = None,
    sal_map: Optional[np.ndarray] = None
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
    device_ = image.device

    # 1) Infer shape and number of pixels
    _, C, H, W = image.shape
    total_pixels = H * W

    # 2) If no pixel_list provided, default to all pixels in [0..H*W-1]
    if pixel_list is None:
        if sal_map is not None:
            sal_map = sal_map.to(device_)
            sal_map = sal_map.view(-1)
            return sal_map

        pixel_list = list(range(total_pixels))
                        

    # 3) Prepare an output tensor for the subset, shape [len(pixel_list)]
    subset_size = len(pixel_list)
    saliency_subset = torch.zeros((subset_size,), device=device_) 

    iter_count = 0

    # 4) Process the subset in batches
    for start_idx in range(0, subset_size, batch_size): 
        iter_count += 1
        end_idx = min(start_idx + batch_size, subset_size)
        batch_indices = pixel_list[start_idx:end_idx]

        # 4a) Pass this batch of pixels to the "compute_subset_importance" function
        #     which should uniformly sample each pixel in 'batch_indices' and return
        #     a 1D tensor of shape (len(batch_indices),)
        importance_batch = compute_subset_importance(
            model=model,
            images=image,
            pixels=batch_indices,
            label_idx=label_idx,
            reg_params=reg_params
        )  
        # e.g., shape: (len(batch_indices),)
        #print(importance_batch.shape, "importance_batch.shape,  shape: (len(batch_indices),)")

        saliency_subset[start_idx : end_idx] = importance_batch

        if iter_count % 100:
            print(f"Processed {end_idx}/{subset_size} pixels.")
            #return saliency_subset #cancel this line to get the full image

    # 5) Return the importance for the requested subset.
    #    The caller can reshape if needed (e.g., (H, W) if the subset is all pixels).
    return saliency_subset


def compute_ucb_scores(importance_est, n_i, sum_n_i, c):
    """
    Computes the UCB scores for each pixel based on the current estimates and counts.
    """
    num_pixels = len(n_i)
    ucb_scores = np.zeros(num_pixels, dtype=np.float32)
    for i in range(num_pixels):
        if n_i[i] == 0:
            # Large bonus to ensure exploration
            bonus = 1e6
        else:
            bonus = c * np.sqrt(np.log(sum_n_i) / n_i[i])
        ucb_scores[i] = importance_est[i] + bonus
    return ucb_scores
        

def do_initial_pass_uniform(
    model, image, label_idx, reg_params,
    global_batch_size_for_perturbations,
    n_init, csv_path, num_pixels, sal_map=None
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
        batch_size=global_batch_size_for_perturbations,
        pixel_list=None,  # or range(num_pixels),
        sal_map = sal_map
    )
    #print('uniform_init_map.shape:', uniform_init_map.shape)
    
    # Restore original n_samples
    reg_params.n_samples = old_n_samples

    _, C, H, W = image.shape
    total_pixels = H * W
    pixel_list = list(range(total_pixels))
    n_total = n_init * total_pixels

    importance_est = np.zeros(num_pixels, dtype=np.float32)
    n_i = np.zeros(num_pixels, dtype=np.int32)

    for i in range(num_pixels):
        n_i[i] = n_init
        importance_est[i] = uniform_init_map[i].item()

    ucb_scores = compute_ucb_scores(importance_est, n_i, n_total, reg_params.c)

    if sal_map is None:
        log_csv_batch(csv_path, 0, pixel_list, ucb_scores, importance_est, n_i, n_total)

    return uniform_init_map, n_i, importance_est

def do_ucb_iteration_partialsort(
    iteration, top_percent, 
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
    ucb_scores = compute_ucb_scores(importance_est, n_i, sum_n_i, reg_params.c)
    
    # 2) Partial-sort => top-k
    top_k_indices = np.argpartition(ucb_scores, -k)[-k:]
    
    # 3) Sample these top-k pixels with your uniform function
    importance_batch = compute_importance_batch(
        model=model,
        image=image,
        label_idx=label_idx,
        reg_params=reg_params,
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
    log_csv_batch(csv_path, iteration, top_k_indices, ucb_scores, importance_est, n_i, new_sum_n_i)
    
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
    sal_map: Optional[np.ndarray] = None,
    last_ucb_t: Optional[int] = None
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
    uniform_init_map, n_i, importance_est = do_initial_pass_uniform(
        model=model,
        image=image,
        label_idx=label_idx,
        reg_params=reg_params,
        global_batch_size_for_perturbations=global_batch_size_for_perturbations,
        n_init=n_init,
        num_pixels=num_pixels,
        csv_path=csv_path,
        sal_map = sal_map
    )

    # (2) Main UCB loop
    for t in range(last_ucb_t + 1, last_ucb_t + ucb_iterations + 1):
        sum_n_i = max(1, np.sum(n_i))
        importance_est, n_i = do_ucb_iteration_partialsort(
            iteration=t,
            top_percent=top_percent,
            importance_est=importance_est,
            n_i=n_i,
            sum_n_i=sum_n_i,
            compute_importance_batch=compute_importance_batch,
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


def load_importance_map_from_csv(H: int, W: int ,csv_path: str) -> np.ndarray:
    """
    Loads the final importance map from a CSV file.
    """
    num_pixels = H * W
    final_map = np.zeros(num_pixels, dtype=np.float32)

    with open(csv_path, mode='r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader][1:]  # Skip header
        indices = [int(row[1]) for row in rows]
        values = [float(row[3]) for row in rows]
    
    if len(rows) == 0:
        return final_map, 0
    
    last_ucv_iteration = rows[-1][0]

    for idx, val in zip(indices, values):
        final_map[idx] = val

    final_map = torch.from_numpy(final_map.reshape(H, W))    
    return final_map, last_ucv_iteration

def generate_importance_map_ucb(
    image_path: str, 
    model: nn.Module, 
    labels: list, 
    reg_params,
    ucb_iterations: int = 5,
    top_percent: float = 0.1,
    batch_size_for_perturbations: int = 64,
    n_init: int = 3,
    csv_path: str = "ucb_log.csv",
    calculate: bool = True
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
        #if label == gt_labels[1]:
        #    continue
        try:
            label_idx = labels.index(label)
        except ValueError:
            print(f"Label '{label}' not found in the provided labels list.")
            continue

        # We choose a separate text file for the initial pass log
        save_dir = f"{name_no_ext}/{label}/"
        csv_path = f"{save_dir}ucb_log.csv"
        ensure_dir_exists(csv_path)

        if calculate:
            print(f"[UCB] Computing information map for label: {label}")
            if not os.path.exists(csv_path):
                log_csv_header(csv_path)
                last_ucb_t = 0
                sal_map = None
            else:
                _, C, H, W = img_tensor.shape
                sal_map, last_ucb_t = load_importance_map_from_csv(H, W ,csv_path) 

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
                sal_map = sal_map,
                last_ucb_t = int(last_ucb_t)
            )
            sal_map = sal_map.detach().cpu().numpy()
        else:
            _, C, H, W = img_tensor.shape
            sal_map, ucb_iteration = load_importance_map_from_csv(H, W ,csv_path) 

            sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)
            sal_map_uint8 = (sal_map.detach().cpu().numpy() * 255).astype(np.uint8)
            colored_sal_map = cv.applyColorMap(sal_map_uint8, cv.COLORMAP_JET)
            alpha = 0.3  # Transparency factor
            sal_map = cv.addWeighted(img, alpha, colored_sal_map, 1 - alpha, 0)

        info_maps[label] = sal_map

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
        #axes[i].imshow(info_map, cmap='hot')
        axes[i].imshow(cv.cvtColor(info_map, cv.COLOR_BGR2RGB)) #
        axes[i].set_title(f"UCB Info Map: {label}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

    save_dir = save_dir.split('/')[0] + '/'
    info_map_path = f"{save_dir}ucb_info_map.png"
    fig.savefig(info_map_path)
    print(f"UCB-based information map saved to {info_map_path}")

    del info_maps, img_tensor
    torch.cuda.empty_cache()
    gc.collect()

    


def main():
    #PATH = "checkpoints/checkpoint_59epoch_0.9599acc_0.9446valacc_18c.pth"
    #PATH = "checkpoints/checkpoint_47epoch_0.9576acc_0.9529valacc_4c.pth"
    #PATH = "checkpoints/checkpoint_86epoch_0.9327trainF1_0.9344valF1_4c.pth"
    #labels = get_model_labels()  
    #model = load_model(PATH, labels) 

    #IMAGE_PATH = "images2explain/Horse_Zebra.png"
    #IMAGE_PATH = "images2explain/Giraffe_Lion.png" 
    #IMAGE_PATH = "images2explain/Zebra_Lion.png"
    #IMAGE_PATH = "images2explain/Lion_Horse.png"
    IMAGE_PATH = "images2explain/bull mastiff_tabby.png"

    weights = ResNet50_Weights.DEFAULT
    labels = weights.meta["categories"]
    model = resnet50(weights=weights)
    model.eval()
    model.to(device)

    reg_params = RegParameters()
    reg_params.estimation = 'var'  # e.g. variance-based
    reg_params.n_samples = 2
    reg_params.c = 0.25

    generate_importance_map_ucb(
        image_path=IMAGE_PATH,
        model=model,
        labels=labels,
        reg_params=reg_params,
        ucb_iterations=0,
        top_percent=0.7,
        batch_size_for_perturbations=4,
        n_init=2,
        csv_path="ucb_log.csv",
        calculate=True
    )

if __name__ == "__main__":
    main()