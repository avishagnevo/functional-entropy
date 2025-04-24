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
from captum.attr import IntegratedGradients, NoiseTunnel, Saliency
from typing import Dict, List, Tuple
from torchvision.io import read_image
from torchvision.models import resnet50 #, ResNet50_Weights


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


def get_top_influential_flat_indexes(
    attr_tensors: Dict[str, torch.Tensor],
    top_k: int = 10
) -> Dict[str, List[int]]:
    """
    Collects the top-k pixel indices (in a flattened [H*W] array) from each single-channel
    attribution map, returning a dictionary where:
      - Key = label
      - Value = list of flattened indices (no attribution values).

    :param attr_tensors: Dict of { label: single-channel tensor [1, H, W] }.
    :param top_k: Number of top influential pixels to retrieve for each label.
    :return: { label: [ flat_index_1, flat_index_2, ..., flat_index_top_k ] }.
    """
    top_indices_dict = {}

    for label, tensor in attr_tensors.items():
        # Ensure it's single-channel: [1, H, W]
        if tensor.dim() != 3 or tensor.size(0) != 1:
            print(f"Skipping '{label}': Expected shape [1, H, W], got {tensor.shape}.")
            continue
        
        # Squeeze out the channel dimension => shape: [H, W]
        attribution_map = tensor[0]  # shape: [H, W]
        # Flatten => shape: [H*W]
        flat_map = attribution_map.view(-1)
        
        # Get the indices of the top-k values
        _, top_indices = torch.topk(flat_map, k=top_k)
        
        # Convert top_indices from tensor to Python int list
        top_indices_list = sorted([int(idx.item()) for idx in top_indices], reverse=True)
        
        # Store in the result dictionary
        top_indices_dict[label] = top_indices_list

    return top_indices_dict




def load_attr_maps_to_tensors(
    attr_maps_dir: str,
    image_size: tuple = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, torch.Tensor]:
    """
    Reads single-channel (grayscale) VarGrad attribution map images from a directory 
    and loads them into PyTorch tensors (shape: [1, H, W], values in [0,1]).
    
    :param attr_maps_dir: Path to the directory containing attribution map images.
    :param image_size: Optional resize dimensions (width, height). If None, keeps original size.
    :param device: Device to load tensors on (default is CPU).
    :return: Dictionary mapping labels (derived from filenames) to PyTorch tensors.
    """
    attr_map_tensors = {}
    
    # Transform for single-channel images: 
    #   - Reads a [H,W] numpy array 
    #   - Converts it to a torch tensor in [1, H, W], scaled to [0,1].
    transform_pipeline = transforms.Compose([
        transforms.ToTensor()
    ])
    
    for filename in os.listdir(attr_maps_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Example: If files are named "giraffe_vargrad.png", we'll remove "_vargrad"
            label = os.path.splitext(filename)[0].replace("_vargrad", "")
            file_path = os.path.join(attr_maps_dir, filename)
            
            # 1) Load image in single-channel (grayscale)
            img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Failed to load image {file_path}. Skipping.")
                continue
            
            # 2) Resize if required
            if image_size:
                img = cv.resize(img, image_size)
            
            # 3) Convert to a [1, H, W] tensor and move to device
            tensor_img = transform_pipeline(img).to(device)  # shape = [1, H, W]
            
            attr_map_tensors[label] = tensor_img
            
    return attr_map_tensors



def generate_importance_map_vargrad(
    image_path: str,
    model: torch.nn.Module,
    labels: list,
    nt_samples: int = 20,
    stdevs: float = 0.02
) -> None:
    # 1) Load & preprocess the image (assuming your prepare_image function returns a BGR image + tensor)
    orig_img, img_tensor = prepare_image(image_path)
    img_tensor.requires_grad_(True)

    # 2) Parse ground-truth labels from the filename
    gt_labels = get_labels(image_path)

    # Create dictionaries to store both:
    #  a) Single-channel (raw) grayscale maps
    #  b) Color-overlaid images
    raw_grayscale_maps = {}
    info_maps = {}

    # 3) For each ground-truth label, compute vargrad overlay
    for label in gt_labels:
        try:
            label_idx = labels.index(label)
        except ValueError:
            print(f"Label '{label}' not found in the provided labels list.")
            continue

        # Compute attributions
        saliency = Saliency(model)
        nt = NoiseTunnel(saliency)
        attributions = nt.attribute(
            img_tensor,
            nt_type='smoothgrad_sq',
            stdevs=stdevs,
            nt_samples=nt_samples,
            target=label_idx
        )

        # Convert to numpy and combine channels (L2 norm) -> shape: (H, W)
        attr_map = attributions.detach().cpu().numpy()[0]
        per_chunnel_mean = np.mean(attr_map, axis=0)
        print(per_chunnel_mean)
        # Write per_channel_mean to a file
        with open("per_channel_mean.txt", "w") as f:
            f.write("Per-channel mean values:\n")
            f.write("\n".join(map(str, per_chunnel_mean)))
        attr_map = np.linalg.norm(attr_map, axis=0)

        # Normalize to [0,1] range
        attr_map = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)

        # Convert to single-channel 8-bit grayscale for saving
        attr_map_uint8 = (attr_map * 255).astype(np.uint8)
        raw_grayscale_maps[label] = attr_map_uint8

        # (Optionally) create a color heatmap for visualization
        colored_attr_map = cv.applyColorMap(attr_map_uint8, cv.COLORMAP_JET)
        colored_attr_map = cv.resize(colored_attr_map, (orig_img.shape[1], orig_img.shape[0]))

        # Overlay color heatmap on the original BGR image
        overlay = cv.addWeighted(orig_img, 0.3, colored_attr_map, 0.7, 0)
        info_maps[label] = overlay

    # 4) Save individual single-channel (grayscale) attribution maps for each label
    single_channel_dir = "individual_vargrad_maps_grayscale"
    os.makedirs(single_channel_dir, exist_ok=True)

    for label, grayscale_map in raw_grayscale_maps.items():
        # Directly save single-channel map as .png
        save_path = os.path.join(single_channel_dir, f"{label}.png")
        cv.imwrite(save_path, grayscale_map)
        print(f"Single-channel raw attribution map for {label} saved at {save_path}")

    # 6) Optionally, display the original + overlays
    num_plots = len(info_maps) + 1
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    # Show original
    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Show each overlay
    for i, (label, overlay) in enumerate(info_maps.items(), start=1):
        axes[i].imshow(cv.cvtColor(overlay, cv.COLOR_BGR2RGB))
        axes[i].set_title(f"VarGrad Map: {label}")
        axes[i].axis("off")

    plt.tight_layout()
    
    # 7) Save the combined figure if desired
    save_dir = "vargrad_sal_maps"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "vargrad_info_map.png")
    fig.savefig(save_path)
    print(f"Composite VarGrad-based information map saved to {save_path}")
    plt.show()



# Example main usage:
def main():
    try:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    except AttributeError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

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

    try:
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        labels = weights.meta["categories"]
        model = resnet50(weights=weights)
    except ImportError:
        labels = ['tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead', 'electric ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house finch', 'junco', 'indigo bunting', 'robin', 'bulbul', 'jay', 'magpie', 'chickadee', 'water ouzel', 'kite', 'bald eagle', 'vulture', 'great grey owl', 'European fire salamander', 'common newt', 'eft', 'spotted salamander', 'axolotl', 'bullfrog', 'tree frog', 'tailed frog', 'loggerhead', 'leatherback turtle', 'mud turtle', 'terrapin', 'box turtle', 'banded gecko', 'common iguana', 'American chameleon', 'whiptail', 'agama', 'frilled lizard', 'alligator lizard', 'Gila monster', 'green lizard', 'African chameleon', 'Komodo dragon', 'African crocodile', 'American alligator', 'triceratops', 'thunder snake', 'ringneck snake', 'hognose snake', 'green snake', 'king snake', 'garter snake', 'water snake', 'vine snake', 'night snake', 'boa constrictor', 'rock python', 'Indian cobra', 'green mamba', 'sea snake', 'horned viper', 'diamondback', 'sidewinder', 'trilobite', 'harvestman', 'scorpion', 'black and gold garden spider', 'barn spider', 'garden spider', 'black widow', 'tarantula', 'wolf spider', 'tick', 'centipede', 'black grouse', 'ptarmigan', 'ruffed grouse', 'prairie chicken', 'peacock', 'quail', 'partridge', 'African grey', 'macaw', 'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted merganser', 'goose', 'black swan', 'tusker', 'echidna', 'platypus', 'wallaby', 'koala', 'wombat', 'jellyfish', 'sea anemone', 'brain coral', 'flatworm', 'nematode', 'conch', 'snail', 'slug', 'sea slug', 'chiton', 'chambered nautilus', 'Dungeness crab', 'rock crab', 'fiddler crab', 'king crab', 'American lobster', 'spiny lobster', 'crayfish', 'hermit crab', 'isopod', 'white stork', 'black stork', 'spoonbill', 'flamingo', 'little blue heron', 'American egret', 'bittern', 'crane bird', 'limpkin', 'European gallinule', 'American coot', 'bustard', 'ruddy turnstone', 'red-backed sandpiper', 'redshank', 'dowitcher', 'oystercatcher', 'pelican', 'king penguin', 'albatross', 'grey whale', 'killer whale', 'dugong', 'sea lion', 'Chihuahua', 'Japanese spaniel', 'Maltese dog', 'Pekinese', 'Shih-Tzu', 'Blenheim spaniel', 'papillon', 'toy terrier', 'Rhodesian ridgeback', 'Afghan hound', 'basset', 'beagle', 'bloodhound', 'bluetick', 'black-and-tan coonhound', 'Walker hound', 'English foxhound', 'redbone', 'borzoi', 'Irish wolfhound', 'Italian greyhound', 'whippet', 'Ibizan hound', 'Norwegian elkhound', 'otterhound', 'Saluki', 'Scottish deerhound', 'Weimaraner', 'Staffordshire bullterrier', 'American Staffordshire terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'wire-haired fox terrier', 'Lakeland terrier', 'Sealyham terrier', 'Airedale', 'cairn', 'Australian terrier', 'Dandie Dinmont', 'Boston bull', 'miniature schnauzer', 'giant schnauzer', 'standard schnauzer', 'Scotch terrier', 'Tibetan terrier', 'silky terrier', 'soft-coated wheaten terrier', 'West Highland white terrier', 'Lhasa', 'flat-coated retriever', 'curly-coated retriever', 'golden retriever', 'Labrador retriever', 'Chesapeake Bay retriever', 'German short-haired pointer', 'vizsla', 'English setter', 'Irish setter', 'Gordon setter', 'Brittany spaniel', 'clumber', 'English springer', 'Welsh springer spaniel', 'cocker spaniel', 'Sussex spaniel', 'Irish water spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old English sheepdog', 'Shetland sheepdog', 'collie', 'Border collie', 'Bouvier des Flandres', 'Rottweiler', 'German shepherd', 'Doberman', 'miniature pinscher', 'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great Dane', 'Saint Bernard', 'Eskimo dog', 'malamute', 'Siberian husky', 'dalmatian', 'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland', 'Great Pyrenees', 'Samoyed', 'Pomeranian', 'chow', 'keeshond', 'Brabancon griffon', 'Pembroke', 'Cardigan', 'toy poodle', 'miniature poodle', 'standard poodle', 'Mexican hairless', 'timber wolf', 'white wolf', 'red wolf', 'coyote', 'dingo', 'dhole', 'African hunting dog', 'hyena', 'red fox', 'kit fox', 'Arctic fox', 'grey fox', 'tabby', 'tiger cat', 'Persian cat', 'Siamese cat', 'Egyptian cat', 'cougar', 'lynx', 'leopard', 'snow leopard', 'jaguar', 'lion', 'tiger', 'cheetah', 'brown bear', 'American black bear', 'ice bear', 'sloth bear', 'mongoose', 'meerkat', 'tiger beetle', 'ladybug', 'ground beetle', 'long-horned beetle', 'leaf beetle', 'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant', 'grasshopper', 'cricket', 'walking stick', 'cockroach', 'mantis', 'cicada', 'leafhopper', 'lacewing', 'dragonfly', 'damselfly', 'admiral', 'ringlet', 'monarch', 'cabbage butterfly', 'sulphur butterfly', 'lycaenid', 'starfish', 'sea urchin', 'sea cucumber', 'wood rabbit', 'hare', 'Angora', 'hamster', 'porcupine', 'fox squirrel', 'marmot', 'beaver', 'guinea pig', 'sorrel', 'zebra', 'hog', 'wild boar', 'warthog', 'hippopotamus', 'ox', 'water buffalo', 'bison', 'ram', 'bighorn', 'ibex', 'hartebeest', 'impala', 'gazelle', 'Arabian camel', 'llama', 'weasel', 'mink', 'polecat', 'black-footed ferret', 'otter', 'skunk', 'badger', 'armadillo', 'three-toed sloth', 'orangutan', 'gorilla', 'chimpanzee', 'gibbon', 'siamang', 'guenon', 'patas', 'baboon', 'macaque', 'langur', 'colobus', 'proboscis monkey', 'marmoset', 'capuchin', 'howler monkey', 'titi', 'spider monkey', 'squirrel monkey', 'Madagascar cat', 'indri', 'Indian elephant', 'African elephant', 'lesser panda', 'giant panda', 'barracouta', 'eel', 'coho', 'rock beauty', 'anemone fish', 'sturgeon', 'gar', 'lionfish', 'puffer', 'abacus', 'abaya', 'academic gown', 'accordion', 'acoustic guitar', 'aircraft carrier', 'airliner', 'airship', 'altar', 'ambulance', 'amphibian', 'analog clock', 'apiary', 'apron', 'ashcan', 'assault rifle', 'backpack', 'bakery', 'balance beam', 'balloon', 'ballpoint', 'Band Aid', 'banjo', 'bannister', 'barbell', 'barber chair', 'barbershop', 'barn', 'barometer', 'barrel', 'barrow', 'baseball', 'basketball', 'bassinet', 'bassoon', 'bathing cap', 'bath towel', 'bathtub', 'beach wagon', 'beacon', 'beaker', 'bearskin', 'beer bottle', 'beer glass', 'bell cote', 'bib', 'bicycle-built-for-two', 'bikini', 'binder', 'binoculars', 'birdhouse', 'boathouse', 'bobsled', 'bolo tie', 'bonnet', 'bookcase', 'bookshop', 'bottlecap', 'bow', 'bow tie', 'brass', 'brassiere', 'breakwater', 'breastplate', 'broom', 'bucket', 'buckle', 'bulletproof vest', 'bullet train', 'butcher shop', 'cab', 'caldron', 'candle', 'cannon', 'canoe', 'can opener', 'cardigan', 'car mirror', 'carousel', "carpenter's kit", 'carton', 'car wheel', 'cash machine', 'cassette', 'cassette player', 'castle', 'catamaran', 'CD player', 'cello', 'cellular telephone', 'chain', 'chainlink fence', 'chain mail', 'chain saw', 'chest', 'chiffonier', 'chime', 'china cabinet', 'Christmas stocking', 'church', 'cinema', 'cleaver', 'cliff dwelling', 'cloak', 'clog', 'cocktail shaker', 'coffee mug', 'coffeepot', 'coil', 'combination lock', 'computer keyboard', 'confectionery', 'container ship', 'convertible', 'corkscrew', 'cornet', 'cowboy boot', 'cowboy hat', 'cradle', 'crane', 'crash helmet', 'crate', 'crib', 'Crock Pot', 'croquet ball', 'crutch', 'cuirass', 'dam', 'desk', 'desktop computer', 'dial telephone', 'diaper', 'digital clock', 'digital watch', 'dining table', 'dishrag', 'dishwasher', 'disk brake', 'dock', 'dogsled', 'dome', 'doormat', 'drilling platform', 'drum', 'drumstick', 'dumbbell', 'Dutch oven', 'electric fan', 'electric guitar', 'electric locomotive', 'entertainment center', 'envelope', 'espresso maker', 'face powder', 'feather boa', 'file', 'fireboat', 'fire engine', 'fire screen', 'flagpole', 'flute', 'folding chair', 'football helmet', 'forklift', 'fountain', 'fountain pen', 'four-poster', 'freight car', 'French horn', 'frying pan', 'fur coat', 'garbage truck', 'gasmask', 'gas pump', 'goblet', 'go-kart', 'golf ball', 'golfcart', 'gondola', 'gong', 'gown', 'grand piano', 'greenhouse', 'grille', 'grocery store', 'guillotine', 'hair slide', 'hair spray', 'half track', 'hammer', 'hamper', 'hand blower', 'hand-held computer', 'handkerchief', 'hard disc', 'harmonica', 'harp', 'harvester', 'hatchet', 'holster', 'home theater', 'honeycomb', 'hook', 'hoopskirt', 'horizontal bar', 'horse cart', 'hourglass', 'iPod', 'iron', "jack-o'-lantern", 'jean', 'jeep', 'jersey', 'jigsaw puzzle', 'jinrikisha', 'joystick', 'kimono', 'knee pad', 'knot', 'lab coat', 'ladle', 'lampshade', 'laptop', 'lawn mower', 'lens cap', 'letter opener', 'library', 'lifeboat', 'lighter', 'limousine', 'liner', 'lipstick', 'Loafer', 'lotion', 'loudspeaker', 'loupe', 'lumbermill', 'magnetic compass', 'mailbag', 'mailbox', 'maillot', 'maillot tank suit', 'manhole cover', 'maraca', 'marimba', 'mask', 'matchstick', 'maypole', 'maze', 'measuring cup', 'medicine chest', 'megalith', 'microphone', 'microwave', 'military uniform', 'milk can', 'minibus', 'miniskirt', 'minivan', 'missile', 'mitten', 'mixing bowl', 'mobile home', 'Model T', 'modem', 'monastery', 'monitor', 'moped', 'mortar', 'mortarboard', 'mosque', 'mosquito net', 'motor scooter', 'mountain bike', 'mountain tent', 'mouse', 'mousetrap', 'moving van', 'muzzle', 'nail', 'neck brace', 'necklace', 'nipple', 'notebook', 'obelisk', 'oboe', 'ocarina', 'odometer', 'oil filter', 'organ', 'oscilloscope', 'overskirt', 'oxcart', 'oxygen mask', 'packet', 'paddle', 'paddlewheel', 'padlock', 'paintbrush', 'pajama', 'palace', 'panpipe', 'paper towel', 'parachute', 'parallel bars', 'park bench', 'parking meter', 'passenger car', 'patio', 'pay-phone', 'pedestal', 'pencil box', 'pencil sharpener', 'perfume', 'Petri dish', 'photocopier', 'pick', 'pickelhaube', 'picket fence', 'pickup', 'pier', 'piggy bank', 'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate', 'pitcher', 'plane', 'planetarium', 'plastic bag', 'plate rack', 'plow', 'plunger', 'Polaroid camera', 'pole', 'police van', 'poncho', 'pool table', 'pop bottle', 'pot', "potter's wheel", 'power drill', 'prayer rug', 'printer', 'prison', 'projectile', 'projector', 'puck', 'punching bag', 'purse', 'quill', 'quilt', 'racer', 'racket', 'radiator', 'radio', 'radio telescope', 'rain barrel', 'recreational vehicle', 'reel', 'reflex camera', 'refrigerator', 'remote control', 'restaurant', 'revolver', 'rifle', 'rocking chair', 'rotisserie', 'rubber eraser', 'rugby ball', 'rule', 'running shoe', 'safe', 'safety pin', 'saltshaker', 'sandal', 'sarong', 'sax', 'scabbard', 'scale', 'school bus', 'schooner', 'scoreboard', 'screen', 'screw', 'screwdriver', 'seat belt', 'sewing machine', 'shield', 'shoe shop', 'shoji', 'shopping basket', 'shopping cart', 'shovel', 'shower cap', 'shower curtain', 'ski', 'ski mask', 'sleeping bag', 'slide rule', 'sliding door', 'slot', 'snorkel', 'snowmobile', 'snowplow', 'soap dispenser', 'soccer ball', 'sock', 'solar dish', 'sombrero', 'soup bowl', 'space bar', 'space heater', 'space shuttle', 'spatula', 'speedboat', 'spider web', 'spindle', 'sports car', 'spotlight', 'stage', 'steam locomotive', 'steel arch bridge', 'steel drum', 'stethoscope', 'stole', 'stone wall', 'stopwatch', 'stove', 'strainer', 'streetcar', 'stretcher', 'studio couch', 'stupa', 'submarine', 'suit', 'sundial', 'sunglass', 'sunglasses', 'sunscreen', 'suspension bridge', 'swab', 'sweatshirt', 'swimming trunks', 'swing', 'switch', 'syringe', 'table lamp', 'tank', 'tape player', 'teapot', 'teddy', 'television', 'tennis ball', 'thatch', 'theater curtain', 'thimble', 'thresher', 'throne', 'tile roof', 'toaster', 'tobacco shop', 'toilet seat', 'torch', 'totem pole', 'tow truck', 'toyshop', 'tractor', 'trailer truck', 'tray', 'trench coat', 'tricycle', 'trimaran', 'tripod', 'triumphal arch', 'trolleybus', 'trombone', 'tub', 'turnstile', 'typewriter keyboard', 'umbrella', 'unicycle', 'upright', 'vacuum', 'vase', 'vault', 'velvet', 'vending machine', 'vestment', 'viaduct', 'violin', 'volleyball', 'waffle iron', 'wall clock', 'wallet', 'wardrobe', 'warplane', 'washbasin', 'washer', 'water bottle', 'water jug', 'water tower', 'whiskey jug', 'whistle', 'wig', 'window screen', 'window shade', 'Windsor tie', 'wine bottle', 'wing', 'wok', 'wooden spoon', 'wool', 'worm fence', 'wreck', 'yawl', 'yurt', 'web site', 'comic book', 'crossword puzzle', 'street sign', 'traffic light', 'book jacket', 'menu', 'plate', 'guacamole', 'consomme', 'hot pot', 'trifle', 'ice cream', 'ice lolly', 'French loaf', 'bagel', 'pretzel', 'cheeseburger', 'hotdog', 'mashed potato', 'head cabbage', 'broccoli', 'cauliflower', 'zucchini', 'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber', 'artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith', 'strawberry', 'orange', 'lemon', 'fig', 'pineapple', 'banana', 'jackfruit', 'custard apple', 'pomegranate', 'hay', 'carbonara', 'chocolate sauce', 'dough', 'meat loaf', 'pizza', 'potpie', 'burrito', 'red wine', 'espresso', 'cup', 'eggnog', 'alp', 'bubble', 'cliff', 'coral reef', 'geyser', 'lakeside', 'promontory', 'sandbar', 'seashore', 'valley', 'volcano', 'ballplayer', 'groom', 'scuba diver', 'rapeseed', 'daisy', "yellow lady's slipper", 'corn', 'acorn', 'hip', 'buckeye', 'coral fungus', 'agaric', 'gyromitra', 'stinkhorn', 'earthstar', 'hen-of-the-woods', 'bolete', 'ear', 'toilet tissue']
        model = resnet50(pretrained=True)
        import json
        import urllib.request
        #LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        #labels = urllib.request.urlopen(LABELS_URL).read().decode("utf-8").splitlines()

    model.eval()
    model.to(device)
    
    generate_importance_map_vargrad(
        image_path=IMAGE_PATH,
        model=model,
        labels=labels,
        nt_samples=20,
        stdevs=1.0
    )

    stop

    attr_maps_dir = "individual_vargrad_maps_grayscale"
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    attr_tensors = load_attr_maps_to_tensors(attr_maps_dir, image_size=(224, 224), device=device)

    for label, tensor in attr_tensors.items():
        print(f"{label}: shape={tensor.shape}, device={tensor.device}")

    top_indices_dict = get_top_influential_flat_indexes(attr_tensors, top_k=20)

    for label, index_list in top_indices_dict.items():
        print(f"\nLabel: {label}")
        print("Flattened Indices:", index_list)

    


if __name__ == "__main__":
    main()
