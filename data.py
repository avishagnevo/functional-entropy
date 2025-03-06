import os
import zipfile
import os
import cv2 as cv
import numpy as np
import pytorch_lightning as pl
import requests
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import json



class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.hparams = args
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

    def download_weights():
        url = (
            "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
        )

        # Streaming, so we can iterate over the response.
        r = requests.get(url, stream=True)

        # Total size in Mebibyte
        total_size = int(r.headers.get("content-length", 0))
        block_size = 2 ** 20  # Mebibyte
        t = tqdm(total=total_size, unit="MiB", unit_scale=True)

        with open("state_dicts.zip", "wb") as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            raise Exception("Error, something went wrong")

        print("Download successful. Unzipping file...")
        path_to_zip_file = os.path.join(os.getcwd(), "state_dicts.zip")
        directory_to_extract_to = os.path.join(os.getcwd(), "cifar10_models")
        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
            print("Unzip file successful!")

    def train_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.hparams.data_dir, train=True, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.hparams.data_dir, train=False, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


def get_train_data():
    import os
    import cv2 as cv
    import numpy as np

    train_dir = 'animals-detection-images-dataset/train/'

    # List of labels from directory names
    #listdir = os.listdir(train_dir)
    #labels = [label for label in listdir] #if label != '.DS_Store']
    config = load_config('config.json')
    labels = config.get('labels', [])
    label_len = len(labels)
    print(label_len)
    print(labels)

    # Lists to store training data
    X = []
    Y = []

    # Load the data
    for label in labels:
        if label == '.DS_Store':
            continue
        folder_path = os.path.join(train_dir, label)
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv.imread(img_path)
            if img is not None:
                # Resize image to 224x224
                img = cv.resize(img, (224, 224))
                X.append(img)
                Y.append(labels.index(label))

    # Convert lists to NumPy arrays
    X = np.array(X)
    Y = np.array(Y)

    print("Training data dimensions:")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    return train_dir, labels, X, Y



def get_valid_data():

    test_dir = 'animals-detection-images-dataset/test/'

    # List of labels from directory names
    #listdir = os.listdir(test_dir)
    #labels = [label for label in listdir] #if label != '.DS_Store']
    # Load configuration
    config = load_config('config.json')
    labels = config.get('labels', [])
    print("Chosen labels:", labels)
    label_len = len(labels)
    print(label_len)

    # Variables for validation data
    X_valid = []
    Y_valid = []
    X_valid_path = []

    # Load validation data
    for label in labels:
        if label == '.DS_Store':
            continue
        folder_path = os.path.join(test_dir, label)
        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv.imread(img_path)
            if img is not None:
                # Resize image to 224x224
                img = cv.resize(img, (224, 224))
                X_valid.append(img)
                X_valid_path.append(img_path)
                Y_valid.append(labels.index(label))
            break    

    X_valid = np.array(X_valid)
    Y_valid = np.array(Y_valid)

    print("\nValidation data dimensions:")
    print("X_valid shape:", X_valid.shape)
    print("Y_valid shape:", Y_valid.shape)  

    return test_dir, labels, X_valid, Y_valid  



def load_config(config_path: str) -> dict:
    """
    Loads the configuration from a JSON file.
    
    :param config_path: Path to the JSON config file.
    :return: A dictionary with configuration parameters.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config