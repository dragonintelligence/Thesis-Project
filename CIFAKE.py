## Data Loading Script

# Imports
import os
import torch
import random
import torchvision
from PIL import Image

# Function for accessing the dataset of a certain type (train/test)
def access_data(data_type: str, source: str, device: str) -> tuple:
    """
    Function that reads the folders of image data & generates a list of image 
    tensors and a list of labels (0 if real image, 1 if AI-generated image) for
    a given dataset
    Parameters: type of dataset (string, train or test), path, device
    Returns: resulting list of inputs (tensors) and list of labels (int)
    """

    # Initialization
    instances: list = []
    labels: list = []
    transform = torchvision.transforms.ToTensor()

    # Extracting images from their respective folders
    for folder in ["FAKE", "REAL"]:
        path: str = os.path.join(source, data_type, folder)
        index: int = 0
        for file in os.listdir(path):
            labels.append(0 if folder == "REAL" else 1)
            image = Image.open(os.path.join(path, file))
            image = transform(image).to(device)
            instances.append(image)
            index += 1
            if index % 10000 == 0:
                print(f"- {index} {folder} images accessed.")
        print(f"All {folder} images are accessed.")
        print()

    # Returning resulting dataset
    print("All images are accessed.")
    return instances, labels

# Function for shuffling the data (otherwise it would have all fake data first and all 
# real data second)
def shuffle_dataset(inputs: list, labels: list) -> tuple:
    """
    Function that takes a list of input tensors and a list of labels, shuffles
    their order (while keeping the inputs and their respective labels on the
    same index) and returns the result
    Parameters: 
    """

    # Asserts
    assert len(inputs) == len(labels), f"The number of inputs ({len(inputs)}) and \
        that of labels ({len(labels)}) don't match."
    
    # Dictionary that keeps maps inputs to labels
    order: dict = {inputs[i]: labels[i] for i in range(len(inputs))}

    # Shuffling datasets
    random.shuffle(inputs)
    labels = [order[instance] for instance in inputs]

    # Returning resulting dataset
    return inputs, labels

# Function for splitting dataset into batches and returning a dataloader
def batch_data(inputs: list, labels: list, batch_size: int, device: str) -> list:

    # Asserts
    assert len(inputs) == len(labels), f"The number of inputs ({len(inputs)}) and \
        that of labels ({len(labels)}) don't match."
    assert len(inputs) % batch_size == 0, f"The dataset size ({len(inputs)}) is not \
        divisible by the batch size ({batch_size})."

    # Initialization
    dataloader: list = []

    # Batching
    for i in range(0, len(inputs), batch_size):
        x = torch.stack(inputs[i : i + batch_size], dim=0).to(device)
        y = torch.tensor(labels[i : i + batch_size]).to(device)
        dataloader.append((x, y))
    
    # Return resulting dataset
    return dataloader
