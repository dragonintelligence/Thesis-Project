## Main Script

# Importing scripts
import Eaticx # the neural network objects
import Experiments # train & test functions

# Importing more libraries
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from torchvision.transforms import v2
import matplotlib.pyplot as plt

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # current device
PATH: str = "dragonintelligence/CIFAKE-image-dataset" # CIFAKE dataset source
VERBOSE: bool = True # whether or not to print training & validation metrics every epoch
CHANNELS: int = 3 # RGB image channels
IMG_SIZE: int = 32 # CIFAKE image height / width
PATCH_SIZE: int = 4 # for ViT
BATCH_SIZE: int = 256 # arbitrary
VAL_TIMES: int = 1 # evaluate once per epoch
SPLIT: int = 0.5 # Validation & Test data 
GRADIENT_CLIP: int = 1 # from paper ViT
VIT_EMB: int = 128 # BEST
VIT_HEADS: int = 8 # BEST
VIT_DEPTH: int = 8 # BEST
VIT_FF: int = 4 # from paper ViT-Base
VIT_DROPOUT: float = 0.2 # from paper VIT
PER_EMB: int = 128 # BEST
PER_LAT: int = 128 # BEST
PER_HEADS: int = 8  # BEST
PER_DEPTH: int = 8  # BEST
PER_LAT_DEPTH: int = 1 # BEST
NR_CLASSES: int = 2 # binary classification
NR_EPOCHS: int = 5 # arbitrary
LR: float = 0.0006 # maximum learning rate in OneCycleLR scheduler
CRITERION = nn.CrossEntropyLoss() # loss function

# Data Preprocessing
def batch_transform(batch):
    """
    Function that converts images from HuggingFace dataset to tensors using torch 
    transforms and normalizes the images to [-1, 1] range.
    Input: image collection
    Output: normalized image tensor
    """
    main_transform = v2.Compose([v2.ToTensor(), v2.Lambda(lambda tensor: (tensor * 2) - 1)])
    batch["img"] = [main_transform(x.convert("RGB")) for x in batch["image"]]
    del batch["image"]
    return batch

# Import Datasets - Training = 100000, Val = 10000, Test = 10000
train = load_dataset(PATH, split = "train").with_transform(batch_transform)
train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_test_split = load_dataset(PATH, split = "test").train_test_split(test_size=SPLIT)
val = val_test_split["train"].with_transform(batch_transform)
val_dataloader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test = val_test_split["test"].with_transform(batch_transform)
test_dataloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# For Plots
epochs: list = [i + 1 for i in range(NR_EPOCHS)]
epochs2: list = [i + 1 for i in range(2 * NR_EPOCHS)]

## ViT
# Train
print(f"ViT {VIT_DEPTH} blocks {VIT_HEADS} heads {VIT_EMB} emb")
print()
net = Eaticx.VisionTransformer(DEVICE, CHANNELS, IMG_SIZE, BATCH_SIZE, \
    PATCH_SIZE, VIT_EMB, VIT_HEADS, VIT_FF, VIT_DEPTH, VIT_DROPOUT, NR_CLASSES)\
        .to(DEVICE)
vectors = Experiments.training_loop(net, "ViT", train_dataloader, val_dataloader, NR_EPOCHS, \
    CRITERION, LR, GRADIENT_CLIP, VAL_TIMES, DEVICE, VERBOSE)
# Test
print("Test Set Evaluation:")
path: str = './eaticx-ViT.pth'
net.load_state_dict(torch.load(path))
tacc, tprec, trec, tf1, tloss = Experiments.evaluation(test_dataloader, net, CRITERION, "test", DEVICE)
print(f"- Test loss: {tloss:.3f}")
print(f"- Test accuracy: {tacc:.3f}")
print(f"- Test precision: {tprec:.3f}")
print(f"- Test recall: {trec:.3f}")
print(f"- Test F1 score: {tf1:.3f}")
print()
# Plot 1
plt.figure(1)
plt.plot(epochs, vectors[0], label="Training Loss")
plt.plot(epochs, vectors[1], label="Validation Loss")
plt.legend(loc="upper left")
plt.xlabel("Nr. Epochs")
plt.ylabel("Cross Entropy Loss")
plt.show()
plt.savefig("ViT Loss.png")
# Plot 2
plt.figure(2)
plt.plot(epochs, vectors[2])
plt.xlabel("Nr. Epochs")
plt.ylabel("Validation Accuracy")
plt.show()
plt.savefig("ViT Accuracy.png")

## Per
# Train
print(f"Per {PER_DEPTH} blocks {PER_LAT_DEPTH} transformers {PER_HEADS} heads {PER_EMB} emb {PER_LAT}")
print()
net = Eaticx.Perceiver(DEVICE, CHANNELS, IMG_SIZE, BATCH_SIZE / 2, \
        PER_EMB, PER_LAT, PER_HEADS, PER_DEPTH, PER_LAT, NR_CLASSES).to(DEVICE)
vectors = Experiments.training_loop(net, "Perceiver", train_dataloader, val_dataloader, NR_EPOCHS, \
    CRITERION, LR / 2, GRADIENT_CLIP, VAL_TIMES, DEVICE, VERBOSE)
# Test
print("Test Set Evaluation:")
path: str = './eaticx-Perceiver.pth'
net.load_state_dict(torch.load(path))
tacc, tprec, trec, tf1, tloss = Experiments.evaluation(test_dataloader, net, CRITERION, "test", DEVICE)
print(f"- Test loss: {tloss:.3f}")
print(f"- Test accuracy: {tacc:.3f}")
print(f"- Test precision: {tprec:.3f}")
print(f"- Test recall: {trec:.3f}")
print(f"- Test F1 score: {tf1:.3f}")
print()
# Plot 1
plt.figure(3)
plt.plot(epochs, vectors[0], label="Training Loss")
plt.plot(epochs, vectors[1], label="Validation Loss")
plt.legend(loc="upper left")
plt.xlabel("Nr. training steps")
plt.ylabel("Cross Entropy Loss")
plt.show()
plt.savefig("Per Loss.png")
# Plot 2
plt.figure(4)
plt.plot(epochs, vectors[2])
plt.xlabel("Nr. Epochs")
plt.ylabel("Validation Accuracy")
plt.show()
plt.savefig("Per Accuracy.png")