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

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # current device
PATH: str = "dragonintelligence/CIFAKE-image-dataset" # CIFAKE dataset source
VERBOSE: bool = False # whether or not to print training & validation metrics every epoch
CHANNELS: int = 3 # RGB image channels
IMG_SIZE: int = 32 # CIFAKE image height / width
BATCH_SIZE: int = 128 # arbitrary
VAL_TIMES: int = 2
SPLIT: int = 0.5 # Validation & Test data 
GRADIENT_CLIP: int = 1 # from paper ViT
EMB: list = 128
PER_HEADS: list = [8, 12] # to test
PER_DEPTH: list = [1, 2, 4, 8] # to test
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

# Perceiver Experiments
print("Perceiver Experiments That Didn't Work")
loss: dict = {}
accuracy: dict = {}
f1_score: dict = {}

print("A) 4 2 12 128")
print()
net = Eaticx.Perceiver(DEVICE, CHANNELS, IMG_SIZE, BATCH_SIZE, \
    128, 128, 12, 4, 2, NR_CLASSES).to(DEVICE)
Experiments.training_loop(net, "Perceiver", train_dataloader, val_dataloader, NR_EPOCHS, \
    CRITERION, LR, GRADIENT_CLIP, VAL_TIMES, DEVICE, VERBOSE)
# Test Accuracy 
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
loss[f"4 2 12 128"] = tloss
accuracy[f"4 2 12 128"] = tacc
f1_score[f"4 2 12 128"] = tf1

print("B) 8 1 8 128")
print()
net = Eaticx.Perceiver(DEVICE, CHANNELS, IMG_SIZE, BATCH_SIZE, \
    128, 128, 8, 8, 1, NR_CLASSES).to(DEVICE)
Experiments.training_loop(net, "Perceiver", train_dataloader, val_dataloader, NR_EPOCHS, \
    CRITERION, LR, GRADIENT_CLIP, VAL_TIMES, DEVICE, VERBOSE)
# Test Accuracy 
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
loss[f"8 1 8 128"] = tloss
accuracy[f"8 1 8 128"] = tacc
f1_score[f"8 1 8 128"] = tf1

print("C) 8 1 12 128")
print()
net = Eaticx.Perceiver(DEVICE, CHANNELS, IMG_SIZE, BATCH_SIZE, \
    128, 128, 12, 8, 1, NR_CLASSES).to(DEVICE)
Experiments.training_loop(net, "Perceiver", train_dataloader, val_dataloader, NR_EPOCHS, \
    CRITERION, LR, GRADIENT_CLIP, VAL_TIMES, DEVICE, VERBOSE)
# Test Accuracy 
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
loss[f"8 1 12 128"] = tloss
accuracy[f"8 1 12 128"] = tacc
f1_score[f"8 1 12 128"] = tf1

# Visualize results sorted by lowest test loss
loss = dict(sorted(loss.items(), key=lambda item: item[1]))
print("Perceiver Configurations sorted by test loss:")
for i in range(len(list(loss.keys()))):
    print(f"{i + 1}) Perceiver - {list(loss.keys())[i]}: loss {list(loss.values())[i]}, accuracy {accuracy[list(loss.keys())[i]] * 100 :.3f} %, F1 score {f1_score[list(loss.keys())[i]] :.3f}")
