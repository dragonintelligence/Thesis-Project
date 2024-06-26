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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH: str = "dragonintelligence/CIFAKE-image-dataset"
WANDB: bool = False
CHANNELS: int = 3
IMG_SIZE: int = 32
PATCH_SIZE: int = 4
BATCH_SIZE: int = 256
VAL_TIMES: int = 5
SPLIT: int = 0.5
GRADIENT_CLIP: int = 1
EMB: list = [64, 128]
VIT_HEADS: list = [8, 12, 16]
VIT_FF: int = 4 # from paper ViT-Base
VIT_DEPTH: list = [8, 12, 24]
VIT_DROPOUT: float = 0.2 # from paper VIT
PER_LAT: list = [64, 128]
PER_HEADS: list = [8, 12, 16]
PER_DEPTH: list = [2, 3, 4, 6]
NR_CLASSES: int = 2
NR_EPOCHS: int = 5
LR: float = 0.0003 # from paper VIT for global average ViT
CRITERION = nn.CrossEntropyLoss()

# Data Preprocessing
def batch_transform(batch):
    # turn the images into PyTorch tensors & normalize the images to [-1, 1] range
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

# WANDB
if WANDB:
    wandb.init(entity="dragonintelligence", project=f"Eaticx{model}")

#A
steps: list = [i for i in range(NR_EPOCHS * len(train_dataloader))]

## ViT
# Train
print("ViT 8 blocks 8 heads 128 emb")
print()
net = Eaticx.VisionTransformer(DEVICE, CHANNELS, IMG_SIZE, BATCH_SIZE, \
    PATCH_SIZE, 128, 8, VIT_FF, 8, VIT_DROPOUT, NR_CLASSES)\
        .to(DEVICE)
vectors = Experiments.training_loop(net, "ViT", train_dataloader, val_dataloader, NR_EPOCHS, \
    CRITERION, LR, GRADIENT_CLIP, VAL_TIMES, DEVICE, WANDB)
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
# Plots
plt.plot(vectors[0], steps)
plt.show()


# Per 5 epochs

# Per 10 epochs