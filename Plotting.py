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
VAL_TIMES: int = 1
SPLIT: int = 0.5
GRADIENT_CLIP: int = 1
VIT_FF: int = 4 # from paper ViT-Base
VIT_DROPOUT: float = 0.2 # from paper VIT
NR_CLASSES: int = 2
NR_EPOCHS: int = 5
LR: float = 0.0006
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
epochs: list = [i for i in range(NR_EPOCHS)]
steps: list = [i for i in range(NR_EPOCHS * len(train_dataloader))]
epochs2: list = [i for i in range(2 * NR_EPOCHS)]
steps2: list = [i for i in range(2 * NR_EPOCHS * len(train_dataloader))]

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
# Plot 1
plt.figure(1)
plt.plot(epochs, vectors[0], label="Training Loss")
plt.plot(epochs, vectors[1], label="Validation Loss")
plt.legend(loc="upper left")
plt.xlabel("Nr. training steps")
plt.ylabel("Cross Entropy Loss")
plt.show()
plt.savefig("B1loss.png")
# Plot 2
plt.figure(2)
plt.plot(steps, vectors[2])
plt.xlabel("Nr. steps")
plt.ylabel("Learning Rate")
plt.show()
plt.savefig("B1lr.png")

## Per 5 epochs
# Train
print("Per 4 blocks 2 transformers 8 heads 128 emb & lat 5 epochs")
print()
net = Eaticx.Perceiver(DEVICE, CHANNELS, IMG_SIZE, BATCH_SIZE, \
        128, 128, 8, 4, 2, NR_CLASSES).to(DEVICE)
vectors = Experiments.training_loop(net, "Perceiver", train_dataloader, val_dataloader, NR_EPOCHS, \
    CRITERION, LR, GRADIENT_CLIP, VAL_TIMES, DEVICE, WANDB)
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
plt.savefig("B2loss.png")
# Plot 2
plt.figure(4)
plt.plot(steps, vectors[2])
plt.xlabel("Nr. steps")
plt.ylabel("Learning Rate")
plt.show()
plt.savefig("B2lr.png")

## Per 10 epochs
# Train
print("Per 4 blocks 2 transformers 8 heads 128 emb & lat 10 epochs")
print()
net = Eaticx.Perceiver(DEVICE, CHANNELS, IMG_SIZE, BATCH_SIZE, \
        128, 128, 8, 4, 2, NR_CLASSES).to(DEVICE)
vectors = Experiments.training_loop(net, "Perceiver", train_dataloader, val_dataloader, 2 * NR_EPOCHS, \
    CRITERION, LR, GRADIENT_CLIP, VAL_TIMES, DEVICE, WANDB)
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
plt.figure(5)
plt.plot(epochs2, vectors[0], label="Training Loss")
plt.plot(epochs, vectors[1], label="Validation Loss")
plt.legend(loc="upper left")
plt.xlabel("Nr. training steps")
plt.ylabel("Cross Entropy Loss")
plt.show()
plt.savefig("B3loss.png")
# Plot 2
plt.figure(6)
plt.plot(steps2, vectors[2])
plt.xlabel("Nr. steps")
plt.ylabel("Learning Rate")
plt.show()
plt.savefig("B3lr.png")

## Per 2 10 epochs
# Train
print("Per 4 blocks 2 transformers 8 heads 64 emb 128 lat 10 epochs")
print()
net = Eaticx.Perceiver(DEVICE, CHANNELS, IMG_SIZE, BATCH_SIZE, \
        64, 128, 8, 4, 2, NR_CLASSES).to(DEVICE)
vectors = Experiments.training_loop(net, "Perceiver", train_dataloader, val_dataloader, 2 * NR_EPOCHS, \
    CRITERION, LR, GRADIENT_CLIP, VAL_TIMES, DEVICE, WANDB)
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
plt.figure(7)
plt.plot(epochs2, vectors[0], label="Training Loss")
plt.plot(epochs2, vectors[1], label="Validation Loss")
plt.legend(loc="upper left")
plt.xlabel("Nr. training steps")
plt.ylabel("Cross Entropy Loss")
plt.show()
plt.savefig("B4loss.png")
# Plot 2
plt.figure(8)
plt.plot(steps2, vectors[2])
plt.xlabel("Nr. steps")
plt.ylabel("Learning Rate")
plt.show()
plt.savefig("B4lr.png")
