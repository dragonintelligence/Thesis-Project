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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH: str = "dragonintelligence/CIFAKE-image-dataset"
WANDB: bool = False
CHANNELS: int = 3
IMG_SIZE: int = 32
PATCH_SIZE: int = 4
BATCH_SIZE: int = 256
VAL_TIMES: int = 2
SPLIT: int = 0.5
GRADIENT_CLIP: int = 1
EMB: list = [64, 128]
PER_LAT: list = [64, 128]
PER_HEADS: list = [8, 12, 16]
PER_DEPTH: list = [2, 3, 4, 6]
NR_CLASSES: int = 2
NR_EPOCHS: int = 10 # first tried 5
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

# Perceiver Experiments
print("Perceiver Experiments")
accuracy: dict = {}
f1_score: dict = {}

print("Phase 1: to compare to a 8-Self Attention VIT")
print()
for pdepth in [2, 4]:
    tdepth = 8 // pdepth
    for heads in PER_HEADS:
        for emb in EMB:
            for lat in PER_LAT:
                print(f"{pdepth} perceiver blocks: each has 1 cross attention and {tdepth} blocks of {heads}-headed self attention, embedding size of {emb} and latent size of {lat}")
                print()
                try:
                    net = Eaticx.Perceiver(DEVICE, CHANNELS, IMG_SIZE, BATCH_SIZE, \
                        emb, lat, heads, pdepth, tdepth, NR_CLASSES).to(DEVICE)
                    Experiments.training_loop(net, "Perceiver", train_dataloader, val_dataloader, NR_EPOCHS, \
                        CRITERION, LR, GRADIENT_CLIP, VAL_TIMES, DEVICE, WANDB)
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
                    accuracy[f"{pdepth} {tdepth} {heads} {emb} {lat}"] = tacc
                    f1_score[f"{pdepth} {tdepth} {heads} {emb} {lat}"] = tf1
                except:
                    print("CUDA out of memory.")
                    print()

print("Phase 2: to compare to a 12-Self Attention VIT")
print()
for pdepth in PER_DEPTH:
    tdepth = 12 // pdepth
    for heads in PER_HEADS:
        for emb in EMB:
            for lat in PER_LAT:
                print(f"{pdepth} perceiver blocks: each has 1 cross attention and {tdepth} blocks of {heads}-headed self attention, embedding size of {emb} and latent size of {lat}")
                print()
                try:
                    net = Eaticx.Perceiver(DEVICE, CHANNELS, IMG_SIZE, BATCH_SIZE, \
                        emb, lat, heads, pdepth, tdepth, NR_CLASSES).to(DEVICE)
                    Experiments.training_loop(net, "Perceiver", train_dataloader, val_dataloader, NR_EPOCHS, \
                        CRITERION, LR, GRADIENT_CLIP, VAL_TIMES, DEVICE, WANDB)
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
                    accuracy[f"{pdepth} {tdepth} {heads} {emb} {lat}"] = tacc
                    f1_score[f"{pdepth} {tdepth} {heads} {emb} {lat}"] = tf1
                except:
                    print("CUDA out of memory.")
                    print()

print("Phase 3: to compare to a 24-Self Attention VIT")
print()
for pdepth in [2, 3, 4, 6, 8, 12]:
    tdepth = 24 // pdepth
    for heads in PER_HEADS:
        for emb in EMB:
            for lat in PER_LAT:
                print(f"{pdepth} perceiver blocks: each has 1 cross attention and {tdepth} blocks of {heads}-headed self attention, embedding size of {emb} and latent size of {lat}")
                print()
                try:
                    net = Eaticx.Perceiver(DEVICE, CHANNELS, IMG_SIZE, BATCH_SIZE, \
                        emb, lat, heads, pdepth, tdepth, NR_CLASSES).to(DEVICE)
                    Experiments.training_loop(net, "Perceiver", train_dataloader, val_dataloader, NR_EPOCHS, \
                        CRITERION, LR, GRADIENT_CLIP, VAL_TIMES, DEVICE, WANDB)
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
                    accuracy[f"{pdepth} {tdepth} {heads} {emb} {lat}"] = tacc
                    f1_score[f"{pdepth} {tdepth} {heads} {emb} {lat}"] = tf1
                except:
                    print("CUDA out of memory.")
                    print()

# Getting top 5
accuracy = dict(sorted(accuracy.items(), key=lambda item: item[1], reverse=True))
f1_score = dict(sorted(f1_score.items(), key=lambda item: item[1], reverse=True))
print("Top 5 Perceiver configurations:")
for i in range(5):
    print(f"{i}) Per - {list(f1_score.keys())[i]}: accuracy {accuracy[list(f1_score.keys())[i]] * 100 :.3f} % and F1 score {list(f1_score.values())[i] :.3f}")
