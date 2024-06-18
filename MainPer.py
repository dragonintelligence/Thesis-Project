## Main Script

# Importing scripts
import Eaticx # the neural network objects

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
CHANNELS: int = 3
IMG_SIZE: int = 32
PATCH_SIZE: int = 4
BATCH_SIZE: int = 256
VAL_TIMES: int = 5
SPLIT: int = 0.5
GRADIENT_CLIP: int = 1
VIT_EMB: int = 64 # next power of 2 after 48
VIT_HEADS: int = 12 # from paper ViT-Base
VIT_FF: int = 4 # from paper ViT-Base
VIT_DEPTH: int = 12 # from paper ViT-Base
PER_LAT: int = 64 # same as VIT_EMB
PER_HEADS: int = 8 # from paper Perceiver
PER_DEPTH: int = 8 # ?
NR_CLASSES: int = 2
NR_EPOCHS: int = 7
VIT_LR: float = 0.0003 # from paper VIT
PER_LR: float = 0.0003
CRITERION = nn.CrossEntropyLoss()

# Training Function
def training_loop(net, name: str, t, v, epochs: int, criterion, lr: float, clip: int, eval: int, device: str) -> None:
    """
    Function that trains a chosen Neural Network.
    Parameters: Neural Network object, name for save file, training dataset,
    number of epochs, loss function, learning rate, device
    Prints: Cross Entropy loss at the end of each epoch
    """

    optimizer = optim.Adam(lr=lr, params=net.parameters())
    if name == "ViT":
        scheduler = lr_scheduler.OneCycleLR(max_lr=lr, optimizer=optimizer, total_steps=len(t) * epochs)
    print(f"Start training the {name}.")
    print(len(t))
    for epoch in range(epochs):
        for i, data in enumerate(t, 0):  
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data["img"].to(device), data["label"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()
            if name == "ViT":
                scheduler.step()
            if (i + 1) % (len(t) // eval) == 0 or i == len(t) - 1:
                accuracy, vloss = accuracy_test(v, net, criterion, device)
                print(f'Epoch {epoch + 1}:')
                print(f'- Training running loss: {loss.item():.3f}')
                print(f"- Validation loss: {vloss:.3f}")
                print(f"- Validation accuracy: {accuracy:.3f} %")
                running_loss = 0.0
        
        # elif name == "Perceiver":
        #     if epoch in [84, 102, 114]:
        #         for g in optimizer.param_groups:
        #             g['lr'] /= 10  
    print(f'Finished Training the {name}.')
    PATH: str = f'./eaticx-{name}.pth'
    torch.save(net.state_dict(), PATH)

# Test Accuracy Function
def accuracy_test(dataloader, net, criterion, device: str) -> tuple:
    net.eval()
    correct: int = 0
    total: int = 0
    loss: float = 0.0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data["img"].to(device), data["label"].to(device)
            outputs = net(inputs).to(device)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            l = criterion(outputs, labels)
            loss += l.item() * inputs.size(0)

    accuracy: float = 100 * correct / total
    loss /= total
    return accuracy, loss

def experiment(model: str, dataloaders: tuple) -> None:
    tr, val, te = dataloaders

    #  Training Loop
    if model == "ViT":
        desired_net = Eaticx.VisionTransformer(DEVICE, CHANNELS, IMG_SIZE, BATCH_SIZE, \
            PATCH_SIZE, VIT_EMB, VIT_HEADS, VIT_FF, VIT_DEPTH, NR_CLASSES).to(DEVICE)
        training_loop(desired_net, model, tr, val, NR_EPOCHS, CRITERION, VIT_LR, \
            GRADIENT_CLIP, VAL_TIMES, DEVICE)
    elif model == "Perceiver":
        desired_net = Eaticx.Perceiver(DEVICE, CHANNELS, IMG_SIZE, BATCH_SIZE, \
            PER_LAT, PER_HEADS, PER_DEPTH, NR_CLASSES).to(DEVICE)
        training_loop(desired_net, model, tr, val, NR_EPOCHS, CRITERION, PER_LR, \
            GRADIENT_CLIP, VAL_TIMES, DEVICE)

    print()
    
    # Test Accuracy 
    path: str = f'./eaticx-{model}.pth'
    desired_net.load_state_dict(torch.load(path))
    tacc, tloss = accuracy_test(te, desired_net, CRITERION, DEVICE)
    print(f"Test loss: {tloss:.3f}")
    print(f"Test accuracy: {tacc:.3f} %")

# Transforming images within a batch
def batch_transform(batch):
    # turn the images into PyTorch tensors & normalize the images to [-1, 1] range
    main_transform = v2.Compose([v2.ToTensor(), v2.Lambda(lambda tensor: (tensor * 2) - 1)])
    batch["img"] = [main_transform(x.convert("RGB")) for x in batch["image"]]
    del batch["image"]
    return batch


# Main

# Data - Training = 100000, Val = 10000, Test = 10000
train = load_dataset(PATH, split = "train").with_transform(batch_transform)
train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
val_test_split = load_dataset(PATH, split = "test").train_test_split(test_size=SPLIT)
val = val_test_split["train"].with_transform(batch_transform)
val_dataloader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)
test = val_test_split["test"].with_transform(batch_transform)
test_dataloader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

# Run Experiments
print("Perceiver Experiment")
print()
experiment("Perceiver", (train_dataloader, val_dataloader, test_dataloader))