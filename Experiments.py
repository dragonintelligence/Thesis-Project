## Main Script

# Importing scripts
import Eaticx # the neural network objects

# Importing more libraries
import os
import sys
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from torchvision.transforms import v2

# Training Function
def training_loop(net, name: str, t, v, epochs: int, criterion, lr: float, clip: int, eval: int, device: str, wb: bool) -> None:
    """
    Function that trains a chosen Neural Network.
    Parameters: Neural Network object, name for save file, training dataset,
    number of epochs, loss function, learning rate, device
    Prints: Cross Entropy loss at the end of each epoch
    """

    optimizer = optim.Adam(lr=lr, params=net.parameters())
    scheduler = lr_scheduler.OneCycleLR(max_lr=lr, optimizer=optimizer, total_steps=len(t) * epochs)
    print(f"Start training the {name}.")
    if not wb:
        tloss_plot = []
        vloss_plot = []
        lr_plot = []
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
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()
            scheduler.step()
            
            # Weights and Biases Log
            if wb:
                wandb.log(
                    {
                        "Training Loss": loss.item(),
                        "Learning Rate": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "grad_norm": grad_norm
                    }
                )
            else:
                tloss_plot.append(loss.item())
                lr_plot.append(optimizer.param_groups[0]["lr"])

            if (i + 1) % (len(t) // eval) == 0 or i == len(t) - 1:
                # Calculating & printing
                accuracy, vloss = evaluation(v, net, criterion, "val", device)
                print(f'Epoch {epoch + 1}:')
                print(f'- Training running loss: {loss.item():.3f}')
                print(f"- Validation loss: {vloss:.3f}")
                print(f"- Validation accuracy: {accuracy:.3f}")
                running_loss = 0.0
                
                # Weights & Biases log
                if wb:
                    wandb.log(
                        {
                            "Validation Loss": vloss,
                            "Validation Accuracy": accuracy
                        }
                    )
                else:
                    vloss_plot.append(vloss)
        
    print(f'Finished Training the {name}.')
    PATH: str = f'./eaticx-{name}.pth'
    torch.save(net.state_dict(), PATH)
    if not wb:
        return tloss_plot, lr_plot, vloss_plot

# Evaluation metrics Function
def evaluation(dataloader, net, criterion, type: str, device: str) -> tuple:
    net.eval()
    TP, TN, FP, FN = (0,0,0,0)
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
            if type == "test":
                TP += ((predicted == 1) & (labels == 1)).sum().item()
                TN += ((predicted == 0) & (labels == 0)).sum().item()
                FP += ((predicted == 1) & (labels == 0)).sum().item()
                FN += ((predicted == 0) & (labels == 1)).sum().item()
            l = criterion(outputs, labels)
            loss += l.item() * inputs.size(0)

    accuracy: float = correct / total
    loss /= total
    if type == "test":
        precision: float = TP / (TP + FP)
        recall: float = TP / (TP + FN)
        f1: float = (2 * precision * recall) / (precision + recall)
        return accuracy, precision, recall, f1, loss
    else:
        return accuracy, loss