## Main Script

# Init based on platform / device
import os
import sys

try:
    from google.colab import drive
    COLAB: bool = True
    drive.mount('/content/drive')
    sys.path.append('/content/drive/My Drive/Colab Notebooks')
    DATA_PATH = os.path.join(sys.path[-1], "Dataset")
except:
    COLAB: bool = False
    sys.path.append(os.getcwd())
    DATA_PATH = os.path.join(os.getcwd(), "data")

# Importing scripts
import CIFAKE
import EaticxTransformer

# Importing more libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNELS: int = 3
IMG_SIZE: int = 32
PATCH_SIZE: int = 4
TRAIN_BATCH_SIZE: int = 100
TEST_BATCH_SIZE: int = 1000
TRAIN_TOTAL_PATCHES: int = 6400
TEST_TOTAL_PATCHES: int = 64000
EMBEDDING_SIZE: int = 100
ATTENTION_HEADS: int = 12 # from paper ViT-Base
FF: int = 4 # from paper ViT-Base
DROPOUT: float = 0.0 # from paper ViT
DEPTH: int = 12 # from paper ViT-Base
NR_CLASSES: int = 2
NUM_EPOCHS: int = 7 # from paper ViT
LEARNING_RATE: float = 0.0008 # from paper ViT
CRITERION = nn.CrossEntropyLoss()

# Training Data
xtrain, ytrain = CIFAKE.access_data("train", DATA_PATH, DEVICE)
xtrain, ytrain = CIFAKE.shuffle_dataset(xtrain, ytrain)
train_dataloader = CIFAKE.batch_data(xtrain, ytrain, TRAIN_BATCH_SIZE, DEVICE)

# Training Function
def train(net, name, dataloader: list, nr_epochs: int, criterion, lr: float, device: str) -> None:
    """
    Function that trains a chosen Neural Network.
    Parameters: Neural Network object, name for save file, training dataset,
    number of epochs, loss function, learning rate, device
    Prints: Cross Entropy loss at the end of each epoch
    """

    optimizer = Adam(lr=lr, params=net.parameters())
    nr_train_batches = len(dataloader)
    print("Start training.")
    
    for epoch in range(nr_epochs):
        running_loss: float = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.type(torch.LongTensor).to(device)
            labels = labels.type(torch.LongTensor).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            outputs = outputs.type(torch.FloatTensor).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i == nr_train_batches - 1:
                print(f'Epoch {epoch + 1} loss: {running_loss / (i+1):.3f}')
                running_loss = 0.0

    print('Finished Training')
    PATH: str = f'./eaticx-{name}.pth'
    torch.save(net.state_dict(), PATH)

# Training Loop
desired_net = EaticxTransformer.Transformer(DEVICE, CHANNELS, IMG_SIZE, TRAIN_BATCH_SIZE, \
    PATCH_SIZE, EMBEDDING_SIZE, TRAIN_TOTAL_PATCHES, ATTENTION_HEADS, FF, DROPOUT, \
    DEPTH, NR_CLASSES).to(DEVICE)
desired_net_name: str = "transformer"
train(desired_net, desired_net_name, train_dataloader, NUM_EPOCHS, CRITERION, \
    LEARNING_RATE, DEVICE)
