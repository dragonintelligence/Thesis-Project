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
import CIFAKE # the dataset loading functions
import Eaticx # the neural network objects

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
BATCH_SIZE: int = 100
VIT_EMB: int = 200
VIT_HEADS: int = 12 # from paper ViT-Base
FF: int = 4 # from paper ViT-Base
VIT_DEPTH: int = 12 # from paper ViT-Base
PER_EMB: int = 100
PER_LAT: int = 200
PER_HEADS: int = 6
PER_DEPTH: int = 8
NR_CLASSES: int = 2
NUM_EPOCHS: int = 2
LEARNING_RATE: float = 0.00001
CRITERION = nn.CrossEntropyLoss()
MODEL: int = "ViT"

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
    print("Start training the VisionTransformer.")
    
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

print()

# Test Accuracy Function

def accuracy_test(net, name) -> None:
    correct: int = 0
    total: int = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            sentences, labels = data
            # calculate outputs by running images through the network
            outputs = net(sentences)
            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the {name} on the 20000 test images: {100 * correct // total} %')

def experiment(model: str) -> None:

    print(f"Start loading training images for the {model} model")
    print()

    # Training Data
    xtrain, ytrain = CIFAKE.access_data("train", DATA_PATH, DEVICE)
    xtrain, ytrain = CIFAKE.shuffle_dataset(xtrain, ytrain)
    train_dataloader = CIFAKE.batch_data(xtrain, ytrain, BATCH_SIZE, DEVICE)

    print()

    #  Training Loop
    if model == "ViT":
        desired_net = Eaticx.VisionTransformer(DEVICE, CHANNELS, IMG_SIZE, BATCH_SIZE, \
            PATCH_SIZE, VIT_EMB, VIT_HEADS, FF, VIT_DEPTH, NR_CLASSES).to(DEVICE)
    elif model == "Perceiver":
        desired_net = Eaticx.Perceiver(DEVICE, CHANNELS, IMG_SIZE, BATCH_SIZE, \
    PER_EMB, PER_LAT, PER_HEADS, PER_DEPTH, NR_CLASSES).to(DEVICE)
    train(desired_net, model, train_dataloader, NUM_EPOCHS, CRITERION, \
        LEARNING_RATE, DEVICE)

    # Test Accuracy 
    xtest, ytest = CIFAKE.access_data("test", DATA_PATH, DEVICE)
    xtest, ytest = CIFAKE.shuffle_dataset(xtest, ytest)
    test_dataloader = CIFAKE.batch_data(xtest, ytest, BATCH_SIZE, DEVICE)
    PATH: str = f'./eaticx-{model}.pth'
    desired_net.load_state_dict(torch.load(PATH))
    accuracy_test(desired_net, model)

experiment(MODEL)
