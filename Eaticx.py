# This is to be used only if importing the separate scripts doesn't work

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

# Importing more libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import random
import torchvision
from PIL import Image

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
DROPOUT: float = 0.0
DEPTH: int = 12 # from paper ViT-Base
NR_CLASSES: int = 2
NUM_EPOCHS: int = 5
LEARNING_RATE: float = 0.001
CRITERION = nn.CrossEntropyLoss()

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

    # Extracting images from their respective folders
    for folder in ["FAKE", "REAL"]:
        path: str = os.path.join(source, data_type, folder)
        index: int = 0
        for file in os.listdir(path):
            labels.append(0 if folder == "REAL" else 1)
            instances.append(torchvision.io.read_image(os.path.join(path, file)).to(device))
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

## Vision Transformer Architecture


# Image Patching Function
def images_to_patches(images, patch_size, image_size, channels, embedding_size, device):
    """
    Function that splits the images from a batch in patches of a given size,
    then flattens the patches across all channels into linear vectors.
    Parameters: the batch of images, the patch size
    Returns: new tensor representing the batch of patched images
    """

    # Initialization
    new_batch: list = []
    p = patch_size # rename for easier use
    c = channels # number of channels
    i = image_size # width & height
    patch_dim = (patch_size ** 2) * channels

    # Patching
    for image in images:
        new_image = []
        for channel in range(c):
            # For each channel, split in patches & fold patches in linear form
            new_part = image[channel].unfold(0, p, p).unfold(1, p, p).to(device)
            new_part = new_part.contiguous().view(i // p, i // p, p ** 2)
            # Merge channels
            new_image.append(new_part)
        # Convert to tensor
        new_image = torch.stack(new_image, dim = 0).view((i // p) ** 2, c * (p**2)).to(device)
        # Linear embedding to a lower dimension
        embedding = nn.Linear(patch_dim, embedding_size)
        new_image = embedding(new_image.to(torch.float)).to(device)
        # Append image
        new_batch.append(new_image)
    new_batch = torch.stack(new_batch, dim=0).to(device).to(torch.float)
    
    # Return final output
    return new_batch


# Self Attention Module
class SelfAttention(nn.Module):
    def __init__(self, heads: int, embedding_size: int) -> None:
        """
        Initializing (multi-headed) Self Attention object.
        Parameters: number of heads, embedding size
        """
        # Initialization
        super().__init__()

        # Parameters
        self.heads: int = heads
        self.head_size: int = embedding_size * heads

        # Layers
        self.tokeys = nn.Linear(embedding_size, self.head_size, bias=False)
        self.toqueries = nn.Linear(embedding_size, self.head_size, bias=False)
        self.tovalues = nn.Linear(embedding_size, self.head_size, bias=False)
        self.unify_heads = nn.Linear(self.head_size, embedding_size)

    def forward(self, batch):
        """
        Applying layers on an input batch & computing the attention matrix:
            - Queries
            - Keys
            - Values
            - Layer to unify attention heads
        Parameter: batch of dimension (nr_sequences, nr_tokens, embedding_size)
        Returns: result of multi headed self attention (same dimensionality)
        """

        # Input Dimensions
        b, p, emb = batch.size()

        # Applying queries-keys-values layers
        keys = self.tokeys(batch).view(b, p, self.heads, emb)
        queries = self.toqueries(batch).view(b, p, self.heads, emb)
        values = self.tovalues(batch).view(b, p, self.heads, emb)

        # Old Self attention is back
        keys = keys.transpose(1, 2).contiguous().view(b * self.heads, p, emb)
        queries = queries.transpose(1, 2).contiguous().view(b * self.heads, p, emb)
        values = values.transpose(1, 2).contiguous().view(b * self.heads, p, emb)

        # Self-Attention Operations
        weights = torch.bmm(queries, keys.transpose(1, 2))
        weights = weights / math.sqrt(emb // self.heads)
        weights = F.softmax(weights, dim=2)
        attention = torch.bmm(weights, values).view(b, self.heads, p, emb)
        attention = attention.transpose(1,2).contiguous().view(b, p, self.head_size)

        # Final output
        final = self.unify_heads(attention)
        assert final.size() == (b, p, emb), "The size of the final output differs from the input size."
        return final

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, heads: int, embedding_size: int, ff: int, dropout: float) -> None:
        """
        Initializing Transformer Block object.
        Parameters: number of heads, embedding size, feedforward constant, dropout
        """

        # Initialization
        super().__init__()

        # Layers
        self.attention = SelfAttention(heads, embedding_size)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_size, ff * embedding_size),
            nn.GELU(),
            nn.Linear(ff * embedding_size, embedding_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        """
        Applying transformer block layers on an input batch:
            - Multi-headed Self Attention
            - 2 Layers of normalization
            - Feed-Forward block:
                - Linear Layer
                - ReLU activation
                - Linear Layer
            - Dropout
        Parameter: batch of dimension (nr_sequences, nr_tokens, embedding_size)
        Returns: result of transformer block (same dimensionality)
        """

        # Applying layers
        attended = self.attention(batch)
        batch = self.norm1(attended + batch)
        batch = self.dropout(batch)
        feedforward = self.feedforward(batch)
        batch = self.norm2(feedforward + batch)

        # Final output
        final = self.dropout(batch)
        return final

# Transformer Neural Network
class Transformer(nn.Module):
    def __init__(self, device, channels: int, image_size: int, batch_size: int,
        patch_size: int, embedding_size: int, nr_patches: int, 
        attention_heads: int, ff: int, dropout: int, depth: int, 
        nr_classes: int) -> None:
        """
        Function that initializes the Transformer class.
        Parameters: 
        - image channels (should be 3 for RGB)
        - image size (for CIFAKE dataset this will be 32 i think)
        - batch size
        - patch size for splitting input images
        - embedding size for representing patches
        - total number of distinct patches
        - number of attention heads
        - constant to use in feedforward network in transformer block
        - dropout
        - number of transformer blocks
        - number of classes (should be 2 for real and fake)
        Layers:
        - Embedding layer for mapping each distinct patch to a linear embedding
        - Embedding layer for mapping positions of patches within each image
        - Linear layer for unifying the two embeddings
        - Sequence of transformer blocks
        - Linear layer for assigning class values
        """

        # Initialization
        super().__init__()

        # Parameters
        self.device = device
        self.image_size = image_size
        self.channels = channels
        self.patch_size = patch_size
        self.embedding_size = embedding_size

        # Layers
        nr_patches = (image_size // patch_size) ** 2
        self.position_embeddings = nn.Embedding(num_embeddings=nr_patches, embedding_dim=embedding_size)
        self.unify_embeddings = nn.Linear(2 * embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(attention_heads, embedding_size, ff, dropout) for block in range(depth)])
        self.classes = nn.Linear(embedding_size, nr_classes)

    def forward(self, batch: list) -> list:
        """
        Forward call of the Transformer class.
        Parameter: batch of images (batch of images (sequences of patches in 3 channels))
        Operations:
        - Split images from the batch in flattened patches
        - Apply layers defined in init
        - Apply a global average operation before applying the last layer
        Returns: output of transformer network
        """
        patch_emb = images_to_patches(batch, self.patch_size, self.image_size, self.channels, self.embedding_size, self.device)
        x, y, z = patch_emb.size()
        positions = torch.arange(y, device=self.device)
        position_emb = self.position_embeddings(positions)[None, :, :].expand(x, y, z)
        batch = self.unify_embeddings(torch.cat((patch_emb, position_emb), dim=2).view(-1, 2 * z)).view(x, y, z)
        batch = self.dropout(batch)
        batch = self.transformer_blocks(batch)
        batch = torch.mean(batch, dim=1)
        batch = self.classes(batch)
        return batch

# Training Data
xtrain, ytrain = access_data("train", DATA_PATH, DEVICE)
xtrain, ytrain = shuffle_dataset(xtrain, ytrain)
train_dataloader = batch_data(xtrain, ytrain, TRAIN_BATCH_SIZE, DEVICE)

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
            if i % 50 == 49:
                print(f'Epoch {epoch + 1} loss: {running_loss / 50:.3f}')
                running_loss = 0.0

    print('Finished Training')
    PATH: str = f'./eaticx-{name}.pth'
    torch.save(net.state_dict(), PATH)

# Training Loop
desired_net = Transformer(DEVICE, CHANNELS, IMG_SIZE, TRAIN_BATCH_SIZE, \
    PATCH_SIZE, EMBEDDING_SIZE, TRAIN_TOTAL_PATCHES, ATTENTION_HEADS, FF, DROPOUT, \
    DEPTH, NR_CLASSES).to(DEVICE)
desired_net_name: str = "transformer"
train(desired_net, desired_net_name, train_dataloader, NUM_EPOCHS, CRITERION, \
    LEARNING_RATE, DEVICE)