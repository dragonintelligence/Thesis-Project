## Vision Transformer Architecture


# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        embedding = nn.Linear(patch_dim, embedding_size).to(device)
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
