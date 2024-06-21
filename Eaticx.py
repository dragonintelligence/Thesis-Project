## The library of modules & neural network objects

# Imports

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

# 1) Modules

# Cross Attention Module
class CrossAttention(nn.Module):
    def __init__(self, embedding: int) -> None:
        """
        Initializing (multi-headed) Self Attention object.
        Parameters: embedding (= 2 * input channels)
        """
        # Initialization
        super().__init__()

        # Layers
        self.tokeys = nn.Linear(embedding, embedding, bias=False)
        self.toqueries = nn.Linear(embedding, embedding, bias=False)
        self.tovalues = nn.Linear(embedding, embedding, bias=False)

    def forward(self, batch, latent):
        """
        Applying layers on an input batch & latent, and computing the attention matrix:
            - Queries (latent)
            - Keys (batch)
            - Values (batch)
        Parameter: batch of dimension (batch_size, nr pixels, embedding size),
        latent of dimension (batch size, nr latents, latent size)
        Returns: result of cross attention (same dimensionality as latent)
        """

        # Input Dimensions
        a, b, c = batch.size()
        x, y, z = latent.size()

        # Applying queries-keys-values layers
        keys = self.tokeys(batch)
        queries = self.toqueries(latent)
        values = self.tovalues(batch)

        # Self-Attention Operations
        weights = torch.bmm(queries, keys.transpose(1, 2))
        weights = weights / math.sqrt(c)
        weights = F.softmax(weights, dim=2)
        attention = torch.bmm(weights, values)
        return attention

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
        Parameters: number of heads, embedding size, feedforward constant
        """

        # Initialization
        super().__init__()

        # Layers
        self.self_attention = SelfAttention(heads, embedding_size)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.mlp = nn.Sequential(
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
                - GELU activation
                - Linear Layer
            - Dropout
        Parameter: batch of dimension (nr_sequences, nr_batches, embedding_size)
        Returns: result of transformer block (same dimensionality)
        """
        output = batch + self.self_attention(self.norm1(batch))
        output = self.dropout(output)
        output = output + self.mlp(self.norm2(output))
        output = self.dropout(output)
        return output

# Perceiver Block (1 cross attention & 1 self attention)
class PerceiverBlock(nn.Module):
    def __init__(self, attention_heads: int, embedding_size: int, depth: int) -> None:
        """
        Initializing Transformer Block object.
        Parameters: number of heads, embedding size (= 2 * input channels)
        """

        # Initialization
        super().__init__()

        # Layers
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.norm3 = nn.LayerNorm(embedding_size)
        self.cross_attention = CrossAttention(embedding_size)
        self.latent_transformer = nn.Sequential(*[TransformerBlock(attention_heads, embedding_size, 1, 0) for block in range(depth)])
        self.linear1 = nn.Linear(embedding_size, embedding_size)
        self.linear2 = nn.Linear(embedding_size, embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.GELU(),
            nn.Linear(embedding_size, embedding_size)
        )

    def forward(self, inputs: tuple):
        """
        Applying transformer block layers on an input batch:
            - Multi-headed Self Attention
            - 2 Layers of normalization
            - Feed-Forward block:
                - Linear Layer
                - GELU activation
                - Linear Layer
            - Dropout
        Parameter: batch of dimension (nr_sequences, nr_batches, embedding_size)
        Returns: result of transformer block (same dimensionality)
        """
        batch, latent = inputs
        output = latent + self.cross_attention(self.norm1(batch), latent)
        output = self.linear1(output)
        output = output + self.latent_transformer(self.norm2(output))
        output = self.linear2(output)
        output = output + self.mlp(self.norm3(output))
        return (batch, output)


# 2) Neural Network Objects

# Perceiver Neural Network
class Perceiver(nn.Module):
    def __init__(self, device, channels: int, image_size: int, batch_size: int,
        embedding_size: int, latent_size: int, attention_heads: int, perceiver_depth: int, 
        transformer_depth: int, nr_classes: int) -> None:

        # Initialization
        super().__init__()

        # Parameters
        self.device = device

        # Layers
        self.expand_channels = nn.Conv1d(channels, embedding_size, 1)
        self.position_embeddings = nn.Embedding(image_size ** 2, embedding_size)
        self.unify_embeddings = nn.Linear(2 * embedding_size, embedding_size)
        self.latents = nn.Parameter(torch.randn(batch_size, latent_size, embedding_size))
        self.perceiver_blocks = nn.Sequential(*[PerceiverBlock(attention_heads, embedding_size, transformer_depth) for block in range(perceiver_depth)])
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
        b, c, h, w = batch.size() # b = nr images, c = nr channels, h = height, w = width
        batch = self.expand_channels(batch.view(b, c, h * w))
        batch = torch.permute(batch, (0, 2, 1))
        b, i, e = batch.size() # b is the same, i = h * w, e = expanded channels
        pos_emb = self.position_embeddings(torch.arange(i, device=self.device))[None, :, :].expand(b, i, e)
        batch = self.unify_embeddings(torch.cat((batch, pos_emb), dim=2))
        batch, output = self.perceiver_blocks((batch, self.latents))
        output = torch.mean(output, dim=1)
        output = self.classes(output)
        return output


# VisionTransformer Neural Network
class VisionTransformer(nn.Module):
    def __init__(self, device, channels: int, image_size: int, batch_size: int,
        patch_size: int, embedding_size: int, attention_heads: int, ff: int,
        depth: int, dropout: float, nr_classes: int) -> None:
        """
        Function that initializes the Transformer class.
        Parameters: 
        - device (gpu or cuda)
        - image channels (should be 3 for RGB)
        - image size (should be 32)
        - batch size
        - patch size 
        - embedding size for representing patches
        - number of attention heads
        - constant to use in feedforward network in transformer block
        - number of transformer blocks
        - number of classes (should be 2 for real and fake)
        Layers:
        - Patch embeddings
        - Position embeddings
        - Linear layer for unifying the patched input and position embeddings
        - Sequence of transformer blocks
        - Linear layer for assigning class values
        """

        # Initialization
        super().__init__()

        # Parameters to use in forward call
        self.device = device
        self.patch_size = patch_size

        # Layers
        self.patch_embeddings = nn.Linear(channels * (patch_size ** 2), embedding_size)
        self.position_embeddings = nn.Embedding((image_size // patch_size) ** 2, embedding_size)
        self.unify_embeddings = nn.Linear(2 * embedding_size, embedding_size)
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
        batch = einops.rearrange(batch, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=self.patch_size, p2=self.patch_size)
        patch_emb = self.patch_embeddings(batch)
        x, y, z = patch_emb.size()
        pos_emb = self.position_embeddings(torch.arange(y, device=self.device))[None, :, :].expand(x, y, z)
        batch = self.unify_embeddings(torch.cat((patch_emb, pos_emb), dim=2).view(-1, 2 * z)).view(x, y, z)
        output = self.transformer_blocks(batch)
        output = torch.mean(output, dim=1)
        output = self.classes(output)
        return output
