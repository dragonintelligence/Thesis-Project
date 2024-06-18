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
    def __init__(self, heads: int, embedding_size: int, ff: int) -> None:
        """
        Initializing Transformer Block object.
        Parameters: number of heads, embedding size, feedforward constant
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
        batch = batch + self.attention(self.norm1(batch))
        batch = batch + self.feedforward(self.norm2(batch))
        return batch

# Perceiver Block (1 cross attention & 1 self attention)
class PerceiverBlock(nn.Module):
    def __init__(self, heads: int, embedding: int) -> None:
        """
        Initializing Transformer Block object.
        Parameters: number of heads, embedding size (= 2 * input channels)
        """

        # Initialization
        super().__init__()

        # Layers
        self.norm1 = nn.LayerNorm(embedding)
        self.norm2 = nn.LayerNorm(embedding)
        self.norm3 = nn.LayerNorm(embedding)
        self.norm4 = nn.LayerNorm(embedding)
        self.cross = CrossAttention(embedding)
        self.self = SelfAttention(heads, embedding)
        self.linear1 = nn.Linear(embedding, embedding)
        self.linear2 = nn.Linear(embedding, embedding)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding, embedding),
            nn.GELU(),
            nn.Linear(embedding, embedding)
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
        output =  self.cross(self.norm1(batch), latent)
        output = self.linear1(output)
        output = output + self.self(self.norm3(output))
        output = self.linear2(output)
        output = output + self.feedforward(self.norm4(output))
        return output


# 2) Neural Network Objects

# Perceiver Neural Network
class Perceiver(nn.Module):
    def __init__(self, device, channels: int, image_size: int, batch_size: int,
        latent_size: int, attention_heads: int, depth: int, 
        nr_classes: int) -> None:

        # Initialization
        super().__init__()

        # Parameters
        self.device = device
        self.depth = depth

        # Layers
        self.pos_emb = nn.Parameter(torch.randn(batch_size, image_size ** 2, channels))
        self.latents = nn.Parameter(torch.randn(batch_size, latent_size, 2 * channels))
        self.perceiver_block1 = PerceiverBlock(attention_heads, 2 *  channels)
        self.perceiver_block2 = PerceiverBlock(attention_heads, 2 * channels)
        self.classes = nn.Linear(2 * channels, nr_classes)


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
        a, b, c, d = batch.size()
        batch = batch.to(torch.float).view(a, b, c * d)
        batch = batch.view(a, c * d, b)
        batch = torch.cat((batch, self.pos_emb), dim=2).to(self.device)
        output = self.perceiver_block1((batch, self.latents))
        for block in range(self.depth - 1):
            output = self.perceiver_block2((batch, output))
        output = torch.mean(output, dim=1)
        output = self.classes(output)
        return output

# VisionTransformer Neural Network
class VisionTransformer(nn.Module):
    def __init__(self, device, channels: int, image_size: int, batch_size: int,
        patch_size: int, embedding_size: int, attention_heads: int, ff: int,
        depth: int, nr_classes: int) -> None:
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
        self.position_embeddings = nn.Embedding(embedding_dim=embedding_size, num_embeddings=(image_size // patch_size) ** 2)
        self.unify_embeddings = nn.Linear(2 * embedding_size, embedding_size)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(attention_heads, embedding_size, ff) for block in range(depth)])
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
        batch = self.transformer_blocks(batch)
        batch = torch.mean(batch, dim=1)
        batch = self.classes(batch)
        return batch
