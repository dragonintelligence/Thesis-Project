## Perceiver Architecture


# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Self Attention Module
class CrossAttention(nn.Module):
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

    def forward(self, batch, latent):
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
        a, b, c = batch.size()
        x, y, z = latent.size()

        # Applying queries-keys-values layers
        keys = self.tokeys(batch).view(a, b, self.heads, c)
        queries = self.toqueries(latent).view(x, y, self.heads, z)
        values = self.tovalues(batch).view(a, b, self.heads, c)

        # Old Self attention is back
        keys = keys.transpose(1, 2).contiguous().view(a * self.heads, b, c)
        queries = queries.transpose(1, 2).contiguous().view(x * self.heads, y, z)
        values = values.transpose(1, 2).contiguous().view(a * self.heads, b, c)

        # Self-Attention Operations
        weights = torch.bmm(queries, keys.transpose(1, 2))
        weights = weights / math.sqrt(c // self.heads)
        weights = F.softmax(weights, dim=2)
        attention = torch.bmm(weights, values).view(a, self.heads, b, c)
        attention = attention.transpose(1,2).contiguous().view(a, b, self.head_size)

        # Final output
        final = self.unify_heads(attention)
        return final

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

# Perceiver Neural Network
class Perceiver(nn.Module):
    def __init__(self, device, channels: int, image_size: int, batch_size: int,
        embedding_size: int, attention_heads: int, ff: int, dropout: int, 
        depth: int, nr_classes: int) -> None:

        # Initialization
        super().__init__()

        # Parameters
        self.device = device
        self.channels = channels

        # Layers
        self.convolution = nn.Conv1d(channels, 256, 1)
        self.pos_emb1 = nn.Parameter(nn.Embedding(image_size ** 2, 256))
        self.latents = nn.Parameter(torch.randn(512, 1024))
        self.cross_attention = CrossAttention(attention_heads, embedding_size)
        self.norm1 = nn.LayerNorm(1024)
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(attention_heads, embedding_size, ff, dropout) for block in range(depth)])
        self.pos_emb2 = nn.Parameter(nn.Embedding(1, 2))


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

        batch = self.convolution(batch)
        a, b, c, d = batch.size()
        batch = batch.view(a, c, d, b).view(a, c*d, b)
        positions1 = self.pos_emb1(torch.arange(256, device=self.device))
        batch = torch.cat((batch, positions1), dim=1)
        attention1 = self.cross_attention(batch, self.latents)
        batch = self.dropout(self.norm1(attention1 + batch))
        batch = self.transformer_blocks(batch)
        positions2 = self.pos_emb2(torch.arange(256, device=self.device))
        attention2 = self.cross_attention(batch, positions2)
        batch = self.dropout(self.norm2(attention2 + batch))[:, None, :]
        return batch
