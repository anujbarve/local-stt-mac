"""
brain.backbone — Shared transformer backbone.

Processes tokenized text into contextual representations.
4-layer transformer with learned positional encoding and
global attention pooling.

Output:
  context: [batch, EMBEDDING_DIM]   — global representation for classification heads
  tokens:  [batch, seq_len, EMBEDDING_DIM] — per-token representations for sequence heads
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain.config import (
    EMBEDDING_DIM, NUM_LAYERS, NUM_HEADS, FF_DIM,
    DROPOUT, MAX_SEQ_LEN, VOCAB_SIZE, PAD_TOKEN,
)


class PositionalEncoding(nn.Module):
    """Learned positional encoding."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model] — token embeddings
        Returns:
            [batch, seq_len, d_model] — embeddings + positional encoding
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.embedding(positions)
        return x + pos_emb


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: optional attention mask [batch, seq_len] True=ignore
        Returns:
            [batch, seq_len, d_model]
        """
        # Pre-norm self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=mask,
        )
        x = x + attn_out

        # Pre-norm feed-forward
        x = x + self.ff(self.norm2(x))
        return x


class AttentionPool(nn.Module):
    """Learned attention pooling: compresses sequence into a single vector."""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, 1, batch_first=True)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            tokens: [batch, seq_len, d_model]
            mask: optional [batch, seq_len] True=ignore
        Returns:
            [batch, d_model] — pooled representation
        """
        batch_size = tokens.size(0)
        query = self.query.expand(batch_size, -1, -1)  # [batch, 1, d_model]
        pooled, _ = self.attn(query, tokens, tokens, key_padding_mask=mask)
        return pooled.squeeze(1)  # [batch, d_model]


class Backbone(nn.Module):
    """
    Shared backbone for all task heads.

    Architecture:
        Token Embedding + Positional Encoding
        → TransformerBlock × NUM_LAYERS
        → Global Attention Pool → context vector
        → Per-token outputs → token representations
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = EMBEDDING_DIM,
        n_layers: int = NUM_LAYERS,
        n_heads: int = NUM_HEADS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
        max_len: int = MAX_SEQ_LEN,
    ):
        super().__init__()
        self.d_model = d_model

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(max_len, d_model)

        # Embedding dropout
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Attention pooling for context vector
        self.pool = AttentionPool(d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_ids: [batch, seq_len] — integer token IDs

        Returns:
            context: [batch, d_model] — global representation
            tokens: [batch, seq_len, d_model] — per-token representations
        """
        # Create padding mask: True where token is PAD (id=0)
        padding_mask = (token_ids == 0)

        # Embed tokens + positions
        x = self.token_embedding(token_ids)
        x = self.pos_encoding(x)
        x = self.embed_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask=padding_mask)

        x = self.final_norm(x)

        # Pool to context vector
        context = self.pool(x, mask=padding_mask)

        return context, x

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
