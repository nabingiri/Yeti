"""
__author__: Nabin
Transformer block with Rotary embeddings.
"""

from typing import Optional
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rotary import RotaryEmbedding


def sinusoidal_time_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: float = 10000.0,
) -> torch.Tensor:
    half_dim = embedding_dim // 2
    device = timesteps.device
    emb = math.log(max_period) / max(half_dim - 1, 1)
    freq = torch.exp(torch.arange(half_dim, device=device) * -emb)
    args = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = hidden_dim * mult
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rotary: bool = True,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rotary = use_rotary
        if use_rotary and self.head_dim % 2 != 0:
            raise ValueError("Rotary embeddings require even head_dim.")

        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.head_dim) if use_rotary else None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        qkv = self.to_qkv(x)
        # (B, L, 3*H*D) -> (3, B, H, L, D)
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, L, D)

        if self.rotary is not None:
            q, k = self.rotary(q, k)

        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)

        dropout_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(B, L, self.hidden_dim)
        return self.out_proj(out)

class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rotary: bool = True,
        ff_mult: int = 6,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(hidden_dim, num_heads, dropout, use_rotary)
        self.ff = FeedForward(hidden_dim, mult=ff_mult, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        def _inner_forward(x_in: torch.Tensor) -> torch.Tensor:
            h = x_in + self.self_attn(self.norm1(x_in), mask=mask)
            h = h + self.ff(self.norm2(h))
            return h

        if self.gradient_checkpointing and self.training:
            return checkpoint(_inner_forward, x, use_reentrant=False)
        return _inner_forward(x)

