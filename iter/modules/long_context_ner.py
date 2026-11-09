"""Hierarchical context refinement for long-document NER.

The base ITER model already has a strong token-level span decoder.  This
module keeps that decoder intact and supplies it with a representation that
combines short-range boundary cues with document-level chunk context.  It is
deliberately independent of relation extraction so NER can be improved first
without introducing a second-stage relation model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LongContextNERRefiner(nn.Module):
    """Chunk-context boundary refinement (HCBR).

    Tokens are processed by a depthwise local convolution and by a compact
    self-attention layer over masked chunk means.  Chunk representations are
    broadcast back to their tokens and fused with a learned gate.  Complexity
    is O(N) for the local path plus O((N/chunk_size)^2) for the chunk path.
    """

    def __init__(
        self,
        hidden_dim: int,
        chunk_size: int = 32,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if chunk_size < 4:
            raise ValueError("chunk_size must be at least 4")
        if hidden_dim % num_heads != 0:
            divisors = [n for n in range(min(num_heads, hidden_dim), 0, -1) if hidden_dim % n == 0]
            num_heads = divisors[0]
        self.chunk_size = chunk_size
        self.local = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.chunk_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.chunk_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
        batch, seq_len, hidden = hidden_states.shape
        mask = (
            attention_mask.bool()
            if attention_mask is not None
            else torch.ones(batch, seq_len, device=hidden_states.device, dtype=torch.bool)
        )

        local = self.local(hidden_states.transpose(1, 2)).transpose(1, 2)
        local = local * mask.unsqueeze(-1).to(local.dtype)

        chunk_count = (seq_len + self.chunk_size - 1) // self.chunk_size
        padded_len = chunk_count * self.chunk_size
        pad = padded_len - seq_len
        padded = F.pad(hidden_states, (0, 0, 0, pad))
        padded_mask = F.pad(mask, (0, pad), value=False)
        chunks = padded.view(batch, chunk_count, self.chunk_size, hidden)
        chunk_mask = padded_mask.view(batch, chunk_count, self.chunk_size)
        denom = chunk_mask.sum(dim=2, keepdim=True).clamp_min(1).to(chunks.dtype)
        chunk_repr = (chunks * chunk_mask.unsqueeze(-1).to(chunks.dtype)).sum(dim=2) / denom
        all_padding = ~chunk_mask.any(dim=2)
        chunk_context, _ = self.chunk_attention(
            chunk_repr,
            chunk_repr,
            chunk_repr,
            key_padding_mask=all_padding,
            need_weights=False,
        )
        chunk_context = chunk_context + self.chunk_ffn(chunk_context)
        global_context = chunk_context.unsqueeze(2).expand(-1, -1, self.chunk_size, -1).reshape(
            batch, padded_len, hidden
        )[:, :seq_len]

        gate = self.gate(torch.cat((local, global_context), dim=-1))
        fused = gate * global_context + (1.0 - gate) * local
        return self.output_norm(hidden_states + self.dropout(fused) * mask.unsqueeze(-1).to(fused.dtype))
