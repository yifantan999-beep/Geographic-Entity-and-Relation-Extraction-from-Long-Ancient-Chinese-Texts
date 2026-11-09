"""
Multi-Scale Geographic Representation Module for Wenyanwen Geo-NER.

Wenyanwen (文言文) place names often contain hierarchical structures:
  - Character-level: individual characters carry semantic meaning
  - Local n-gram level: 2-4 character combinations form place names
  - Span-level: complete entity spans encode geographic semantics

Architecture (3-branch with gated fusion):

  Branch A: Transformer embeddings
    - Original token-level representations from the encoder
    - Captures contextualized character-level features

  Branch B: Multi-Kernel CNN
    - CNN with kernel sizes 2, 3, 4, 5
    - Captures local n-gram patterns at different granularities
    - Appropriate for Classical Chinese where words are 1-4 chars

  Branch C: Span Pooling
    - For candidate geographic spans, pool representations
    - Captures span-level geographic features

  Gated Fusion:
    - Learnable gates to adaptively combine all three branches
    - Output: Enhanced geo-aware representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List


class TransformerBranch(nn.Module):
    """
    Branch A: Transformer-based character-level features.

    A small additional Transformer layer to refine character-level representations.
    This operates on top of the encoder's output to produce character-aware features.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: [B, N, H] token representations
            attention_mask: [B, N] optional attention mask

        Returns:
            [B, N, H] refined character-level representations
        """
        # Self-attention
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = self.layer_norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)

        return x


class MultiKernelCNN(nn.Module):
    """
    Branch B: Multi-Kernel CNN for local n-gram features.

    Uses CNN kernels of sizes 2, 3, 4, 5 to capture n-gram patterns.
    Each kernel size corresponds to different character-level n-gram granularity,
    which is particularly important for Classical Chinese where "words" are
    typically 1-4 characters.

    Outputs are concatenated and projected back to hidden_dim.
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_sizes: List[int] = None,
        num_filters_per_kernel: int = 128,
        dropout: float = 0.3,
    ):
        """
        Args:
            hidden_dim: Input/output hidden dimension (H)
            kernel_sizes: List of CNN kernel sizes (default: [2, 3, 4, 5])
            num_filters_per_kernel: Number of output filters per kernel
            dropout: Dropout rate
        """
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4, 5]

        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters_per_kernel
        total_filters = len(kernel_sizes) * num_filters_per_kernel

        # CNN layers: one per kernel size
        # Input: [B, N, H] -> permute to [B, H, N] for Conv1d
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=num_filters_per_kernel,
                    kernel_size=k,
                    padding='same',  # Preserve sequence length
                ),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for k in kernel_sizes
        ])

        # Project concatenated CNN outputs back to hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(total_filters, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, N, H] token representations

        Returns:
            [B, N, H] n-gram enhanced representations
        """
        # [B, N, H] -> [B, H, N] for Conv1d
        x_conv = x.permute(0, 2, 1)

        # Apply each CNN kernel and collect outputs
        cnn_outputs = []
        for conv in self.convs:
            out = conv(x_conv)          # [B, num_filters, N]
            out = out.permute(0, 2, 1)  # [B, N, num_filters]
            cnn_outputs.append(out)

        # Concatenate along filter dimension: [B, N, total_filters]
        combined = torch.cat(cnn_outputs, dim=-1)

        # Project back to hidden_dim: [B, N, H]
        output = self.output_proj(combined)

        return output


class SpanPoolingBranch(nn.Module):
    """
    Branch C: Span-level geographic feature pooling.

    For each token position, pools features from candidate geographic spans
    that contain or are near this position. This captures span-level geographic
    semantics that may not be visible at the character level.

    Uses self-attention over candidate spans weighted by position proximity.
    """

    def __init__(self, hidden_dim: int, max_span_length: int = 10, dropout: float = 0.3):
        """
        Args:
            hidden_dim: Hidden dimension (H)
            max_span_length: Maximum span length to consider for pooling
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_span_length = max_span_length

        # Attention-based span feature aggregation
        self.span_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # Position-aware span pooling
        # For each position i, look at spans [i-k, i], [i, i+k] for k in [1, max_span_length]
        self.position_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: [B, N, H] token representations
            attention_mask: [B, N] optional attention mask

        Returns:
            [B, N, H] span-pooling enhanced representations
        """
        batch_size, seq_len, hidden_dim = x.shape

        # For each position, compute span-pooled features by:
        # 1. Computing multi-scale span representations around each position
        # 2. Using attention to aggregate the most relevant span features

        # Collect span representations for each position
        # For efficiency, we use a sliding window approach
        span_features_list = []

        for span_len in range(1, min(self.max_span_length + 1, seq_len)):
            # Compute left-anchored span: [i - span_len, i]
            # and right-anchored span: [i, i + span_len]
            left_padded = F.pad(x, (0, 0, span_len, 0))  # [B, N+span_len, H]
            right_padded = F.pad(x, (0, 0, 0, span_len))  # [B, N+span_len, H]

            # Extract spans by shifting
            # Left span at position i: mean of [i-span_len, i]
            left_spans = []
            right_spans = []
            for b in range(batch_size):
                left_batch = []
                right_batch = []
                for i in range(seq_len):
                    # Left span: tokens [max(0, i-span_len+1), i]
                    l_start = max(0, i - span_len + 1)
                    left_tokens = x[b, l_start:i + 1, :]
                    left_batch.append(left_tokens.mean(dim=0))

                    # Right span: tokens [i, min(seq_len-1, i+span_len-1)]
                    r_end = min(seq_len - 1, i + span_len - 1)
                    right_tokens = x[b, i:r_end + 1, :]
                    right_batch.append(right_tokens.mean(dim=0))

                left_spans.append(torch.stack(left_batch))   # [N, H]
                right_spans.append(torch.stack(right_batch))  # [N, H]

            left_spans = torch.stack(left_spans)   # [B, N, H]
            right_spans = torch.stack(right_spans)  # [B, N, H]

            # Combine left and right span features
            combined = torch.cat([left_spans, right_spans], dim=-1)  # [B, N, 2H]
            span_feat = self.position_proj(combined)  # [B, N, H]
            span_features_list.append(span_feat)

        if span_features_list:
            # Stack all span length features: [B, S, N, H] where S = num span lengths
            all_span_features = torch.stack(span_features_list, dim=1)  # [B, S, N, H]

            # Average over span lengths to get final span features
            # [B, S, N, H] -> [B, N, H]
            span_features = all_span_features.mean(dim=1)
        else:
            span_features = x

        # Self-attention over span features
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        attn_out, _ = self.span_attention(span_features, span_features, span_features,
                                          key_padding_mask=key_padding_mask)
        output = self.output_proj(attn_out + span_features)

        return output


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism to adaptively combine multiple representation branches.

    For each branch output, computes a gating weight that determines how much
    of that branch's information to use at each position.

    fusion = sum_i (gate_i * branch_i)
    where gate_i = sigmoid(W_i * [branch_i; context])
    """

    def __init__(self, hidden_dim: int, num_branches: int, dropout: float = 0.3):
        """
        Args:
            hidden_dim: Hidden dimension (H)
            num_branches: Number of representation branches to fuse
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_branches = num_branches

        # Gate computation: for each branch, compute a scalar gate value per position
        # Input: [B, N, H] -> Output: [B, N, 1]
        self.gate_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_branches)
        ])

        # Fusion projection to combine gated outputs
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, branches: List[Tensor]) -> Tensor:
        """
        Args:
            branches: List of [B, N, H] tensors from each branch

        Returns:
            [B, N, H] gated fusion output
        """
        assert len(branches) == self.num_branches, (
            f"Expected {self.num_branches} branches, got {len(branches)}"
        )

        # Compute gates for each branch
        gated_outputs = []
        for branch, gate_layer in zip(branches, self.gate_layers):
            gate = gate_layer(branch)  # [B, N, 1]
            gated = gate * branch       # [B, N, H]
            gated_outputs.append(gated)

        # Sum gated outputs
        fused = sum(gated_outputs)  # [B, N, H]

        # Final projection
        output = self.fusion_proj(fused)

        return output


class MultiScaleGeoModule(nn.Module):
    """
    Complete Multi-Scale Geographic Representation Module.

    Combines three branches via gated fusion:
      Branch A: Transformer character-level features
      Branch B: Multi-kernel CNN for n-gram features
      Branch C: Span pooling for geographic span features

    Output: Enhanced geo-aware representations that can be used
    alongside or in place of the original encoder outputs.
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_sizes: List[int] = None,
        num_filters_per_kernel: int = 128,
        num_attention_heads: int = 8,
        max_span_length: int = 10,
        dropout: float = 0.3,
    ):
        """
        Args:
            hidden_dim: Hidden dimension of token representations (H)
            kernel_sizes: CNN kernel sizes (default: [2, 3, 4, 5])
            num_filters_per_kernel: CNN filters per kernel
            num_attention_heads: Attention heads for Transformer branch
            max_span_length: Max span length for span pooling branch
            dropout: Dropout rate
        """
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4, 5]

        # Branch A: Transformer character-level features
        self.branch_a = TransformerBranch(hidden_dim, num_attention_heads, dropout)

        # Branch B: Multi-kernel CNN for n-gram features
        self.branch_b = MultiKernelCNN(hidden_dim, kernel_sizes, num_filters_per_kernel, dropout)

        # Branch C: Span pooling for geographic span features
        self.branch_c = SpanPoolingBranch(hidden_dim, max_span_length, dropout)

        # Gated fusion of three branches
        self.gated_fusion = GatedFusion(hidden_dim, num_branches=3, dropout=dropout)

        # Final layer norm for the enhanced representations
        self.output_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        return_branches: bool = False,
    ) -> Tensor | tuple:
        """
        Forward pass through all three branches with gated fusion.

        Args:
            hidden_states: [B, N, H] original encoder output token representations
            attention_mask: [B, N] optional attention mask
            return_branches: If True, return individual branch outputs along with fusion

        Returns:
            If return_branches=False:
                enhanced_repr: [B, N, H] geo-aware enhanced representations
            If return_branches=True:
                (enhanced_repr, branch_outputs) where branch_outputs is a dict
        """
        # Branch A: Transformer character-level features
        branch_a_out = self.branch_a(hidden_states, attention_mask)

        # Branch B: Multi-kernel CNN n-gram features
        branch_b_out = self.branch_b(hidden_states)

        # Branch C: Span pooling geographic features
        branch_c_out = self.branch_c(hidden_states, attention_mask)

        # Gated fusion
        fused = self.gated_fusion([branch_a_out, branch_b_out, branch_c_out])

        # Residual connection + layer norm
        enhanced = self.output_layer_norm(hidden_states + fused)

        if return_branches:
            return enhanced, {
                'branch_a': branch_a_out,
                'branch_b': branch_b_out,
                'branch_c': branch_c_out,
                'fused': fused,
            }
        return enhanced

    def compute_consistency_loss(self, branch_outputs: dict) -> Tensor:
        """
        Compute multi-scale consistency loss between branches.

        Encourages the three branches to produce consistent representations
        while preserving their complementary information.

        Uses cosine similarity between branch representations.

        Args:
            branch_outputs: Dict with 'branch_a', 'branch_b', 'branch_c' tensors [B, N, H]

        Returns:
            consistency_loss: Scalar tensor
        """
        a = F.normalize(branch_outputs['branch_a'], p=2, dim=-1)
        b = F.normalize(branch_outputs['branch_b'], p=2, dim=-1)
        c = F.normalize(branch_outputs['branch_c'], p=2, dim=-1)

        # Pairwise cosine similarities (we want these to be high, so use 1 - cosine_sim)
        loss_ab = 1.0 - (a * b).sum(dim=-1).mean()
        loss_bc = 1.0 - (b * c).sum(dim=-1).mean()
        loss_ac = 1.0 - (a * c).sum(dim=-1).mean()

        consistency_loss = (loss_ab + loss_bc + loss_ac) / 3.0
        return consistency_loss
