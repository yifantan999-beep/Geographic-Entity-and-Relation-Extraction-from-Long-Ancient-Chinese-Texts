"""
Fact Decomposition Module for Wenyanwen Geographical NER + RE.

Inspired by JEDI's two-stage design:

Stage 1: Entity Candidate Extraction (lightweight)
  - Input: single sentence token representations from encoder [B, N, H]
  - Span start/end prediction heads (pointer-network style)
  - Entity type prediction for candidate spans
  - Output: set of candidate entities {e1, e2, ...}

Stage 2: Relation Verification (core)
  - For each entity pair (e_i, e_j) and relation type r:
    - Construct hypothesis h = (e_i, r, e_j)
    - Embed hypothesis into continuous vector space
    - Use Group Bilinear Layer to compute premise-hypothesis interaction
    - Output: 3-class {Entail, Contradict, Neutral}

Key design: the "hypothesis" is NOT a natural language sentence but a structured
triple embedded into a continuous vector space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class SpanPredictionHead(nn.Module):
    """
    Lightweight span prediction head for entity candidate extraction.

    Predicts start/end positions and entity types for each token position.
    Similar to pointer networks.

    Input:  [B, N, H] token representations
    Output: start_logits [B, N, 1], end_logits [B, N, 1], type_logits [B, N, num_types]
    """

    def __init__(self, hidden_dim: int, num_types: int, dropout: float = 0.3):
        """
        Args:
            hidden_dim: Dimension of input token representations (H)
            num_types: Number of entity types
            dropout: Dropout rate
        """
        super().__init__()
        # Start boundary predictor: [B, N, H] -> [B, N, 1]
        self.start_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        # End boundary predictor: [B, N, H] -> [B, N, 1]
        self.end_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Entity type predictor: [B, N, H] -> [B, N, num_types]
        # Predicts the entity type for spans that START at this position
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_types),
        )

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            hidden_states: [B, N, H] token representations from encoder

        Returns:
            start_logits: [B, N, 1] - logits for span start positions
            end_logits:   [B, N, 1] - logits for span end positions
            type_logits:  [B, N, num_types] - logits for entity type at start position
        """
        start_logits = self.start_head(hidden_states)   # [B, N, 1]
        end_logits = self.end_head(hidden_states)        # [B, N, 1]
        type_logits = self.type_head(hidden_states)      # [B, N, num_types]
        return start_logits, end_logits, type_logits


class HypothesisEncoder(nn.Module):
    """
    Encodes a structured triple hypothesis (e_i, r, e_j) into a continuous vector.

    The hypothesis is formed by:
      - Span representation of e_i (mean pooling over span tokens)
      - Learnable relation type embedding for r
      - Span representation of e_j (mean pooling over span tokens)

    These are combined into a fixed-dimension hypothesis vector.
    """

    def __init__(self, hidden_dim: int, num_relations: int, dropout: float = 0.3):
        """
        Args:
            hidden_dim: Dimension of token representations (H)
            num_relations: Number of relation types
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learnable relation type embeddings
        # Each relation type gets a learnable vector of dimension hidden_dim
        self.relation_embeddings = nn.Embedding(num_relations, hidden_dim)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        # Projection to combine [h_ei; r_emb; h_ej] into hypothesis vector
        # Concatenation size: 3 * hidden_dim
        self.hypothesis_proj = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def extract_span_representation(
        self, hidden_states: Tensor, start_positions: Tensor, end_positions: Tensor
    ) -> Tensor:
        """
        Extract span representations by mean-pooling over token range [start, end].

        Args:
            hidden_states: [B, N, H] token representations
            start_positions: [K] or [B, K] start indices of spans
            end_positions: [K] or [B, K] end indices of spans (inclusive)

        Returns:
            span_reprs: [K, H] or [B, K, H] span representations
        """
        # For simplicity, implement batch-agnostic version
        # For each span, gather tokens in [start:end+1] and mean-pool
        batch_size, seq_len, hidden_dim = hidden_states.shape

        if start_positions.dim() == 1:
            # Single batch, multiple spans
            span_reprs = []
            for s, e in zip(start_positions, end_positions):
                span_tokens = hidden_states[0, s:e + 1, :]  # [span_len, H]
                span_repr = span_tokens.mean(dim=0)           # [H]
                span_reprs.append(span_repr)
            return torch.stack(span_reprs, dim=0)              # [K, H]
        else:
            # Batched spans
            K = start_positions.size(1)
            span_reprs = []
            for b in range(batch_size):
                batch_spans = []
                for k in range(K):
                    s = start_positions[b, k].item()
                    e = end_positions[b, k].item()
                    if s >= 0 and e >= s:
                        span_tokens = hidden_states[b, s:e + 1, :]
                        batch_spans.append(span_tokens.mean(dim=0))
                    else:
                        batch_spans.append(torch.zeros(hidden_dim, device=hidden_states.device))
                span_reprs.append(torch.stack(batch_spans, dim=0))
            return torch.stack(span_reprs, dim=0)  # [B, K, H]

    def forward(
        self,
        hidden_states: Tensor,
        head_start: Tensor,
        head_end: Tensor,
        tail_start: Tensor,
        tail_end: Tensor,
        relation_ids: Tensor,
    ) -> Tensor:
        """
        Encode hypothesis for entity pairs.

        Args:
            hidden_states: [B, N, H] token representations (premise)
            head_start: [B, K] start positions of head entities
            head_end: [B, K] end positions of head entities
            tail_start: [B, K] start positions of tail entities
            tail_end: [B, K] end positions of tail entities
            relation_ids: [R] relation type IDs to encode

        Returns:
            hypothesis_vectors: [B, K, R, H] encoded hypothesis representations
        """
        batch_size, K = head_start.shape
        num_relations = relation_ids.size(0)

        # Extract head entity span representations: [B, K, H]
        head_repr = self.extract_span_representation(hidden_states, head_start, head_end)

        # Extract tail entity span representations: [B, K, H]
        tail_repr = self.extract_span_representation(hidden_states, tail_start, tail_end)

        # Get relation embeddings: [R, H]
        rel_emb = self.relation_embeddings(relation_ids)  # [R, H]

        # Expand for all combinations: [B, K, R, H]
        head_repr = head_repr.unsqueeze(2).expand(-1, -1, num_relations, -1)  # [B, K, R, H]
        tail_repr = tail_repr.unsqueeze(2).expand(-1, -1, num_relations, -1)  # [B, K, R, H]
        rel_emb = rel_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, K, -1, -1)  # [B, K, R, H]

        # Concatenate: [h_ei; r_emb; h_ej] -> [B, K, R, 3H]
        combined = torch.cat([head_repr, rel_emb, tail_repr], dim=-1)

        # Project to hypothesis vector: [B, K, R, H]
        hypothesis = self.hypothesis_proj(combined)

        return hypothesis


class GroupBilinearLayer(nn.Module):
    """
    Group Bilinear Layer for premise-hypothesis interaction.

    Splits the hidden dimension into K groups and computes bilinear interaction
    within each group. This is more parameter-efficient than a full bilinear layer.

    Inspired by JEDI's Group Bilinear:
      score_g = (premise_g)^T W_g (hypothesis_g) + U_g [premise_g; hypothesis_g] + b_g

    where g indexes the groups.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_groups: int = 4,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        """
        Args:
            hidden_dim: Dimension of premise/hypothesis vectors (H)
            num_groups: Number of groups to split the hidden dimension into
            num_classes: Number of output classes (3: Entail, Contradict, Neutral)
            dropout: Dropout rate
        """
        super().__init__()
        assert hidden_dim % num_groups == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_groups ({num_groups})"
        )
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.group_dim = hidden_dim // num_groups

        # Bilinear weight matrices: one per group
        # W_g: [group_dim, group_dim] for each group
        self.bilinear_weights = nn.Parameter(
            torch.empty(num_groups, self.group_dim, self.group_dim)
        )
        nn.init.xavier_uniform_(self.bilinear_weights)

        # Linear projection for concatenated features: [2 * group_dim] -> [1]
        self.linear_proj = nn.Linear(2 * self.group_dim, 1, bias=False)

        # Final classification head: [num_groups] -> [num_classes]
        self.classifier = nn.Sequential(
            nn.Linear(num_groups, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, premise: Tensor, hypothesis: Tensor) -> Tensor:
        """
        Compute premise-hypothesis interaction scores.

        Args:
            premise: [B, N, H] premise token representations
            hypothesis: [B, K, R, H] hypothesis vectors

        Returns:
            logits: [B, K, R, 3] classification logits
        """
        batch_size, seq_len, hidden_dim = premise.shape
        B, K, R, H = hypothesis.shape

        # Mean-pool premise over sequence to get a global representation
        # [B, N, H] -> [B, H]
        premise_global = premise.mean(dim=1)

        # Reshape for group-wise computation
        # [B, H] -> [B, num_groups, group_dim]
        premise_groups = premise_global.view(batch_size, self.num_groups, self.group_dim)

        # [B, K, R, H] -> [B, K, R, num_groups, group_dim]
        hypothesis_groups = hypothesis.view(B, K, R, self.num_groups, self.group_dim)

        # Expand premise for all entity pairs: [B, 1, 1, num_groups, group_dim]
        premise_groups = premise_groups.unsqueeze(1).unsqueeze(1)

        # Group bilinear interaction:
        # For each group g: score_g = premise_g^T @ W_g @ hypothesis_g
        # [B, 1, 1, num_groups, group_dim] @ [num_groups, group_dim, group_dim]
        # -> [B, 1, 1, num_groups, group_dim]
        weighted_premise = torch.einsum(
            'bkrgd,gdw->bkrgw',
            premise_groups.expand(-1, K, R, -1, -1),
            self.bilinear_weights
        )
        # [B, K, R, num_groups, group_dim] * [B, K, R, num_groups, group_dim]
        # -> [B, K, R, num_groups]
        group_scores = torch.einsum(
            'bkrgd,bkrgd->bkrg',
            weighted_premise,
            hypothesis_groups
        )

        # Also add concatenation feature for each group
        # [B, K, R, num_groups, 2*group_dim]
        concat_features = torch.cat([
            premise_groups.expand(-1, K, R, -1, -1),
            hypothesis_groups,
        ], dim=-1)
        # [B, K, R, num_groups, 1] -> [B, K, R, num_groups]
        linear_scores = self.linear_proj(concat_features).squeeze(-1)

        # Combined group scores: [B, K, R, num_groups]
        group_scores = group_scores + linear_scores

        # Classify: [B, K, R, num_groups] -> [B, K, R, 3]
        logits = self.classifier(group_scores)

        return logits


class RelationVerifier(nn.Module):
    """
    Relation verification module: for each entity pair, verify whether a relation
    holds using the premise-hypothesis framework.

    Outputs 3-class classification:
      - 0: Entail (relation holds)
      - 1: Contradict (relation does NOT hold)
      - 2: Neutral (uncertain / no information)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_relations: int,
        num_groups: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hypothesis_encoder = HypothesisEncoder(hidden_dim, num_relations, dropout)
        self.group_bilinear = GroupBilinearLayer(hidden_dim, num_groups, num_classes=3, dropout=dropout)
        self.num_relations = num_relations

    def forward(
        self,
        hidden_states: Tensor,
        head_start: Tensor,
        head_end: Tensor,
        tail_start: Tensor,
        tail_end: Tensor,
        relation_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            hidden_states: [B, N, H] premise token representations
            head_start: [B, K] start positions of head entities
            head_end: [B, K] end positions of head entities
            tail_start: [B, K] start positions of tail entities
            tail_end: [B, K] end positions of tail entities
            relation_ids: [R] relation type IDs, defaults to all relations

        Returns:
            logits: [B, K, R, 3] entailment/contradiction/neutral logits
        """
        if relation_ids is None:
            relation_ids = torch.arange(
                self.num_relations,
                device=hidden_states.device,
                dtype=torch.long,
            )

        # Encode hypothesis vectors: [B, K, R, H]
        hypothesis = self.hypothesis_encoder(
            hidden_states, head_start, head_end,
            tail_start, tail_end, relation_ids,
        )

        # Compute premise-hypothesis interaction: [B, K, R, 3]
        logits = self.group_bilinear(hidden_states, hypothesis)

        return logits


class FactDecompositionModule(nn.Module):
    """
    Complete Fact Decomposition Module combining both stages.

    Stage 1: Entity candidate extraction via span prediction
    Stage 2: Relation verification via premise-hypothesis model

    This module can be used standalone or integrated into ITER's iterative pipeline.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_entity_types: int,
        num_relation_types: int,
        num_bilinear_groups: int = 4,
        dropout: float = 0.3,
        max_span_length: int = 30,
    ):
        """
        Args:
            hidden_dim: Hidden dimension of token representations (H)
            num_entity_types: Number of entity types
            num_relation_types: Number of relation types
            num_bilinear_groups: Number of groups for bilinear interaction
            dropout: Dropout rate
            max_span_length: Maximum span length for candidate extraction
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_entity_types = num_entity_types
        self.num_relation_types = num_relation_types
        self.max_span_length = max_span_length

        # Stage 1: Span prediction head
        self.span_head = SpanPredictionHead(hidden_dim, num_entity_types, dropout)

        # Stage 2: Relation verification
        self.relation_verifier = RelationVerifier(
            hidden_dim, num_relation_types, num_bilinear_groups, dropout
        )

        # Boundary refinement: small FFN to refine entity boundaries
        self.boundary_refiner = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # [start_adjustment, end_adjustment]
        )

    def extract_candidates(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        top_k: int = 50,
    ) -> dict:
        """
        Stage 1: Extract entity candidates from hidden states.

        Args:
            hidden_states: [B, N, H] token representations
            attention_mask: [B, N] attention mask
            top_k: Maximum number of candidate spans to extract

        Returns:
            dict with keys:
                - start_logits: [B, N, 1]
                - end_logits: [B, N, 1]
                - type_logits: [B, N, num_types]
                - candidate_starts: [B, K] start positions
                - candidate_ends: [B, K] end positions
                - candidate_type_probs: [B, K, num_types]
        """
        start_logits, end_logits, type_logits = self.span_head(hidden_states)

        # Create mask for valid positions
        valid_mask = attention_mask.bool()

        # For candidate extraction, we use start_logits to find top-K starts
        # and then find the best end for each start within max_span_length
        batch_size, seq_len, _ = start_logits.shape

        # Flatten start scores and mask invalid positions
        start_scores = start_logits.squeeze(-1)  # [B, N]
        start_scores = start_scores.masked_fill(~valid_mask, float('-inf'))

        # Get top-K start positions per batch
        K = min(top_k, seq_len)
        _, top_starts = torch.topk(start_scores, k=K, dim=1)  # [B, K]

        # For each start, find the best end within max_span_length
        # Create a scoring matrix [B, K, max_span_length]
        device = hidden_states.device
        candidate_ends = []
        candidate_type_probs = []

        for b in range(batch_size):
            batch_ends = []
            batch_types = []
            for k in range(K):
                start = top_starts[b, k].item()
                if start < 0 or not valid_mask[b, start]:
                    batch_ends.append(torch.tensor(-1, device=device))
                    batch_types.append(torch.zeros(self.num_entity_types, device=device))
                    continue

                # Look at positions [start, start + max_span_length)
                max_end = min(start + self.max_span_length, seq_len)
                end_scores = end_logits[b, start:max_end, 0]  # [span_len]
                end_scores = end_scores.masked_fill(
                    ~valid_mask[b, start:max_end], float('-inf')
                )

                if end_scores.numel() == 0 or end_scores.max() == float('-inf'):
                    batch_ends.append(torch.tensor(-1, device=device))
                    batch_types.append(torch.zeros(self.num_entity_types, device=device))
                else:
                    best_end_rel = end_scores.argmax().item()
                    batch_ends.append(torch.tensor(start + best_end_rel, device=device))
                    # Type prediction at the start position
                    type_probs = F.softmax(type_logits[b, start, :], dim=-1)
                    batch_types.append(type_probs)

            candidate_ends.append(torch.stack(batch_ends))
            candidate_type_probs.append(torch.stack(batch_types))

        candidate_starts = top_starts  # [B, K]
        candidate_ends = torch.stack(candidate_ends)  # [B, K]
        candidate_type_probs = torch.stack(candidate_type_probs)  # [B, K, num_types]

        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'type_logits': type_logits,
            'candidate_starts': candidate_starts,
            'candidate_ends': candidate_ends,
            'candidate_type_probs': candidate_type_probs,
        }

    def verify_relations(
        self,
        hidden_states: Tensor,
        candidate_starts: Tensor,
        candidate_ends: Tensor,
    ) -> Tensor:
        """
        Stage 2: Verify relations between candidate entity pairs.

        Args:
            hidden_states: [B, N, H] premise token representations
            candidate_starts: [B, K] start positions of candidate entities
            candidate_ends: [B, K] end positions of candidate entities

        Returns:
            relation_logits: [B, K, K, R, 3] verification logits for all entity pairs
                For each pair (i, j) and relation r:
                - dim 0: Entail (relation holds)
                - dim 1: Contradict (relation does NOT hold)
                - dim 2: Neutral
        """
        B, K = candidate_starts.shape

        # Use all candidate entities as both heads and tails
        # [B, K, K, R, 3]
        relation_logits = self.relation_verifier(
            hidden_states,
            candidate_starts,      # head starts [B, K]
            candidate_ends,        # head ends [B, K]
            candidate_starts,      # tail starts [B, K]
            candidate_ends,        # tail ends [B, K]
        )

        return relation_logits

    def refine_boundaries(
        self,
        hidden_states: Tensor,
        candidate_starts: Tensor,
        candidate_ends: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Refine entity boundaries using local context.

        Args:
            hidden_states: [B, N, H] token representations
            candidate_starts: [B, K] start positions
            candidate_ends: [B, K] end positions

        Returns:
            refined_starts: [B, K] refined start positions
            refined_ends: [B, K] refined end positions
        """
        B, K = candidate_starts.shape
        _, N, H = hidden_states.shape

        # Gather representations at start and end positions
        starts_safe = candidate_starts.clamp(min=0)  # [B, K]
        ends_safe = candidate_ends.clamp(min=0)      # [B, K]

        # [B, K, H]
        start_repr = torch.gather(
            hidden_states, dim=1,
            index=starts_safe.unsqueeze(-1).expand(-1, -1, H)
        )
        end_repr = torch.gather(
            hidden_states, dim=1,
            index=ends_safe.unsqueeze(-1).expand(-1, -1, H)
        )

        # Concatenate start and end representations: [B, K, 2H]
        boundary_features = torch.cat([start_repr, end_repr], dim=-1)

        # Predict adjustments: [B, K, 2]
        adjustments = self.boundary_refiner(boundary_features)

        # Apply adjustments (rounded to integers, clamped to valid range)
        start_adj = adjustments[..., 0].round().long()  # [B, K]
        end_adj = adjustments[..., 1].round().long()    # [B, K]

        refined_starts = (candidate_starts + start_adj).clamp(min=0, max=N - 1)
        refined_ends = (candidate_ends + end_adj).clamp(min=0, max=N - 1)

        return refined_starts, refined_ends

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        return_all: bool = False,
    ) -> dict:
        """
        Complete forward pass of the fact decomposition module.

        Args:
            hidden_states: [B, N, H] token representations
            attention_mask: [B, N] attention mask
            return_all: If True, return all intermediate outputs

        Returns:
            dict with:
                - span_outputs: Stage 1 outputs
                - relation_logits: Stage 2 outputs [B, K, K, R, 3]
                - fact_loss: Combined fact decomposition loss (if targets provided)
        """
        # Stage 1: Extract candidates
        span_outputs = self.extract_candidates(hidden_states, attention_mask)

        # Stage 2: Verify relations
        relation_logits = self.verify_relations(
            hidden_states,
            span_outputs['candidate_starts'],
            span_outputs['candidate_ends'],
        )

        output = {
            'span_outputs': span_outputs,
            'relation_logits': relation_logits,
        }

        if return_all:
            output['hidden_states'] = hidden_states

        return output

    def compute_fact_loss(
        self,
        span_outputs: dict,
        target_starts: Tensor,
        target_ends: Tensor,
        target_types: Tensor,
        target_relations: Optional[Tensor] = None,
        relation_logits: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute fact decomposition losses.

        Args:
            span_outputs: Stage 1 outputs from extract_candidates()
            target_starts: [B, K_target] ground truth start positions
            target_ends: [B, K_target] ground truth end positions
            target_types: [B, K_target] ground truth entity type IDs
            target_relations: [B, K_target, K_target] ground truth relation labels
            relation_logits: [B, K, K, R, 3] relation verification logits

        Returns:
            span_loss: Span prediction loss (BCE for boundaries + CE for types)
            relation_loss: Relation verification loss
            total_loss: Combined fact decomposition loss
        """
        batch_size = target_starts.size(0)

        # Span boundary loss (binary cross-entropy)
        start_loss = F.binary_cross_entropy_with_logits(
            span_outputs['start_logits'].squeeze(-1),
            target_starts.float().sum(dim=-1).clamp(max=1).unsqueeze(-1).expand(-1, span_outputs['start_logits'].size(1)).float(),
            reduction='mean',
        )

        # Entity type loss (cross-entropy)
        # We only compute type loss for positions that are actually entity starts
        type_logits_flat = span_outputs['type_logits'].reshape(-1, span_outputs['type_logits'].size(-1))
        type_targets_flat = target_types.reshape(-1)
        valid_type_mask = type_targets_flat >= 0
        if valid_type_mask.any():
            type_loss = F.cross_entropy(
                type_logits_flat[valid_type_mask],
                type_targets_flat[valid_type_mask].clamp(min=0),
                reduction='mean',
            )
        else:
            type_loss = torch.tensor(0.0, device=span_outputs['start_logits'].device)

        span_loss = start_loss + type_loss

        # Relation verification loss
        relation_loss = torch.tensor(0.0, device=span_outputs['start_logits'].device)
        if target_relations is not None and relation_logits is not None:
            # target_relations: [B, K_target, K_target] with values in {0, 1, 2}
            #   0 = Entail (relation holds), 1 = Contradict, 2 = Neutral
            # We need to map to the right pairs
            B, K, _, R, _ = relation_logits.shape
            K_target = target_relations.size(1)

            # For now, use a simplified loss: align top-K with target entities
            # Reshape and compute cross-entropy
            min_K = min(K, K_target)
            if min_K > 0:
                rel_logits_subset = relation_logits[:, :min_K, :min_K, :, :]  # [B, minK, minK, R, 3]
                rel_targets_subset = target_relations[:, :min_K, :min_K]       # [B, minK, minK]

                # For each relation type, we want to predict entail/contradict/neutral
                # Simplified: average over relation types
                # [B, minK, minK, R, 3] -> [B, minK, minK, 3]
                avg_logits = rel_logits_subset.mean(dim=3)
                relation_loss = F.cross_entropy(
                    avg_logits.reshape(-1, 3),
                    rel_targets_subset.reshape(-1).clamp(min=0, max=2),
                    reduction='mean',
                )

        total_loss = span_loss + relation_loss
        return span_loss, relation_loss, total_loss
