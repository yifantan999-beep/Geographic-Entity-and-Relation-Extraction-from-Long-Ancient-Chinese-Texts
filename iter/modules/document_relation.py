"""Document-level relation evidence aggregation for windowed ITER models."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

CONFIG_NAME = "document_relation_config.json"
WEIGHTS_NAME = "document_relation.safetensors"


@dataclass
class DocumentRelationConfig:
    hidden_size: int
    entity_types: list[str]
    relation_types: list[str]
    dropout: float = 0.2
    type_embedding_size: int = 64
    distance_embedding_size: int = 32
    type_prior_weight: float = 0.5
    evidence_top_k: int = 2
    threshold: float = 0.5
    fusion_mode: str = "fixed"
    fusion_hidden_size: int = 32
    fusion_relation_embedding_size: int = 8
    # Optional long-context extensions.  They are disabled by default so
    # existing document-relation checkpoints remain loadable.
    distance_conditioned_fusion: bool = False
    fusion_long_distance_boundary: int = 128
    fusion_distance_hidden_size: int = 16
    fusion_long_distance_bias: float = 0.5
    long_distance_head: bool = False
    long_distance_boundary: int = 128
    long_distance_head_weight: float = 0.5
    distance_experts: bool = False
    distance_expert_boundaries: list[int] = field(default_factory=lambda: [32, 64, 128])
    distance_expert_temperature: float = 0.75
    distance_expert_weight: float = 0.5
    distance_expert_active_buckets: list[int] = field(default_factory=list)
    pair_type_compatibility: bool = False
    pair_type_compatibility_bias_weight: float = 0.25
    pair_type_focus_pairs: list[list[str]] = field(
        default_factory=lambda: [["学校", "园林"], ["行政区划", "水系"], ["城市内部区划", "街道"]]
    )
    pair_type_focus_weight: float = 2.0
    # Structural features for the long-context v5 head.  They are additive
    # residuals and therefore do not change the legacy pair projection shape.
    use_direction_embedding: bool = False
    use_window_position_encoding: bool = False
    positive_focal_loss_weight: float = 0.0
    positive_focal_gamma: float = 1.5
    long_distance_sampling_weight: float = 0.0
    focus_relation_names: list[str] = field(default_factory=list)
    focus_relation_positive_weight: float = 1.0
    relation_adapter: bool = False
    relation_adapter_size: int = 128
    relation_adapter_weight: float = 0.25
    recall_head: bool = False
    recall_head_size: int = 128
    recall_head_weight: float = 0.25
    recall_margin: float = 0.35
    recall_threshold: float = 0.24
    # Optional pair-existence gate.  Disabled by default to preserve all
    # existing long-context checkpoints and predictions.
    pair_link_head: bool = False
    pair_link_bias: float = 0.0
    # Interpolate pair-link suppression.  1.0 reproduces the original hard
    # multiplicative gate; values below 1.0 recover relation recall for
    # borderline pairs in IND-specific experiments.
    pair_link_gate_weight: float = 1.0
    pair_link_positive_weight: float = 1.0
    distance_boundaries: list[int] = field(
        default_factory=lambda: [4, 8, 16, 32, 64, 128, 256, 512]
    )
    local_checkpoint: str | None = None
    dataset_config: str | None = None

    @property
    def num_entity_types(self) -> int:
        return len(self.entity_types)

    @property
    def num_relation_types(self) -> int:
        return len(self.relation_types)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / CONFIG_NAME).open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: Path) -> "DocumentRelationConfig":
        with (directory / CONFIG_NAME).open(encoding="utf-8") as handle:
            return cls(**json.load(handle))


@dataclass
class DocumentRelationOutput:
    pairs: Tensor
    logits: Tensor
    probabilities: Tensor
    evidence_attention: Tensor
    fused_logits: Tensor | None = None
    fused_probabilities: Tensor | None = None
    fusion_gate: Tensor | None = None
    pair_type_logits: Tensor | None = None
    long_distance_logits: Tensor | None = None
    distance_expert_logits: Tensor | None = None
    distance_expert_weights: Tensor | None = None
    relation_adapter_logits: Tensor | None = None
    recall_head_logits: Tensor | None = None
    pair_link_logits: Tensor | None = None


class DocumentRelationModel(nn.Module):
    """Global relation head with pair-specific window evidence attention."""

    def __init__(self, config: DocumentRelationConfig):
        super().__init__()
        if config.fusion_mode not in {"fixed", "learned"}:
            raise ValueError(f"Unsupported document relation fusion mode: {config.fusion_mode}")
        self.config = config
        if not 0.0 <= config.pair_link_gate_weight <= 1.0:
            raise ValueError("pair_link_gate_weight must be between 0 and 1")
        if config.pair_link_positive_weight <= 0.0:
            raise ValueError("pair_link_positive_weight must be positive")
        hidden = config.hidden_size
        type_dim = config.type_embedding_size
        distance_dim = config.distance_embedding_size

        self.entity_projection = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.type_embedding = nn.Embedding(config.num_entity_types, type_dim)
        self.distance_embedding = nn.Embedding(len(config.distance_boundaries) + 1, distance_dim)
        pair_input_size = hidden * 4 + type_dim * 2 + distance_dim
        self.pair_projection = nn.Sequential(
            nn.Linear(pair_input_size, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(hidden),
        )
        if self.config.use_direction_embedding:
            self.direction_embedding = nn.Embedding(3, hidden)
            self.direction_projection = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)
            )
        if self.config.use_window_position_encoding:
            self.window_position_projection = nn.Sequential(
                nn.Linear(2, hidden), nn.GELU(), nn.LayerNorm(hidden)
            )
        self.evidence_query = nn.Linear(hidden, hidden, bias=False)
        self.evidence_key = nn.Linear(hidden, hidden, bias=False)
        self.evidence_value = nn.Linear(hidden, hidden, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.num_relation_types),
        )
        if config.pair_link_head:
            self.pair_link_classifier = nn.Sequential(
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, max(16, hidden // 2)),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(max(16, hidden // 2), 1),
            )
            # Start conservatively: the gate is learned during the IND-only
            # experiment and has no effect when pair_link_head=False.
            nn.init.constant_(self.pair_link_classifier[-1].bias, config.pair_link_bias)
        if config.relation_adapter:
            adapter_size = max(16, int(config.relation_adapter_size))
            self.relation_adapter_projection = nn.Sequential(
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, adapter_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.relation_adapter_embedding = nn.Parameter(
                torch.zeros(config.num_relation_types, adapter_size)
            )
            self.relation_adapter_bias = nn.Parameter(
                torch.zeros(config.num_relation_types)
            )
        if config.recall_head:
            recall_size = max(16, int(config.recall_head_size))
            self.recall_head = nn.Sequential(
                nn.LayerNorm(hidden * 2),
                nn.Linear(hidden * 2, recall_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(recall_size, config.num_relation_types),
            )
            # Start as a no-op so continuation training is isolated to the
            # recall objective until this head learns useful residuals.
            nn.init.zeros_(self.recall_head[-1].weight)
            nn.init.zeros_(self.recall_head[-1].bias)
        if config.long_distance_head:
            self.long_distance_classifier = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, config.num_relation_types),
            )
            # The auxiliary head starts as a no-op.  Its contribution is
            # learned during the dedicated experiment instead of changing
            # the baseline predictions at initialization.
            nn.init.zeros_(self.long_distance_classifier[-1].weight)
            nn.init.zeros_(self.long_distance_classifier[-1].bias)
        if config.distance_experts:
            if sorted(config.distance_expert_boundaries) != config.distance_expert_boundaries:
                raise ValueError("distance_expert_boundaries must be sorted")
            if not config.distance_expert_boundaries or config.distance_expert_boundaries[0] <= 0:
                raise ValueError("distance_expert_boundaries must contain positive values")
            expert_count = len(config.distance_expert_boundaries) + 1
            if any(
                bucket < 0 or bucket >= expert_count
                for bucket in config.distance_expert_active_buckets
            ):
                raise ValueError("distance_expert_active_buckets contains an invalid bucket")
            self.distance_expert_classifiers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden * 2, hidden),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(hidden, config.num_relation_types),
                    )
                    for _ in range(expert_count)
                ]
            )
            self.distance_expert_gate = nn.Sequential(
                nn.Linear(2, max(8, config.fusion_distance_hidden_size)),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(max(8, config.fusion_distance_hidden_size), expert_count),
            )
            nn.init.zeros_(self.distance_expert_gate[-1].weight)
            nn.init.zeros_(self.distance_expert_gate[-1].bias)
            for classifier in self.distance_expert_classifiers:
                nn.init.zeros_(classifier[-1].weight)
                nn.init.zeros_(classifier[-1].bias)
        if config.pair_type_compatibility:
            self.pair_type_compatibility = nn.Parameter(
                torch.zeros(
                    config.num_entity_types,
                    config.num_entity_types,
                    config.num_relation_types,
                )
            )
        else:
            self.register_parameter("pair_type_compatibility", None)
        if config.fusion_mode == "learned":
            relation_dim = config.fusion_relation_embedding_size
            self.fusion_relation_embedding = nn.Embedding(config.num_relation_types, relation_dim)
            self.fusion_gate = nn.Sequential(
                nn.Linear(7 + relation_dim, config.fusion_hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.fusion_hidden_size, 1),
            )
            # Start from an unbiased 50/50 logit mixture and let train/eval
            # evidence determine whether the local or global branch dominates.
            nn.init.zeros_(self.fusion_gate[-1].weight)
            nn.init.zeros_(self.fusion_gate[-1].bias)
            if config.distance_conditioned_fusion:
                self.distance_fusion_gate = nn.Sequential(
                    nn.Linear(2, config.fusion_distance_hidden_size),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.fusion_distance_hidden_size, 1),
                )
                nn.init.zeros_(self.distance_fusion_gate[-1].weight)
                nn.init.zeros_(self.distance_fusion_gate[-1].bias)

        self.register_buffer(
            "distance_boundaries",
            torch.tensor(config.distance_boundaries, dtype=torch.float),
        )
        self.register_buffer(
            "compatibility_bias",
            torch.zeros(
                config.num_entity_types,
                config.num_entity_types,
                config.num_relation_types,
            ),
        )
        self.register_buffer(
            "class_priors",
            torch.full((config.num_relation_types,), 1e-3),
        )
        focus_matrix = torch.ones(
            config.num_entity_types,
            config.num_entity_types,
            dtype=torch.float,
        )
        if config.pair_type_focus_weight < 1.0:
            raise ValueError("pair_type_focus_weight must be at least 1")
        entity_to_id = {name: idx for idx, name in enumerate(config.entity_types)}
        for pair in config.pair_type_focus_pairs:
            if len(pair) != 2:
                raise ValueError(f"pair_type_focus_pairs entries must have two names: {pair}")
            head_name, tail_name = pair
            if head_name not in entity_to_id or tail_name not in entity_to_id:
                # Configs may be shared across datasets with a subset of
                # labels.  Ignore unavailable focus pairs rather than making
                # old experiments fail at model construction.
                continue
            head_id, tail_id = entity_to_id[head_name], entity_to_id[tail_name]
            focus_matrix[head_id, tail_id] = config.pair_type_focus_weight
            focus_matrix[tail_id, head_id] = config.pair_type_focus_weight
        self.register_buffer("pair_type_focus_matrix", focus_matrix)
        relation_focus_weights = torch.ones(config.num_relation_types, dtype=torch.float)
        if config.focus_relation_positive_weight < 1.0:
            raise ValueError("focus_relation_positive_weight must be at least 1")
        relation_to_id = {name: idx for idx, name in enumerate(config.relation_types)}
        for name in config.focus_relation_names:
            if name in relation_to_id:
                relation_focus_weights[relation_to_id[name]] = config.focus_relation_positive_weight
        self.register_buffer("relation_focus_weights", relation_focus_weights)

    def set_statistics(self, compatibility_bias: Tensor, class_priors: Tensor) -> None:
        if compatibility_bias.shape != self.compatibility_bias.shape:
            raise ValueError(
                f"compatibility_bias shape {tuple(compatibility_bias.shape)} does not match "
                f"{tuple(self.compatibility_bias.shape)}"
            )
        if class_priors.shape != self.class_priors.shape:
            raise ValueError("class_priors shape does not match relation type count")
        self.compatibility_bias.copy_(compatibility_bias)
        self.class_priors.copy_(class_priors.clamp(1e-5, 0.5))

    @staticmethod
    def all_ordered_pairs(num_entities: int, device: torch.device | str) -> Tensor:
        if num_entities < 2:
            return torch.empty((0, 2), dtype=torch.long, device=device)
        indices = torch.arange(num_entities, device=device)
        head = indices.repeat_interleave(num_entities)
        tail = indices.repeat(num_entities)
        return torch.stack((head[head != tail], tail[head != tail]), dim=-1)

    def forward(
        self,
        entity_repr: Tensor,
        entity_type_probs: Tensor,
        entity_positions: Tensor,
        window_repr: Tensor,
        window_ranges: Tensor,
        pairs: Tensor | None = None,
        local_probabilities: Tensor | None = None,
        local_available: Tensor | None = None,
    ) -> DocumentRelationOutput:
        device = entity_repr.device
        pairs = self.all_ordered_pairs(entity_repr.size(0), device) if pairs is None else pairs
        if pairs.numel() == 0:
            empty_logits = entity_repr.new_empty((0, self.config.num_relation_types))
            empty_attention = entity_repr.new_empty((0, window_repr.size(0)))
            return DocumentRelationOutput(
                pairs=pairs,
                logits=empty_logits,
                probabilities=empty_logits,
                evidence_attention=empty_attention,
                fused_logits=empty_logits if self.config.fusion_mode == "learned" else None,
                fused_probabilities=empty_logits if self.config.fusion_mode == "learned" else None,
                fusion_gate=empty_logits if self.config.fusion_mode == "learned" else None,
                pair_type_logits=(
                    empty_logits if self.config.pair_type_compatibility else None
                ),
                long_distance_logits=(
                    empty_logits if self.config.long_distance_head else None
                ),
                distance_expert_logits=(
                    empty_logits.reshape(0, 1, self.config.num_relation_types)
                    .expand(0, len(self.config.distance_expert_boundaries) + 1, -1)
                    if self.config.distance_experts
                    else None
                ),
                distance_expert_weights=(
                    empty_logits.new_empty(
                        (0, len(self.config.distance_expert_boundaries) + 1)
                    )
                    if self.config.distance_experts
                    else None
                ),
                relation_adapter_logits=(
                    empty_logits if self.config.relation_adapter else None
                ),
                pair_link_logits=(empty_logits.new_empty((0,)) if self.config.pair_link_head else None),
            )

        if window_repr.size(0) == 0:
            raise ValueError("At least one encoded window is required")
        if entity_type_probs.shape != (entity_repr.size(0), self.config.num_entity_types):
            raise ValueError("entity_type_probs has an incompatible shape")

        head_idx, tail_idx = pairs[:, 0], pairs[:, 1]
        encoded_entities = self.entity_projection(entity_repr)
        head = encoded_entities[head_idx]
        tail = encoded_entities[tail_idx]
        head_types = entity_type_probs[head_idx]
        tail_types = entity_type_probs[tail_idx]
        head_type_repr = head_types @ self.type_embedding.weight
        tail_type_repr = tail_types @ self.type_embedding.weight

        distances = (entity_positions[head_idx] - entity_positions[tail_idx]).abs()
        distance_ids = torch.bucketize(distances, self.distance_boundaries)
        distance_repr = self.distance_embedding(distance_ids)
        pair_input = torch.cat(
            (head, tail, (head - tail).abs(), head * tail, head_type_repr, tail_type_repr, distance_repr),
            dim=-1,
        )
        pair_repr = self.pair_projection(pair_input)
        if self.config.use_direction_embedding:
            # 0 = head before tail, 1 = same position, 2 = head after tail.
            direction = torch.ones_like(distances, dtype=torch.long)
            signed = entity_positions[tail_idx] - entity_positions[head_idx]
            direction = torch.where(signed > 0, torch.zeros_like(direction), direction)
            direction = torch.where(signed < 0, torch.full_like(direction, 2), direction)
            pair_repr = pair_repr + self.direction_projection(self.direction_embedding(direction))

        evidence_keys = window_repr
        if self.config.use_window_position_encoding:
            starts, ends = window_ranges[:, 0], window_ranges[:, 1]
            document_end = ends.max().clamp_min(1.0)
            position_features = torch.stack(
                (
                    ((starts + ends) * 0.5) / document_end,
                    (ends - starts) / document_end,
                ),
                dim=-1,
            )
            evidence_keys = evidence_keys + self.window_position_projection(position_features)
        evidence_scores = self.evidence_query(pair_repr) @ self.evidence_key(evidence_keys).T
        evidence_scores = evidence_scores / math.sqrt(self.config.hidden_size)
        window_start, window_end = window_ranges[:, 0], window_ranges[:, 1]
        head_pos = entity_positions[head_idx].unsqueeze(1)
        tail_pos = entity_positions[tail_idx].unsqueeze(1)
        covers_head = (window_start.unsqueeze(0) <= head_pos) & (head_pos < window_end.unsqueeze(0))
        covers_tail = (window_start.unsqueeze(0) <= tail_pos) & (tail_pos < window_end.unsqueeze(0))
        evidence_scores = evidence_scores + covers_head.to(evidence_scores.dtype)
        evidence_scores = evidence_scores + covers_tail.to(evidence_scores.dtype)
        evidence_attention = evidence_scores.softmax(dim=-1)
        evidence = evidence_attention @ self.evidence_value(window_repr)

        pair_context = torch.cat((pair_repr, evidence), dim=-1)
        logits = self.classifier(pair_context)
        pair_link_logits = None
        if self.config.pair_link_head:
            pair_link_logits = self.pair_link_classifier(pair_context).squeeze(-1)
        relation_adapter_logits = None
        if self.config.relation_adapter:
            adapter_repr = self.relation_adapter_projection(pair_context)
            relation_adapter_logits = (
                adapter_repr @ self.relation_adapter_embedding.T
                + self.relation_adapter_bias
            )
            logits = logits + self.config.relation_adapter_weight * relation_adapter_logits
        recall_head_logits = None
        if self.config.recall_head:
            recall_head_logits = self.recall_head(pair_context)
            logits = logits + self.config.recall_head_weight * recall_head_logits
        type_bias = torch.einsum(
            "pa,pb,abr->pr", head_types, tail_types, self.compatibility_bias
        )
        logits = logits + self.config.type_prior_weight * type_bias
        pair_type_logits = None
        if self.pair_type_compatibility is not None:
            pair_type_logits = torch.einsum(
                "pa,pb,abr->pr", head_types, tail_types, self.pair_type_compatibility
            )
            logits = logits + self.config.pair_type_compatibility_bias_weight * pair_type_logits

        long_distance_logits = None
        if self.config.long_distance_head:
            long_distance_logits = self.long_distance_classifier(pair_context)
            long_mask = (distances >= self.config.long_distance_boundary).to(logits.dtype)
            logits = logits + (
                self.config.long_distance_head_weight
                * long_distance_logits
                * long_mask.unsqueeze(-1)
            )
        distance_expert_logits = None
        distance_expert_weights = None
        if self.config.distance_experts:
            distance_expert_logits = torch.stack(
                [classifier(pair_context) for classifier in self.distance_expert_classifiers],
                dim=1,
            )
            boundaries = distances.new_tensor(
                self.config.distance_expert_boundaries, dtype=torch.float
            )
            lower = torch.cat((boundaries.new_zeros(1), boundaries))
            upper = torch.cat((boundaries, boundaries[-1:].mul(2.0)))
            centers = 0.5 * (lower + upper)
            scales = (upper - lower).clamp_min(1.0)
            continuous_prior = -(
                (distances.float().unsqueeze(-1) - centers.unsqueeze(0))
                / scales.unsqueeze(0)
            ).pow(2) / max(self.config.distance_expert_temperature, 1e-4)
            max_distance = max(float(self.config.distance_expert_boundaries[-1] * 2), 1.0)
            normalized = torch.log1p(distances.float()) / math.log1p(max_distance)
            gate_features = torch.stack(
                (normalized, normalized.square()), dim=-1
            )
            learned_gate = self.distance_expert_gate(gate_features)
            distance_expert_weights = (continuous_prior + learned_gate).softmax(dim=-1)
            expert_residual = (
                distance_expert_logits * distance_expert_weights.unsqueeze(-1)
            ).sum(dim=1)
            if self.config.distance_expert_active_buckets:
                active = torch.zeros_like(distances, dtype=torch.bool)
                bucket_ids = torch.bucketize(distances, boundaries)
                for bucket_id in self.config.distance_expert_active_buckets:
                    active |= bucket_ids.eq(bucket_id)
                expert_residual = expert_residual * active.to(logits.dtype).unsqueeze(-1)
            logits = logits + self.config.distance_expert_weight * expert_residual
        fused_logits = None
        fused_probabilities = None
        fusion_gate = None
        if self.config.fusion_mode == "learned":
            if local_probabilities is None or local_available is None:
                raise ValueError(
                    "learned fusion requires local_probabilities and local_available"
                )
            fused_logits, fusion_gate = self.fuse_logits(
                global_logits=logits,
                local_probabilities=local_probabilities,
                local_available=local_available,
                pairs=pairs,
                entity_type_probs=entity_type_probs,
                entity_positions=entity_positions,
                evidence_attention=evidence_attention,
            )
            fused_probabilities = fused_logits.sigmoid()
        return DocumentRelationOutput(
            pairs=pairs,
            logits=logits,
            probabilities=logits.sigmoid(),
            evidence_attention=evidence_attention,
            fused_logits=fused_logits,
            fused_probabilities=fused_probabilities,
            fusion_gate=fusion_gate,
            pair_type_logits=pair_type_logits,
            long_distance_logits=long_distance_logits,
            distance_expert_logits=distance_expert_logits,
            distance_expert_weights=distance_expert_weights,
            relation_adapter_logits=relation_adapter_logits,
            recall_head_logits=recall_head_logits,
            pair_link_logits=pair_link_logits,
        )

    def apply_pair_link_gate(self, probabilities: Tensor, pair_link_logits: Tensor) -> Tensor:
        """Apply the configurable pair-existence gate to relation probabilities."""
        if probabilities.ndim != 2 or pair_link_logits.ndim != 1:
            raise ValueError("pair-link gate expects [pairs, relations] and [pairs] tensors")
        if probabilities.size(0) != pair_link_logits.size(0):
            raise ValueError("pair-link logits and relation probabilities have incompatible sizes")
        link_probability = pair_link_logits.sigmoid().unsqueeze(-1)
        weight = float(self.config.pair_link_gate_weight)
        effective_link = (1.0 - weight) + weight * link_probability
        return probabilities * effective_link

    def fuse_logits(
        self,
        global_logits: Tensor,
        local_probabilities: Tensor,
        local_available: Tensor,
        pairs: Tensor,
        entity_type_probs: Tensor,
        entity_positions: Tensor,
        evidence_attention: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Fuse local and global relation logits with pair-specific learned gates."""
        if self.config.fusion_mode != "learned":
            raise RuntimeError("fuse_logits is only available when fusion_mode='learned'")
        if local_probabilities.shape != global_logits.shape:
            raise ValueError("local_probabilities must have the same shape as global_logits")
        if local_available.shape != global_logits.shape:
            raise ValueError("local_available must have the same shape as global_logits")

        available = local_available.bool()
        local_logits = torch.logit(local_probabilities.clamp(1e-5, 1.0 - 1e-5))
        head_idx, tail_idx = pairs[:, 0], pairs[:, 1]
        distances = (entity_positions[head_idx] - entity_positions[tail_idx]).abs()
        distance_scale = math.log1p(max(self.config.distance_boundaries[-1], 1))
        normalized_distance = torch.log1p(distances) / distance_scale
        head_confidence = entity_type_probs[head_idx].amax(dim=-1)
        tail_confidence = entity_type_probs[tail_idx].amax(dim=-1)
        evidence_quality = evidence_attention.amax(dim=-1)

        pair_features = torch.stack(
            (
                normalized_distance,
                head_confidence,
                tail_confidence,
                evidence_quality,
            ),
            dim=-1,
        ).unsqueeze(1).expand(-1, self.config.num_relation_types, -1)
        relation_ids = torch.arange(
            self.config.num_relation_types, device=global_logits.device
        )
        relation_features = self.fusion_relation_embedding(relation_ids)
        relation_features = relation_features.unsqueeze(0).expand(global_logits.size(0), -1, -1)
        score_features = torch.stack(
            (
                (global_logits.clamp(-8.0, 8.0) / 8.0),
                (local_logits.clamp(-8.0, 8.0) / 8.0),
                available.to(global_logits.dtype),
            ),
            dim=-1,
        )
        gate_input = torch.cat((pair_features, score_features, relation_features), dim=-1)
        local_gate_logits = self.fusion_gate(gate_input).squeeze(-1)
        if self.config.distance_conditioned_fusion:
            long_distance = (
                distances >= self.config.fusion_long_distance_boundary
            ).to(global_logits.dtype)
            distance_features = torch.stack(
                (normalized_distance, long_distance), dim=-1
            )
            # A positive boost is subtracted from the local-logit branch so
            # long-distance pairs rely more on document-level evidence.  The
            # learned term is constrained positive and only active beyond the
            # configured long-distance boundary.
            learned_boost = F.softplus(
                self.distance_fusion_gate(distance_features).squeeze(-1)
            )
            fixed_boost = F.softplus(
                global_logits.new_tensor(self.config.fusion_long_distance_bias)
            )
            local_gate_logits = local_gate_logits - long_distance.unsqueeze(-1) * (
                fixed_boost + 0.5 * learned_boost.unsqueeze(-1)
            )
        local_weight = local_gate_logits.sigmoid()
        fused = local_weight * local_logits + (1.0 - local_weight) * global_logits
        fused = torch.where(available, fused, global_logits)
        local_weight = torch.where(available, local_weight, torch.zeros_like(local_weight))
        return fused, local_weight

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        self.config.save(directory)
        save_file({name: tensor.detach().cpu().contiguous() for name, tensor in self.state_dict().items()}, directory / WEIGHTS_NAME)

    @classmethod
    def from_pretrained(cls, directory: str | Path, device: str | torch.device = "cpu") -> "DocumentRelationModel":
        directory = Path(directory)
        model = cls(DocumentRelationConfig.load(directory))
        missing, unexpected = model.load_state_dict(
            load_file(directory / WEIGHTS_NAME, device=str(device)), strict=False
        )
        allowed_missing = {"relation_focus_weights"}
        if set(missing) - allowed_missing or unexpected:
            raise RuntimeError(
                "Document relation checkpoint incompatibility: "
                f"missing={sorted(set(missing) - allowed_missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        return model.to(device)


def non_negative_pu_loss(logits: Tensor, targets: Tensor, class_priors: Tensor) -> Tensor:
    """Multi-label non-negative positive-unlabeled risk.

    A zero target is treated as unlabeled, not as a certain negative. Classes
    absent from a minibatch receive a small unlabeled-only stabilizing loss.
    """
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    if logits.numel() == 0:
        return logits.sum()
    risks = []
    for relation_id in range(logits.size(1)):
        scores = logits[:, relation_id]
        positive = targets[:, relation_id].bool()
        unlabeled = ~positive
        if positive.any():
            prior = class_priors[relation_id]
            positive_risk = prior * F.softplus(-scores[positive]).mean()
            positive_as_negative = F.softplus(scores[positive]).mean()
            if unlabeled.any():
                unlabeled_negative = F.softplus(scores[unlabeled]).mean()
                negative_risk = unlabeled_negative - prior * positive_as_negative
                risks.append(positive_risk + negative_risk.clamp_min(0.0))
            else:
                risks.append(positive_risk)
        else:
            risks.append(0.05 * F.softplus(scores).mean())
    return torch.stack(risks).mean()


def relation_positive_focal_loss(
    logits: Tensor,
    targets: Tensor,
    entity_positions: Tensor,
    pairs: Tensor,
    long_distance_boundary: int = 128,
    long_distance_weight: float = 2.0,
    gamma: float = 1.5,
    relation_weights: Tensor | None = None,
) -> Tensor:
    """Recall-oriented positive loss for sparse, long-distance relations.

    PU risk is retained for the unlabeled pool, while this term explicitly
    prevents rare positive relations from being washed out by the quadratic
    candidate set.  Positives beyond ``long_distance_boundary`` receive an
    additional weight; no unlabeled pair is treated as a hard negative here.
    """
    if logits.shape != targets.shape or pairs.size(0) != logits.size(0):
        raise ValueError("positive focal tensors have incompatible shapes")
    positive = targets > 0
    if not positive.any():
        return logits.sum() * 0.0
    distances = (entity_positions[pairs[:, 0]] - entity_positions[pairs[:, 1]]).abs()
    distance_weight = torch.where(
        distances >= long_distance_boundary,
        logits.new_tensor(float(long_distance_weight)),
        logits.new_tensor(1.0),
    ).unsqueeze(-1)
    probabilities = logits.sigmoid()
    per_entry = F.softplus(-logits) * (1.0 - probabilities).pow(max(0.0, gamma))
    weights = distance_weight.expand_as(logits)
    if relation_weights is not None:
        if relation_weights.shape != (logits.size(1),):
            raise ValueError("relation_weights must contain one value per relation type")
        weights = weights * relation_weights.to(logits).unsqueeze(0)
    return (per_entry * positive.to(logits.dtype) * weights).sum() / positive.sum().clamp_min(1)


def relation_recall_margin_loss(
    logits: Tensor,
    targets: Tensor,
    entity_positions: Tensor,
    pairs: Tensor,
    *,
    threshold: float = 0.24,
    margin: float = 0.35,
    short_distance_boundary: int = 64,
    short_distance_weight: float = 1.5,
    relation_weights: Tensor | None = None,
) -> Tensor:
    """Push labelled relations above a fixed inference threshold.

    The document head is evaluated with one global probability cutoff.  PU
    risk improves ranking but can leave sparse positives below that cutoff.
    This term supplies a threshold-aware positive margin, with extra weight
    for short/mid-distance pairs where the current diagnostic set has the
    weakest recall.  Unlabelled pairs are deliberately excluded.
    """
    if logits.shape != targets.shape or pairs.size(0) != logits.size(0):
        raise ValueError("recall margin tensors have incompatible shapes")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    if margin < 0.0 or short_distance_weight <= 0.0:
        raise ValueError("margin must be non-negative and short distance weight positive")
    positive = targets > 0
    if not positive.any():
        return logits.sum() * 0.0
    distances = (entity_positions[pairs[:, 0]] - entity_positions[pairs[:, 1]]).abs()
    row_weights = torch.where(
        distances < short_distance_boundary,
        logits.new_tensor(float(short_distance_weight)),
        logits.new_tensor(1.0),
    ).unsqueeze(-1).expand_as(logits)
    if relation_weights is not None:
        if relation_weights.shape != (logits.size(1),):
            raise ValueError("relation_weights must contain one value per relation type")
        row_weights = row_weights * relation_weights.to(logits).unsqueeze(0)
    target_logit = torch.logit(logits.new_tensor(float(threshold))) + float(margin)
    penalty = F.softplus(target_logit - logits)
    return (penalty * positive.to(logits.dtype) * row_weights).sum() / (
        positive.to(logits.dtype) * row_weights
    ).sum().clamp_min(1.0)


def distance_expert_pu_loss(
    expert_logits: Tensor,
    targets: Tensor,
    entity_positions: Tensor,
    pairs: Tensor,
    boundaries: list[int],
    class_priors: Tensor,
    active_buckets: list[int] | None = None,
) -> Tensor:
    """Supervise each distance expert on its primary distance interval."""
    expected_experts = len(boundaries) + 1
    if expert_logits.ndim != 3 or expert_logits.size(1) != expected_experts:
        raise ValueError("expert_logits shape does not match distance boundaries")
    if expert_logits.size(0) != targets.size(0) or expert_logits.size(2) != targets.size(1):
        raise ValueError("expert logits and targets have incompatible shapes")
    distances = (entity_positions[pairs[:, 0]] - entity_positions[pairs[:, 1]]).abs()
    bucket_ids = torch.bucketize(
        distances, distances.new_tensor(boundaries, dtype=distances.dtype)
    )
    losses = []
    for expert_id in range(expected_experts):
        if active_buckets and expert_id not in active_buckets:
            continue
        selected = bucket_ids.eq(expert_id)
        if selected.any():
            losses.append(
                non_negative_pu_loss(
                    expert_logits[selected, expert_id], targets[selected], class_priors
                )
            )
    return torch.stack(losses).mean() if losses else expert_logits.sum() * 0.0


def pair_type_compatibility_loss(
    logits: Tensor,
    targets: Tensor,
    head_type_probs: Tensor,
    tail_type_probs: Tensor,
    focus_matrix: Tensor,
    class_priors: Tensor,
) -> Tensor:
    """Train relation scores to respect the entity-type pair compatibility.

    The regularizer uses the same positive-unlabeled treatment as the main
    relation objective.  Positive pairs matching configured confusion groups
    (for example school/garden or administrative-area/water-system) receive
    extra weight, while unlabeled pairs remain weak evidence rather than hard
    negatives.
    """
    if logits.shape != targets.shape or logits.size(0) != head_type_probs.size(0):
        raise ValueError("pair-type compatibility tensors have incompatible shapes")
    if head_type_probs.shape != tail_type_probs.shape:
        raise ValueError("head and tail type probabilities must have the same shape")
    if focus_matrix.shape != (head_type_probs.size(1), tail_type_probs.size(1)):
        raise ValueError("focus_matrix shape does not match entity type probabilities")
    if logits.numel() == 0:
        return logits.sum() * 0.0

    row_focus = torch.einsum(
        "pa,ab,pb->p", head_type_probs, focus_matrix.to(logits), tail_type_probs
    ).clamp_min(1.0)
    positive = targets.to(logits.dtype)
    priors = class_priors.to(logits.device, logits.dtype).clamp(1e-5, 0.5)
    positive_counts = positive.sum(dim=0)
    positive_loss = (
        F.softplus(-logits) * positive * row_focus.unsqueeze(-1)
    ).sum(dim=0) / (positive * row_focus.unsqueeze(-1)).sum(dim=0).clamp_min(1.0)
    # Keep the unlabeled component deliberately small: relation annotations
    # are incomplete in this corpus and should not turn compatibility into a
    # closed-world classifier.
    unlabeled = 1.0 - positive
    unlabeled_loss = (F.softplus(logits) * unlabeled).sum(dim=0) / unlabeled.sum(
        dim=0
    ).clamp_min(1.0)
    active = positive_counts > 0
    if not active.any():
        return logits.sum() * 0.0
    return (
        (positive_loss[active] * priors[active])
        + 0.1 * unlabeled_loss[active]
    ).mean()


def long_distance_pair_ranking_loss(
    logits: Tensor,
    targets: Tensor,
    entity_positions: Tensor,
    pairs: Tensor,
    boundary: int = 128,
    margin: float = 0.2,
    max_negatives: int = 64,
) -> Tensor:
    """Rank long-distance positive pairs above unlabeled pairs per relation."""
    if logits.shape != targets.shape or logits.size(0) != pairs.size(0):
        raise ValueError("long-distance ranking tensors have incompatible shapes")
    if boundary < 0 or margin <= 0 or max_negatives < 1:
        raise ValueError("invalid long-distance ranking parameters")
    if logits.numel() == 0:
        return logits.sum() * 0.0

    head_idx, tail_idx = pairs[:, 0], pairs[:, 1]
    distances = (entity_positions[head_idx] - entity_positions[tail_idx]).abs()
    long_mask = distances >= boundary
    if not long_mask.any():
        return logits.sum() * 0.0

    losses = []
    for relation_id in range(logits.size(1)):
        positive = (targets[:, relation_id] > 0) & long_mask
        negative = (targets[:, relation_id] <= 0) & long_mask
        positive_ids = positive.nonzero(as_tuple=False).flatten()
        negative_ids = negative.nonzero(as_tuple=False).flatten()
        if not positive_ids.numel() or not negative_ids.numel():
            continue
        if positive_ids.numel() > max_negatives:
            positive_ids = positive_ids[:max_negatives]
        if negative_ids.numel() > max_negatives:
            negative_ids = negative_ids[:max_negatives]
        positive_scores = logits[positive_ids, relation_id]
        negative_scores = logits[negative_ids, relation_id]
        losses.append(
            F.relu(
                margin - positive_scores.unsqueeze(1) + negative_scores.unsqueeze(0)
            ).mean()
        )
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def distance_weighted_positive_loss(
    logits: Tensor,
    targets: Tensor,
    entity_positions: Tensor,
    pairs: Tensor,
    boundaries: list[int] | Tensor = (32, 64, 128),
    weights: list[float] | Tensor = (1.0, 1.25, 1.5, 2.0),
) -> Tensor:
    """Weight positive relation supervision more heavily at long distances.

    This is an auxiliary term for the learned-fusion experiment. It does not
    turn PU negatives into labelled negatives: only positive target entries
    contribute, while the existing non-negative PU risk remains the primary
    objective.
    """
    if logits.shape != targets.shape or logits.size(0) != pairs.size(0):
        raise ValueError("logits, targets, and pairs have incompatible shapes")
    if logits.numel() == 0:
        return logits.sum() * 0.0
    if len(weights) != len(boundaries) + 1:
        raise ValueError("distance weights must have one more value than boundaries")
    boundary_tensor = torch.as_tensor(boundaries, device=logits.device, dtype=torch.float)
    weight_tensor = torch.as_tensor(weights, device=logits.device, dtype=logits.dtype)
    if boundary_tensor.numel() and not torch.all(boundary_tensor[1:] > boundary_tensor[:-1]):
        raise ValueError("distance boundaries must be strictly increasing")
    if not torch.all(weight_tensor > 0):
        raise ValueError("distance weights must be positive")

    head_idx, tail_idx = pairs[:, 0], pairs[:, 1]
    distances = (entity_positions[head_idx] - entity_positions[tail_idx]).abs()
    row_weights = weight_tensor[torch.bucketize(distances, boundary_tensor)]
    positive = targets.to(logits.dtype)
    weighted_positive = F.softplus(-logits) * positive * row_weights.unsqueeze(-1)
    denominator = (positive * row_weights.unsqueeze(-1)).sum()
    if denominator.item() == 0:
        return logits.sum() * 0.0
    return weighted_positive.sum() / denominator.clamp_min(1e-12)


def relation_balanced_positive_loss(
    logits: Tensor,
    targets: Tensor,
    class_priors: Tensor,
    max_weight: float = 4.0,
) -> Tensor:
    """Auxiliary positive loss that prevents rare relation labels vanishing.

    The primary PU objective uses the estimated class prior by design. For
    the type-balanced ablation, positive examples receive a capped inverse
    square-root prior weight so rare relation types still contribute without
    letting a handful of labels dominate training.
    """
    if logits.shape != targets.shape or logits.size(1) != class_priors.numel():
        raise ValueError("logits, targets, and class_priors have incompatible shapes")
    if max_weight < 1.0:
        raise ValueError("max_weight must be at least 1")
    if logits.numel() == 0:
        return logits.sum() * 0.0
    priors = class_priors.to(device=logits.device, dtype=logits.dtype).clamp_min(1e-5)
    reference = priors.median()
    relation_weights = (reference / priors).sqrt().clamp(0.5, max_weight)
    positive = targets.to(logits.dtype)
    positive_counts = positive.sum(dim=0)
    per_relation = (F.softplus(-logits) * positive).sum(dim=0) / positive_counts.clamp_min(1.0)
    active = positive_counts > 0
    if not active.any():
        return logits.sum() * 0.0
    return (per_relation[active] * relation_weights[active]).sum() / relation_weights[active].sum()


def silver_evidence_distribution(
    pairs: Tensor,
    entity_positions: Tensor,
    window_ranges: Tensor,
) -> Tensor:
    """Create a high-recall soft target over evidence windows for entity pairs."""
    if pairs.numel() == 0:
        return entity_positions.new_empty((0, window_ranges.size(0)))
    head_pos = entity_positions[pairs[:, 0]].unsqueeze(1)
    tail_pos = entity_positions[pairs[:, 1]].unsqueeze(1)
    starts = window_ranges[:, 0].unsqueeze(0)
    ends = window_ranges[:, 1].unsqueeze(0)
    centers = (starts + ends) / 2
    covers_head = (starts <= head_pos) & (head_pos < ends)
    covers_tail = (starts <= tail_pos) & (tail_pos < ends)
    lower = torch.minimum(head_pos, tail_pos)
    upper = torch.maximum(head_pos, tail_pos)
    between = (lower <= centers) & (centers <= upper)
    weights = covers_head.float() + covers_tail.float() + 0.25 * between.float()
    no_evidence = weights.sum(dim=-1).eq(0)
    if no_evidence.any():
        midpoint = (head_pos + tail_pos) / 2
        nearest = (centers - midpoint).abs().argmin(dim=-1)
        weights[no_evidence, nearest[no_evidence]] = 1.0
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def evidence_kl_loss(attention: Tensor, target: Tensor, positive_pairs: Tensor) -> Tensor:
    if not positive_pairs.any():
        return attention.sum() * 0.0
    selected_attention = attention[positive_pairs].clamp_min(1e-12)
    selected_target = target[positive_pairs]
    return F.kl_div(selected_attention.log(), selected_target, reduction="batchmean")


def _entity_type_probabilities(
    entities: list[dict[str, Any]], entity_types: list[str], device: torch.device
) -> Tensor:
    type_to_id = {name: idx for idx, name in enumerate(entity_types)}
    probabilities = torch.zeros((len(entities), len(entity_types)), device=device)
    for entity_id, entity in enumerate(entities):
        distribution = entity.get("type_probabilities")
        if isinstance(distribution, dict):
            for name, value in distribution.items():
                if name in type_to_id:
                    probabilities[entity_id, type_to_id[name]] = float(value)
        if probabilities[entity_id].sum() <= 0:
            probabilities[entity_id, type_to_id[entity["type"]]] = 1.0
        probabilities[entity_id] /= probabilities[entity_id].sum().clamp_min(1e-12)
    return probabilities


@torch.no_grad()
def encode_document_features(
    local_model: nn.Module,
    raw_document: dict[str, Any],
    entities: list[dict[str, Any]],
    entity_types: list[str],
    max_length: int,
    overlap_words: int,
    device: str | torch.device,
    window_batch_size: int = 8,
) -> dict[str, Tensor]:
    """Encode label-independent windows and pool all views of each entity."""
    from iter.datasets.windowing import build_document_windows

    device = torch.device(device)
    item = {"tokens": raw_document["tokens"], "entities": entities, "relations": []}
    windows, manifest, coverage = build_document_windows(
        item, local_model.tokenizer, max_length=max_length, overlap_words=overlap_words
    )
    if coverage["uncovered_entity_ids"]:
        raise ValueError(
            "Entities are not fully visible in any window: "
            f"{coverage['uncovered_entity_ids']}. Increase overlap_words."
        )

    window_vectors: list[Tensor] = []
    entity_views: dict[int, list[tuple[Tensor, float]]] = {idx: [] for idx in range(len(entities))}
    for batch_start in range(0, len(windows), window_batch_size):
        batch_windows = windows[batch_start:batch_start + window_batch_size]
        batch_manifest = manifest[batch_start:batch_start + window_batch_size]
        batch_tokens = [window["tokens"] for window in batch_windows]
        encoding = local_model.tokenizer(
            batch_tokens,
            is_split_into_words=True,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        hidden = local_model.model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )[0]
        if getattr(local_model, "multiscale_module", None) is not None:
            hidden = local_model.multiscale_module(hidden, attention_mask)

        for local_batch_id, (window, record) in enumerate(zip(batch_windows, batch_manifest)):
            word_ids = encoding.word_ids(batch_index=local_batch_id)
            content_mask = torch.tensor(
                [word_id is not None for word_id in word_ids], device=device, dtype=torch.bool
            )
            vector = hidden[local_batch_id, content_mask].mean(dim=0)
            window_vectors.append(vector)
            for local_entity, source_entity_id in zip(
                window["entities"], record["source_entity_ids"]
            ):
                token_mask = torch.tensor(
                    [
                        word_id is not None
                        and local_entity["start"] <= word_id < local_entity["end"]
                        for word_id in word_ids
                    ],
                    device=device,
                    dtype=torch.bool,
                )
                if not token_mask.any():
                    continue
                entity_vector = hidden[local_batch_id, token_mask].mean(dim=0)
                global_entity = entities[source_entity_id]
                centrality = min(
                    global_entity["start"] - record["word_start"],
                    record["word_end"] - global_entity["end"],
                )
                entity_views[source_entity_id].append((entity_vector, float(centrality)))

    pooled_entities = []
    for entity_id in range(len(entities)):
        views = entity_views[entity_id]
        if not views:
            raise RuntimeError(f"Entity {entity_id} has no encoded window view")
        vectors = torch.stack([view[0] for view in views])
        centrality = torch.tensor([view[1] for view in views], device=device)
        weights = (centrality / 16.0).softmax(dim=0)
        pooled_entities.append((weights.unsqueeze(1) * vectors).sum(dim=0))

    hidden_size = int(local_model.config.d_model)
    entity_repr = (
        torch.stack(pooled_entities)
        if pooled_entities
        else torch.empty((0, hidden_size), device=device)
    )
    window_repr = (
        torch.stack(window_vectors)
        if window_vectors
        else torch.empty((0, hidden_size), device=device)
    )
    positions = torch.tensor(
        [(entity["start"] + entity["end"] - 1) / 2 for entity in entities],
        device=device,
        dtype=torch.float,
    )
    window_ranges = torch.tensor(
        [[record["word_start"], record["word_end"]] for record in manifest],
        device=device,
        dtype=torch.float,
    )
    return {
        "entity_repr": entity_repr,
        "entity_type_probs": _entity_type_probabilities(entities, entity_types, device),
        "entity_positions": positions,
        "window_repr": window_repr,
        "window_ranges": window_ranges,
    }


def compute_relation_statistics(
    documents: Iterable[dict[str, Any]],
    entity_types: list[str],
    relation_types: list[str],
    symmetric_relation_ids: set[int],
) -> tuple[Tensor, Tensor]:
    """Estimate class priors and smoothed type compatibility log-odds."""
    type_to_id = {name: idx for idx, name in enumerate(entity_types)}
    relation_to_id = {name: idx for idx, name in enumerate(relation_types)}
    type_pair_count = torch.zeros((len(entity_types), len(entity_types)))
    positive_count = torch.zeros((len(entity_types), len(entity_types), len(relation_types)))
    total_pairs = 0
    total_positive = torch.zeros(len(relation_types))

    for document in documents:
        entities = document.get("entities", [])
        for head_id, head in enumerate(entities):
            for tail_id, tail in enumerate(entities):
                if head_id == tail_id:
                    continue
                type_pair_count[type_to_id[head["type"]], type_to_id[tail["type"]]] += 1
                total_pairs += 1
        for relation in document.get("relations", []):
            relation_id = relation_to_id[relation["type"]]
            head = entities[relation["head"]]
            tail = entities[relation["tail"]]
            head_type = type_to_id[head["type"]]
            tail_type = type_to_id[tail["type"]]
            positive_count[head_type, tail_type, relation_id] += 1
            total_positive[relation_id] += 1
            if relation_id in symmetric_relation_ids:
                positive_count[tail_type, head_type, relation_id] += 1
                total_positive[relation_id] += 1

    class_priors = (total_positive + 1.0) / (max(total_pairs, 1) + 2.0)
    pair_denominator = type_pair_count.unsqueeze(-1) + 2.0
    conditional = (positive_count + 1.0) / pair_denominator
    conditional = conditional.clamp(1e-5, 1 - 1e-5)
    global_prior = class_priors.view(1, 1, -1).clamp(1e-5, 1 - 1e-5)
    compatibility = torch.logit(conditional) - torch.logit(global_prior)
    compatibility[type_pair_count.eq(0).unsqueeze(-1).expand_as(compatibility)] = 0.0
    return compatibility.clamp(-4.0, 4.0), class_priors.clamp(1e-5, 0.5)
