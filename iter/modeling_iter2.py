import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcheck
from torch import Tensor
from torch.nn import Dropout, Linear, Module
from torchcheck import batched_index_padded, assert_true
from torchcheck.ops import expand_dim, inclusive_cumsum, unsqueeze_dims, unsqueeze_and_expand_dims
from transformers import PreTrainedModel, PreTrainedTokenizerBase, add_start_docstrings
from transformers.activations import ACT2FN

from iter import ITERConfig
from iter.generation_utils import decode_actions_and_pairings
from iter.modeling_features import FeaturesMixin
from iter.modules.fact_decomposition import FactDecompositionModule
from iter.modules.multiscale_geo import MultiScaleGeoModule
from iter.modules.long_context_ner import LongContextNERRefiner
from iter.utils import NamedTuple2

VERIFY_GENERATION = os.environ.get("ITER_VERIFY_GENERATION", "false") == "true"
NEG_INF = -20000

class FFN(Module):
    gate: Linear
    wi: Linear
    wo: Linear
    act: Module

    def __init__(self, config: ITERConfig, in_channels: int = 1, out_channels: int | str = 1):
        super().__init__()
        self.config = config
        self.dropout = Dropout(config.dropout)
        inner_scale = (config.use_scale + 1) if in_channels > 1 else 1
        in_channels = in_channels // (1 + config.is_feature_sum_representations)
        out_channels = out_channels if isinstance(out_channels, int) else getattr(config, out_channels)
        self.gate = Linear(
            in_channels * config.d_model,
            inner_scale * config.d_ff,
            False
        ) if config.use_gate else None
        self.wi = Linear(in_channels * config.d_model, inner_scale * config.d_ff, False)
        self.wo = Linear(inner_scale * config.d_ff, out_channels, False)
        self.act = ACT2FN[config.activation_fn]

    def resize(self, size: int):
        d_out, d_in = self.wo.weight.size()
        if size == d_out:
            return
        wo = self.wo
        self.wo = Linear(d_in, size, bias=False, device=wo.weight.device, dtype=wo.weight.dtype)
        copy = min(size, wo.weight.size(0))
        if copy > 0:
            self.wo.weight[:copy, :] = wo.weight[:copy, :]

    def forward(self, x: Tensor) -> Tensor:
        if self.gate is None:
            x = self.wi(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.wo(x)
        else:
            gate = self.gate(x)
            gate = self.act(gate)
            x = self.wi(x)
            x = gate * x
            x = self.dropout(x)
            x = self.wo(x)
        return x


class IsLeft(FFN):
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Representation the input of shape [B, N, H]
                        there B is the batch dimension, N the sequence length and H the hidden dimension size

        Returns: A Tensor of shape [B, N, 1], where each logit `> 0` indicates the beginning of a span
        """
        return super().forward(x)


class IsSpan(FFN):
    def __init__(self, config: ITERConfig):
        super().__init__(config, in_channels=2, out_channels="num_types")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Representation the input of shape [B, N, H]
                        where B is the batch dimension, N the sequence length and H the hidden dimension size

        Returns: A Tensor of shape [B, N, num_types], where each logit `> 0` indicates the end of a span.
        The type of the span is inferred from the argmax:

        >>> types = self.lr_score(x).argmax(dim=-1)
        """
        return super().forward(x)


class IsLink(FFN):
    def __init__(self, config: ITERConfig):
        num_links = config.num_links * (1 + config.is_feature_negsample_link_types)
        super().__init__(config, in_channels=2, out_channels=num_links)


@dataclass
class ITERBaseOutput(NamedTuple2):
    loss: Tensor
    hidden_state: Tensor
    l_loss: Tensor
    lr_loss: Tensor

@dataclass
class ITEROutput(ITERBaseOutput):
    rr_loss: Tensor
    fact_loss: Optional[Tensor] = None
    multiscale_loss: Optional[Tensor] = None

@dataclass
class ITERGenerateBaseCache(NamedTuple2):
    closest_left_indices: Tensor
    closest_left_mask: Tensor
    closest_left_hidden: Tensor

@dataclass
class ITERGenerateBaseOutput(NamedTuple2):
    actions: Tensor
    lr_pair_flag: Tensor
    cache: Optional[ITERGenerateBaseCache]
    # Posterior scores used by document-level flat-NER decoding.
    lr_pair_probabilities: Optional[Tensor]
    lr_span_probabilities: Optional[Tensor]

@dataclass
class ITERGenerateOutput(ITERGenerateBaseOutput):
    rr_pair_flag: Tensor
    rr_pair_probabilities: Tensor

@dataclass
class ITERGenerationOutput(NamedTuple2):
    actions: Tensor
    lr_pair_flag: Tensor
    rr_pair_flag: Optional[Tensor]
    rr_pair_probabilities: Optional[Tensor]
    lr_pair_probabilities: Optional[Tensor]
    lr_span_probabilities: Optional[Tensor]
    entities: Optional[list]
    links: Optional[list]


def mask_as_one_hot(mask: Tensor) -> Tensor:
    torchcheck.assert_dtype(mask, dtype=torch.bool)
    return F.one_hot(mask.long(), num_classes=2)


ITER_ACTION_LEFT_BRACKET_BIT = 0
ITER_ACTION_RIGHT_BRACKET_BIT = 1
ITER_ACTION_PAD_BIT = 2


def is_action(actions: Tensor, bit: int, as_one_hot: bool = False, dtype: Optional[torch.dtype] = None) -> Tensor:
    x = actions.bitwise_and(1 << bit).ne(0)
    if as_one_hot:
        x = mask_as_one_hot(x)
    if dtype is not None:
        x = x.to(dtype)
    return x


class ITER(PreTrainedModel, FeaturesMixin):
    config_class = ITERConfig
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    num_types: int
    num_links: int
    max_nest_depth: int

    def __init__(
        self,
        config: ITERConfig,
    ):
        super().__init__(config)
        model_cls = config.guess_model_class()
        model_kwargs = config.model_kwargs
        # Try loading the pretrained backbone from HF / cache first.
        # If offline / not cached, fall back to initialising from the config
        # alone – the real weights will be loaded from the ITER checkpoint
        # state dict by PreTrainedModel.from_pretrained().
        try:
            self.model = model_cls.from_pretrained(
                config.transformer_config.name_or_path, **model_kwargs,
            )
        except (OSError, EnvironmentError):
            self.model = model_cls(config.transformer_config, **model_kwargs)

        self.config = config
        import os as _os
        _offline = _os.environ.get("TRANSFORMERS_OFFLINE", "") == "1"
        try:
            self.tokenizer = config.guess_tokenizer_class(True).from_pretrained(
                config.transformer_config.name_or_path,
                config=config.transformer_config,  # 传入已加载的config，跳过AutoConfig下载
                model_max_length=self.config.max_length,
                local_files_only=_offline,
                **config.tokenizer_kwargs(use_fast=True)
            )
        except (OSError, EnvironmentError):
            raise RuntimeError(
                f"Unable to load tokenizer for '{config.transformer_config.name_or_path}'."
                f"Please ensure the tokenizer files are available in the HuggingFace cache"
                f"or download them first with: "
                f"python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("
                f"'{config.transformer_config.name_or_path}')\""
            )
        self.num_types = config.num_types
        self.max_nest_depth = config.max_nest_depth
        self.features = config.features

        self.is_l = IsLeft(self.config)
        self.lr_score = IsSpan(self.config)
        # Optional direct boundary supervision; decoder fusion is controlled by
        # ner_boundary_decode_weight and ner_type_decode_weight.
        self.ner_boundary_start = Linear(config.d_model, 1)
        self.ner_boundary_end = Linear(config.d_model, 1)
        self.ner_type_classifier = nn.Sequential(
            Linear(2 * config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            Dropout(config.dropout),
            Linear(config.d_model, config.num_types),
        )
        # A low-rank biaffine residual gives the span decoder an explicit
        # start/end interaction and a width prior.  The legacy FFN remains the
        # main scorer; this residual is enabled only by the new v5 config.
        self.ner_span_biaffine = None
        self.ner_span_gate = None
        if bool(getattr(config, "ner_span_biaffine", False)):
            rank = max(4, int(getattr(config, "ner_span_biaffine_rank", 64)))
            self.ner_span_start_projection = Linear(config.d_model, rank, bias=False)
            self.ner_span_end_projection = Linear(config.d_model, rank, bias=False)
            self.ner_span_biaffine = nn.Parameter(
                torch.zeros(config.num_types, rank, rank)
            )
            max_width = max(4, int(getattr(config, "max_span_length", 30)))
            self.ner_span_width = (
                nn.Embedding(max_width + 2, config.num_types)
                if bool(getattr(config, "ner_span_width_embedding", True))
                else None
            )
            if self.ner_span_width is not None:
                nn.init.zeros_(self.ner_span_width.weight)
            if bool(getattr(config, "ner_span_learned_gate", False)):
                self.ner_span_gate = nn.Sequential(
                    Linear(2, 8),
                    nn.GELU(),
                    Linear(8, 1),
                )
                initial_weight = min(
                    max(float(getattr(config, "ner_span_biaffine_weight", 0.35)), 1e-4),
                    1.0 - 1e-4,
                )
                nn.init.zeros_(self.ner_span_gate[-1].weight)
                nn.init.constant_(
                    self.ner_span_gate[-1].bias,
                    math.log(initial_weight / (1.0 - initial_weight)),
                )

        self.ner_type_prototypes = None
        if (
            float(getattr(config, "ner_type_prototype_loss_weight", 0.0)) > 0.0
            or float(getattr(config, "ner_type_prototype_decode_weight", 0.0)) > 0.0
        ):
            self.ner_type_prototypes = nn.Parameter(
                torch.empty(config.num_types, config.d_model)
            )
            nn.init.normal_(self.ner_type_prototypes, std=0.02)

        self.ner_target_adapter = None
        self.ner_target_gate = None
        if bool(getattr(config, "ner_target_adapter", False)):
            adapter_size = max(16, int(getattr(config, "ner_target_adapter_size", 128)))
            self.ner_target_adapter = nn.ModuleDict({
                "norm": nn.LayerNorm(config.d_model),
                "down": Linear(config.d_model, adapter_size),
                "local": nn.Conv1d(
                    adapter_size,
                    adapter_size,
                    kernel_size=3,
                    padding=1,
                    groups=adapter_size,
                ),
                "dropout": Dropout(
                    float(getattr(config, "ner_target_adapter_dropout", 0.10))
                ),
                "up": Linear(adapter_size, config.d_model),
            })
            nn.init.zeros_(self.ner_target_adapter["up"].weight)
            nn.init.zeros_(self.ner_target_adapter["up"].bias)
            self.ner_target_gate = nn.Parameter(torch.tensor(0.0))

        # Initialize new modules as None (subclass may override)
        self.fact_module = None
        self.multiscale_module = None
        self.num_iterations = self.config.num_iterations if hasattr(self.config, 'num_iterations') else 3
        self.fact_loss_weight = self.config.fact_loss_weight if hasattr(self.config, 'fact_loss_weight') else 1.0
        self.multiscale_loss_weight = self.config.multiscale_loss_weight if hasattr(self.config, 'multiscale_loss_weight') else 0.1
        self.long_context_refiner = None
        if config.is_feature_long_context_ner_refiner:
            self.long_context_refiner = LongContextNERRefiner(
                hidden_dim=config.d_model,
                chunk_size=config.long_context_chunk_size,
                num_heads=config.long_context_num_heads,
                dropout=config.long_context_dropout,
            )

    def _apply_long_context_refiner(self, hidden_states: Tensor, attention_mask: Tensor | None) -> Tensor:
        if self.long_context_refiner is None:
            return hidden_states
        return self.long_context_refiner(hidden_states, attention_mask)

    def _apply_ner_target_adapter(
        self, hidden_states: Tensor, attention_mask: Tensor | None
    ) -> Tensor:
        """Apply a zero-initialized local residual adapter for target adaptation."""
        if self.ner_target_adapter is None:
            return hidden_states
        residual = self.ner_target_adapter["norm"](hidden_states)
        residual = F.gelu(self.ner_target_adapter["down"](residual))
        residual = self.ner_target_adapter["local"](
            residual.transpose(1, 2)
        ).transpose(1, 2)
        residual = F.gelu(residual)
        residual = self.ner_target_adapter["dropout"](residual)
        residual = self.ner_target_adapter["up"](residual)
        if attention_mask is not None:
            residual = residual * attention_mask.unsqueeze(-1).to(residual.dtype)
        scale = float(getattr(self.config, "ner_target_adapter_scale", 0.30))
        return hidden_states + scale * torch.sigmoid(self.ner_target_gate) * residual

    def _ner_type_auxiliary_loss(
        self,
        hidden_states: Tensor,
        actions: Tensor,
        lr_pair_flag: Tensor,
    ) -> Tensor:
        """Train an explicit typed-span classifier for long-context NER.

        ITER's marginal span loss decides whether any type is plausible.  The
        auxiliary objective isolates the type decision at gold entity spans,
        allowing optional class weights to address the geographic-type
        confusions without changing the legacy loss when its weight is zero.
        """
        loss_weight = float(getattr(self.config, "ner_type_loss_weight", 0.0))
        prototype_weight = float(
            getattr(self.config, "ner_type_prototype_loss_weight", 0.0)
        )
        if (loss_weight <= 0.0 and prototype_weight <= 0.0) or lr_pair_flag is None:
            return hidden_states.new_zeros(())

        num_real_types = int(self.config.num_types - int(self.is_feature_extra_lr_class))
        if num_real_types <= 0:
            return hidden_states.new_zeros(())
        target_flags = lr_pair_flag[..., :num_real_types]
        positive = target_flags.nonzero(as_tuple=False)
        if positive.numel() == 0:
            return hidden_states.new_zeros(())

        is_left = actions.bitwise_and(1).ne(0)
        left_positions, left_mask = batched_index_padded(is_left, 0, return_mask=True)
        batch_indices, right_positions, left_indices, target_types = positive.unbind(dim=1)
        valid = (left_indices < left_positions.size(1))
        valid &= left_mask[batch_indices, left_indices]
        if not valid.any():
            return hidden_states.new_zeros(())
        batch_indices = batch_indices[valid]
        right_positions = right_positions[valid]
        left_indices = left_indices[valid]
        target_types = target_types[valid]
        left_token_positions = left_positions[batch_indices, left_indices]
        pair_hidden = self._merge_representations(
            hidden_states[batch_indices, left_token_positions],
            hidden_states[batch_indices, right_positions],
        )
        if hasattr(self, "ner_type_classifier"):
            logits = self.ner_type_classifier(pair_hidden)[..., :num_real_types]
        else:
            logits = self.lr_score(pair_hidden)[..., :num_real_types]

        class_weights = getattr(self.config, "ner_type_class_weights", None)
        weight_tensor = None
        if class_weights is not None:
            if len(class_weights) != num_real_types:
                raise ValueError(
                    "ner_type_class_weights must contain one value per real entity type "
                    f"({len(class_weights)} != {num_real_types})"
                )
            weight_tensor = torch.as_tensor(
                class_weights, dtype=logits.dtype, device=logits.device
            )
            weight_tensor = weight_tensor / weight_tensor.mean().clamp_min(1e-6)
        per_example_weights = torch.ones_like(target_types, dtype=logits.dtype)
        focus_types = set(getattr(self.config, "ner_type_focus_types", []) or [])
        if focus_types and self.config.entity_types:
            focus_ids = {
                type_id
                for type_id, type_name in enumerate(self.config.entity_types)
                if type_name in focus_types
            }
            if focus_ids:
                focus_mask = torch.zeros_like(target_types, dtype=torch.bool)
                for type_id in focus_ids:
                    focus_mask |= target_types.eq(type_id)
                focus_weight = float(getattr(self.config, "ner_type_focus_weight", 1.0))
                per_example_weights = torch.where(
                    focus_mask,
                    per_example_weights * focus_weight,
                    per_example_weights,
                )

        total_loss = logits.sum() * 0.0
        if loss_weight > 0.0:
            per_example_loss = F.cross_entropy(
                logits,
                target_types,
                weight=weight_tensor,
                label_smoothing=float(
                    getattr(self.config, "ner_type_label_smoothing", 0.0)
                ),
                reduction="none",
            )
            focal_gamma = float(getattr(self.config, "ner_type_focal_gamma", 0.0))
            if focal_gamma > 0.0:
                target_probability = logits.softmax(dim=-1).gather(
                    1, target_types.unsqueeze(1)
                ).squeeze(1)
                per_example_loss = per_example_loss * (1.0 - target_probability).pow(
                    focal_gamma
                )
            total_loss = total_loss + loss_weight * (
                per_example_loss * per_example_weights
            ).mean()

        if prototype_weight > 0.0:
            if self.ner_type_prototypes is None:
                raise RuntimeError("NER type prototypes are not initialized")
            prototype_logits = self._score_ner_type_prototypes(pair_hidden)[
                ..., :num_real_types
            ]
            prototype_ce = F.cross_entropy(
                prototype_logits, target_types, reduction="none"
            )
            prototype_loss = (prototype_ce * per_example_weights).mean()

            confusion_terms = []
            type_to_id = {
                name: type_id
                for type_id, name in enumerate(self.config.entity_types[:num_real_types])
            }
            margin = float(getattr(self.config, "ner_type_confusion_margin", 0.20))
            for pair in getattr(self.config, "ner_type_confusion_pairs", []) or []:
                if len(pair) != 2 or pair[0] not in type_to_id or pair[1] not in type_to_id:
                    continue
                first, second = type_to_id[pair[0]], type_to_id[pair[1]]
                for target_id, confused_id in ((first, second), (second, first)):
                    selected = target_types.eq(target_id)
                    if selected.any():
                        target_score = prototype_logits[selected, target_id]
                        confused_score = prototype_logits[selected, confused_id]
                        confusion_terms.append(
                            F.relu(margin - target_score + confused_score).mean()
                        )
            if confusion_terms:
                prototype_loss = prototype_loss + torch.stack(confusion_terms).mean()
            total_loss = total_loss + prototype_weight * prototype_loss
        return total_loss

    def _score_ner_type_prototypes(self, pair_hidden: Tensor) -> Tensor:
        """Cosine scores against learned entity-type prototypes."""
        if self.ner_type_prototypes is None:
            raise RuntimeError("NER type prototypes are disabled")
        projected = self.ner_type_classifier[0](pair_hidden)
        projected = self.ner_type_classifier[1](projected)
        projected = self.ner_type_classifier[2](projected)
        projected = F.normalize(projected, dim=-1)
        prototypes = F.normalize(self.ner_type_prototypes, dim=-1)
        temperature = max(
            float(getattr(self.config, "ner_type_prototype_temperature", 0.15)),
            1e-4,
        )
        return projected @ prototypes.T / temperature

    def _ner_boundary_auxiliary_loss(
        self,
        hidden_states: Tensor,
        actions: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Direct start/end supervision with boundary hard negatives."""
        loss_weight = float(getattr(self.config, "ner_boundary_loss_weight", 0.0))
        if loss_weight <= 0.0 or actions is None:
            return hidden_states.new_zeros(())
        valid = attention_mask.bool()
        if not valid.any():
            return hidden_states.new_zeros(())
        starts = actions.bitwise_and(1 << ITER_ACTION_LEFT_BRACKET_BIT).ne(0) & valid
        ends = actions.bitwise_and(1 << ITER_ACTION_RIGHT_BRACKET_BIT).ne(0) & valid
        radius = max(0, int(getattr(self.config, "ner_boundary_hard_negative_radius", 1)))
        boundaries = starts | ends
        if radius:
            padded = F.pad(boundaries.to(hidden_states.dtype).unsqueeze(1), (radius, radius))
            near_boundary = F.max_pool1d(
                padded, kernel_size=2 * radius + 1, stride=1
            ).squeeze(1).bool()
        else:
            near_boundary = boundaries
        # Flat bracket targets: cumulative opens identify non-boundary tokens
        # inside an entity, which are hard negatives for both heads.
        open_entities = starts.long().cumsum(dim=1) > ends.long().cumsum(dim=1)
        internal = open_entities & ~boundaries & valid
        adjacent_negative = near_boundary & ~boundaries & valid
        internal_negative = internal & ~adjacent_negative
        start_logits = self.ner_boundary_start(hidden_states).squeeze(-1)
        end_logits = self.ner_boundary_end(hidden_states).squeeze(-1)
        positive_weight = float(getattr(self.config, "ner_boundary_positive_weight", 1.0))
        hard_weight = float(getattr(self.config, "ner_boundary_hard_negative_weight", 2.0))
        adjacent_weight = getattr(self.config, "ner_boundary_adjacent_negative_weight", None)
        internal_weight = getattr(self.config, "ner_boundary_internal_negative_weight", None)
        adjacent_weight = hard_weight if adjacent_weight is None else float(adjacent_weight)
        internal_weight = hard_weight if internal_weight is None else float(internal_weight)
        focal_gamma = float(getattr(self.config, "ner_boundary_focal_gamma", 0.0))
        consistency_weight = max(
            float(getattr(self.config, "ner_boundary_consistency_loss_weight", 0.0)),
            0.0,
        )

        def weighted_bce(logits: Tensor, labels: Tensor) -> Tensor:
            weights = torch.where(labels, torch.full_like(logits, positive_weight), torch.ones_like(logits))
            weights = torch.where(adjacent_negative, weights * adjacent_weight, weights)
            weights = torch.where(internal_negative, weights * internal_weight, weights)
            element_loss = F.binary_cross_entropy_with_logits(
                logits, labels.to(logits.dtype), reduction="none"
            )
            if focal_gamma > 0.0:
                probabilities = logits.sigmoid()
                target_probability = torch.where(labels, probabilities, 1.0 - probabilities)
                element_loss = element_loss * (1.0 - target_probability).pow(focal_gamma)
            mask = valid.to(element_loss.dtype)
            return (element_loss * weights * mask).sum() / mask.sum().clamp_min(1.0)

        start_loss = weighted_bce(start_logits, starts)
        end_loss = weighted_bce(end_logits, ends)
        total = loss_weight * 0.5 * (start_loss + end_loss)
        if consistency_weight:
            # A token is a boundary when either head fires.  Supervising this
            # union reduces the tendency of independent heads to suppress a
            # valid start or end in long, dense documents.
            union_logits = torch.logaddexp(start_logits, end_logits)
            total = total + consistency_weight * weighted_bce(union_logits, boundaries)
        return total

    def _ner_span_margin_loss(
        self,
        hidden_states: Tensor,
        actions: Tensor,
        lr_pair_flag: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Contrast gold typed spans against locally corrupted spans.

        The legacy marginal span objective treats many boundary/type
        alternatives as equivalent.  This auxiliary ranking objective gives
        the exact gold (start, end, type) a margin over nearby end/start
        shifts and wrong-type alternatives, reducing strict-span false
        positives without changing decoding at inference time.
        """
        weight = float(getattr(self.config, "ner_span_margin_loss_weight", 0.0))
        if weight <= 0.0 or actions is None or lr_pair_flag is None:
            return hidden_states.new_zeros(())
        margin = float(getattr(self.config, "ner_span_margin", 0.20))
        radius = max(1, int(getattr(self.config, "ner_span_margin_hard_negative_radius", 2)))
        max_neg = max(1, int(getattr(self.config, "ner_span_margin_max_negatives", 8)))
        valid = attention_mask.bool()
        starts = actions.bitwise_and(1 << ITER_ACTION_LEFT_BRACKET_BIT).ne(0) & valid
        bs, seq_len, _ = hidden_states.shape
        losses = []
        for b in range(bs):
            left_positions = torch.nonzero(starts[b], as_tuple=False).flatten().tolist()
            if not left_positions:
                continue
            for left_idx, start in enumerate(left_positions):
                gold_pairs = torch.nonzero(lr_pair_flag[b, :, left_idx], as_tuple=False)
                if gold_pairs.numel() == 0:
                    continue
                for end, typ in gold_pairs.tolist():
                    if end < start or end >= seq_len:
                        continue
                    pos = self._score_span_logits(
                        hidden_states[b, start].unsqueeze(0),
                        hidden_states[b, end].unsqueeze(0),
                        hidden_states.new_tensor([end - start]),
                    ).squeeze(0)[int(typ)]
                    negatives = []
                    # Same boundaries, incorrect entity type.
                    all_types = self._score_span_logits(
                        hidden_states[b, start].unsqueeze(0),
                        hidden_states[b, end].unsqueeze(0),
                        hidden_states.new_tensor([end - start]),
                    ).squeeze(0)
                    if self.num_types > 1:
                        wrong = torch.cat((all_types[:typ], all_types[typ + 1:]))
                        negatives.append(wrong.max())
                    # Correct start with shifted end, and correct end with
                    # shifted start: these are the dominant strict-span
                    # errors observed in long documents.
                    for delta in range(1, radius + 1):
                        for bad_end in (end - delta, end + delta):
                            if start <= bad_end < min(seq_len, start + int(getattr(self.config, "max_span_length", 30))):
                                score = self._score_span_logits(
                                    hidden_states[b, start].unsqueeze(0),
                                    hidden_states[b, bad_end].unsqueeze(0),
                                    hidden_states.new_tensor([bad_end - start]),
                                ).squeeze(0).max()
                                negatives.append(score)
                        for bad_start in (start - delta, start + delta):
                            if 0 <= bad_start <= end and end - bad_start <= int(getattr(self.config, "max_span_length", 30)):
                                score = self._score_span_logits(
                                    hidden_states[b, bad_start].unsqueeze(0),
                                    hidden_states[b, end].unsqueeze(0),
                                    hidden_states.new_tensor([end - bad_start]),
                                ).squeeze(0).max()
                                negatives.append(score)
                    if not negatives:
                        continue
                    neg = torch.stack(negatives)
                    if neg.numel() > max_neg:
                        neg = torch.topk(neg, max_neg).values
                    losses.append(F.relu(margin - pos + neg).mean())
        if not losses:
            return hidden_states.new_zeros(())
        return weight * torch.stack(losses).mean()

    def _score_span_logits(
        self,
        left_hidden: Tensor,
        right_hidden: Tensor,
        width: Tensor | None = None,
    ) -> Tensor:
        """Score spans with the legacy head plus an optional biaffine residual."""
        pair_hidden = self._merge_representations(left_hidden, right_hidden)
        logits = self.lr_score(pair_hidden)
        if self.ner_span_biaffine is None:
            return logits
        start = self.ner_span_start_projection(left_hidden)
        end = self.ner_span_end_projection(right_hidden)
        # (..., rank), (types, rank, rank), (..., rank) -> (..., types)
        residual = torch.einsum("...i,tij,...j->...t", start, self.ner_span_biaffine, end)
        if self.ner_span_width is not None and width is not None:
            max_width = self.ner_span_width.num_embeddings - 2
            width_ids = width.clamp_min(0).clamp_max(max_width + 1).long()
            residual = residual + self.ner_span_width(width_ids)
        if self.ner_span_gate is not None:
            gate_features = torch.stack((logits, residual), dim=-1)
            gate = self.ner_span_gate(gate_features).squeeze(-1).sigmoid()
            return logits + gate * residual
        return logits + float(getattr(self.config, "ner_span_biaffine_weight", 0.35)) * residual

    def resize_num_types(self, new_num_types: int):
        self.lr_score.resize(new_num_types)
        if hasattr(self, "ner_type_classifier"):
            output = self.ner_type_classifier[-1]
            if isinstance(output, Linear) and output.out_features != new_num_types:
                replacement = Linear(
                    output.in_features,
                    new_num_types,
                    bias=output.bias is not None,
                    device=output.weight.device,
                    dtype=output.weight.dtype,
                )
                copy = min(new_num_types, output.out_features)
                with torch.no_grad():
                    replacement.weight[:copy].copy_(output.weight[:copy])
                    if output.bias is not None:
                        replacement.bias[:copy].copy_(output.bias[:copy])
                self.ner_type_classifier[-1] = replacement
        if self.ner_type_prototypes is not None and self.ner_type_prototypes.size(0) != new_num_types:
            replacement = self.ner_type_prototypes.new_empty(new_num_types, self.config.d_model)
            nn.init.normal_(replacement, std=0.02)
            copy = min(new_num_types, self.ner_type_prototypes.size(0))
            with torch.no_grad():
                replacement[:copy].copy_(self.ner_type_prototypes[:copy])
            self.ner_type_prototypes = nn.Parameter(replacement)
        self.num_types = new_num_types

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        actions: Tensor,
        lr_pair_flag: Tensor,
    ) -> ITERBaseOutput:
        """
        Args:
            input_ids (Tensor): The tokens the model receives as input of shape [B, N]
            attention_mask (Tensor): The attention mask corresponding to the ``input_ids``
            actions (Tensor): Target actions for each position in a sequence. Shape: [B, N]
                There are two actions the model can perform at a given position: left `[` and right `]`
                bracket pairings. Each action is encoded by a different bit in ``actions`` being set: bit ``0`` for
                left and bit ``1`` for right bracket pairings.
            lr_pair_flag (Tensor): A Boolean tensor of shape [B, N, num_left, num_types], where a `True` value at
                position (b, 1<=n<=N, l, t) indicates a pairing between the l-th bracket and position n of type t.

        Returns: A tuple (loss, hidden state, l_loss, lr_loss)
        """
        # [B, N, H]
        x = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )[0]

        x = self._apply_long_context_refiner(x, attention_mask)
        x = self._apply_ner_target_adapter(x, attention_mask)

        # --- Apply Multi-Scale Geo module during training if available ---
        # This enriches token representations with geographic features
        if self.multiscale_module is not None:
            x = self.multiscale_module(x, attention_mask)

        lr_loss = self._is_span_forward(
            x,
            actions,
            lr_pair_flag
        )
        l_loss = self._is_left_forward(x, actions)
        ner_type_loss = self._ner_type_auxiliary_loss(x, actions, lr_pair_flag)
        ner_boundary_loss = self._ner_boundary_auxiliary_loss(x, actions, attention_mask)
        ner_span_margin_loss = self._ner_span_margin_loss(x, actions, lr_pair_flag, attention_mask)

        loss = l_loss + lr_loss + ner_type_loss + ner_boundary_loss + ner_span_margin_loss
        if self.is_feature_batch_average_loss:
            loss = loss / input_ids.size(0)
        output = ITERBaseOutput(loss, x, l_loss, lr_loss)
        output.ner_type_loss = ner_type_loss
        output.ner_boundary_loss = ner_boundary_loss
        output.ner_span_margin_loss = ner_span_margin_loss
        return output

    def generate(
            self,
            inputs: Optional[Tensor] = None,
            **kwargs,
    ) -> ITERGenerationOutput:
        attention_mask = kwargs.get("attention_mask")
        decode_output = kwargs.get("decode_output", True)

        x = self.model(
            input_ids=inputs,
            attention_mask=attention_mask,
            return_dict=False
        )[0]

        x = self._apply_long_context_refiner(x, attention_mask)
        x = self._apply_ner_target_adapter(x, attention_mask)

        # --- Apply Multi-Scale Geo module during inference if available ---
        # This enriches token representations with geographic features before
        # entity/relation decoding, improving accuracy on place names.
        if self.multiscale_module is not None:
            x = self.multiscale_module(x, attention_mask)

        generation_output = self._generate_forward(
            x,
            attention_mask,
        )

        return self._post_generate_decoding(
            inputs,
            generation_output,
            decode_output
        )

    def _post_generate_decoding(
        self,
        inputs: Tensor,
        generation_output: ITERGenerateBaseOutput,
        decode: Optional[bool] = None,
    ) -> ITERGenerationOutput:
        actions = generation_output.actions
        lr_pair_flag = generation_output.lr_pair_flag
        rr_pair_flag = None
        rr_pair_probabilities = None

        entities, links = None, None
        if hasattr(generation_output, "rr_pair_flag"):
            rr_pair_flag = generation_output.rr_pair_flag
            if hasattr(generation_output, "rr_pair_probabilities"):
                rr_pair_probabilities = generation_output.rr_pair_probabilities
            else:
                rr_pair_probabilities = rr_pair_flag.to(torch.float)

        if decode:
            entities, links = decode_actions_and_pairings(
                self,
                inputs,
                actions,
                lr_pair_flag,
                rr_pair_flag,
                rr_pair_probabilities,
                self.config.entity_types,
                self.config.link_types,
            )

        return ITERGenerationOutput(
            actions=actions,
            lr_pair_flag=lr_pair_flag,
            rr_pair_flag=rr_pair_flag,
            rr_pair_probabilities=rr_pair_probabilities,
            lr_pair_probabilities=generation_output.lr_pair_probabilities,
            lr_span_probabilities=generation_output.lr_span_probabilities,
            entities=entities,
            links=links,
        )

    def _generate_forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        return_cache: bool = False,
    ) -> ITERGenerateBaseOutput:
        batch_size, seq_len, hidden = x.size()
        nested_inference = self.is_feature_nest_depth_gt_1
        attention_mask = attention_mask.bool()

        # [B, N]
        logits = self.is_l(x).squeeze(-1)
        boundary_decode_weight = float(
            getattr(self.config, "ner_boundary_decode_weight", 0.0)
        )
        # This is the critical path for recall: the auxiliary start head must
        # participate before left candidates are selected.  Adding it only to
        # already formed spans cannot recover a missed entity start.
        fuse_start_candidates = bool(
            getattr(self.config, "ner_boundary_start_candidate_fusion", False)
        )
        if not nested_inference and boundary_decode_weight and fuse_start_candidates:
            logits = logits + boundary_decode_weight * self.ner_boundary_start(x).squeeze(-1)
        indices = inclusive_cumsum(attention_mask.long(), 1)

        is_left = torch.gt(logits, 0) & attention_mask
        # [B, num_l]
        num_left = is_left.sum(dim=1).amax().clamp_min(self.max_nest_depth)
        is_left_positions = x.new_empty((batch_size, num_left), dtype=torch.long)
        is_left_positions, is_left_mask = batched_index_padded(is_left, 0, out=is_left_positions, return_mask=True)

        if not nested_inference:
            # [B, N]
            closest_left_indices = torch.cumsum(is_left, dim=1) - 1
            closest_left_mask = torch.ge(closest_left_indices, 0) & attention_mask
            closest_left_indices.masked_fill_(~closest_left_mask, 0)

            # [B, num_l]
            closest_left_positions = torch.gather(
                is_left_positions,
                dim=1,
                index=closest_left_indices
            ).masked_fill(~closest_left_mask, 0)
        else:
            # [B, N, nest_depth, dim]
            x = expand_dim(x.unsqueeze(2), 2, self.max_nest_depth)

            # [B, N, num_l]
            distance_to_previous_left = indices.unsqueeze(2) - is_left_positions.unsqueeze(1)
            distance_to_previous_left.masked_fill_(~is_left_mask.unsqueeze(1), -1)
            distance_to_previous_left_mask = distance_to_previous_left >= 0
            distance_to_previous_left_mask &= attention_mask.unsqueeze(2).bool()
            distance_to_previous_left.masked_fill_(~distance_to_previous_left_mask, torch.iinfo(torch.long).max)

            # [B, N, nest_depth]
            closest_left_indices = torch.topk(
                distance_to_previous_left,
                k=self.max_nest_depth,
                dim=2,
                largest=False
            )[1]
            # ... replace indices 0..num_l by their positions 0..N
            closest_left_positions = torch.gather(
                expand_dim(is_left_positions.unsqueeze(1), 1, seq_len),
                dim=2,
                index=closest_left_indices
            )
            # ... and copy their mask
            closest_left_mask = torch.gather(
                distance_to_previous_left_mask,
                dim=2,
                index=closest_left_indices
            )

        # [B, N, dim] | [B, N, nest_depth, dim]
        is_left_hidden = torch.gather(
            x,
            dim=1,
            index=unsqueeze_and_expand_dims(
                closest_left_positions,
                (2 + nested_inference,),
                (2 + nested_inference, hidden)
            )
        )

        # [B, N, 2*dim | dim] | [B, N, nest_depth, 2*dim | dim]
        width = None
        if not nested_inference:
            width = torch.arange(seq_len, device=x.device).view(1, -1) - closest_left_positions
        else:
            width = torch.arange(seq_len, device=x.device).view(1, -1, 1) - closest_left_positions
        # [B, N, num_types] | [B, N, nest_depth, num_types]
        logits = self._score_span_logits(is_left_hidden, x, width)

        # The auxiliary boundary heads and independent type classifier are
        # optionally fused into the decoder.  Defaults are zero so legacy
        # checkpoints retain their original decoding behavior.
        if not nested_inference:
            boundary_weight = boundary_decode_weight
            type_weight = float(getattr(self.config, "ner_type_decode_weight", 0.0))
            if boundary_weight:
                start_boundary = self.ner_boundary_start(x).squeeze(-1)
                end_boundary = self.ner_boundary_end(x).squeeze(-1)
                nearest_start = start_boundary.gather(
                    1, closest_left_indices.clamp_min(0)
                )
                boundary_adjustment = 0.5 * (nearest_start + end_boundary)
                logits = logits + boundary_weight * boundary_adjustment.unsqueeze(-1)
            if type_weight:
                pair_hidden = self._merge_representations(is_left_hidden, x)
                logits = logits + type_weight * self.ner_type_classifier(pair_hidden)
            prototype_decode_weight = float(
                getattr(self.config, "ner_type_prototype_decode_weight", 0.0)
            )
            if prototype_decode_weight and self.ner_type_prototypes is not None:
                pair_hidden = self._merge_representations(is_left_hidden, x)
                logits = logits + prototype_decode_weight * self._score_ner_type_prototypes(
                    pair_hidden
                )

        float_inf = float("-inf")
        logits.masked_fill_(~closest_left_mask.unsqueeze(2 + nested_inference), float_inf)
        logits.masked_fill_(~closest_left_mask.unsqueeze(2 + nested_inference), float_inf)
        logits.masked_fill_(~unsqueeze_dims(attention_mask, (2,) * (1 + nested_inference)).bool(), float_inf)

        # For flat NER (the long-context dataset has nest depth 1), expose the
        # posterior used by the right-bracket decision.  The implicit
        # no-span score is zero, hence p(span)=sigmoid(logsumexp(type_logits))
        # and p(type)=softmax([0, type_logits])[type].  Invalid/padded
        # positions are zeroed before leaving the model.
        lr_pair_probabilities = None
        lr_span_probabilities = None
        if not nested_inference:
            num_left = is_left_positions.size(1)
            valid_candidates = closest_left_mask & attention_mask.bool()
            span_logit = logits.logsumexp(dim=2)
            none_and_types = torch.logaddexp(torch.zeros_like(span_logit), span_logit)
            type_probabilities = torch.exp(logits - none_and_types.unsqueeze(-1))
            type_probabilities = torch.where(
                valid_candidates.unsqueeze(-1),
                type_probabilities,
                torch.zeros_like(type_probabilities),
            )
            type_probabilities = torch.nan_to_num(
                type_probabilities, nan=0.0, posinf=1.0, neginf=0.0
            )
            span_probabilities = torch.sigmoid(span_logit).masked_fill(
                ~valid_candidates, 0.0
            )

            lr_pair_probabilities = x.new_zeros(
                (batch_size, seq_len, num_left, self.num_types)
            )
            pair_indices = closest_left_indices.clamp_min(0).unsqueeze(-1).unsqueeze(-1)
            pair_indices = pair_indices.expand(-1, -1, 1, self.num_types)
            lr_pair_probabilities.scatter_(
                2, pair_indices, type_probabilities.unsqueeze(2)
            )
            lr_span_probabilities = x.new_zeros((batch_size, seq_len, num_left))
            lr_span_probabilities.scatter_(
                2,
                closest_left_indices.clamp_min(0).unsqueeze(-1),
                span_probabilities.unsqueeze(-1),
            )

        if nested_inference:
            # [B, N, nest_depth, 1]
            compare_to = logits[..., [-1]]
            # [B, N, nest_depth, num_types - 1]
            denominator = logits[..., :-1] > compare_to
            # [B, N, nest_depth]
            denominator = denominator.any(dim=3)
            # [B, N]
            is_right = denominator.any(dim=2)
        else:
            # [B, N]
            denominator = logits.logsumexp(dim=2, keepdim=False)
            is_right = torch.gt(denominator, 0)

        # [B, N]
        actions = (is_left + is_right * 2) * attention_mask

        # (1,)
        num_left = is_left_positions.size(1)
        # [B, N, num_left, num_types]
        lr_pair_flag = x.new_zeros(
            (batch_size, seq_len, num_left, self.num_types),
            dtype=torch.bool
        )
        if nested_inference:
            lr_pair_flag[..., -1] = True

        if not nested_inference:
            # [B, N, 1, 1]
            scatter_indices = unsqueeze_dims(closest_left_indices, 2, 3)
        else:
            # [B, N, nest_depth, 1]
            scatter_indices = unsqueeze_dims(closest_left_indices, 3)
        scatter_indices = expand_dim(scatter_indices, 3, self.num_types)

        # [B, N, 1] | [B, N, nest_depth]
        one_hot_type_input = logits.argmax(dim=2 + nested_inference, keepdim=not nested_inference)
        if nested_inference:
            one_hot_type_input = one_hot_type_input.masked_fill_(~closest_left_mask, self.num_types - 1)
        # [B, N, 1 | nest_depth, num_types]
        one_hot_type_choice = F.one_hot(one_hot_type_input, num_classes=self.num_types).to(torch.bool)
        if not nested_inference:
            one_hot_type_choice = one_hot_type_choice * logits.logsumexp(dim=2, keepdim=True).gt(0).unsqueeze(3)

        # perform the scatter into lr_pair_flag, which we use to decode the output
        lr_pair_flag = torch.scatter(
            lr_pair_flag,
            dim=2,
            index=scatter_indices,
            src=one_hot_type_choice
        )
        cache = None
        if return_cache:
            cache = ITERGenerateBaseCache(closest_left_indices, closest_left_mask, is_left_hidden)
        return ITERGenerateBaseOutput(
            actions,
            lr_pair_flag,
            cache,
            lr_pair_probabilities,
            lr_span_probabilities,
        )

    def _is_left_forward(
        self,
        x: Tensor,
        actions: Tensor
    ) -> Tensor:
        attention_mask = ~is_action(actions, ITER_ACTION_PAD_BIT)
        # [B, N, 1]
        is_left = self.is_l(x)
        no_action = torch.zeros_like(is_left)
        # [B, N, 2]
        is_left = torch.cat((no_action, is_left), dim=2)

        # [B, N]
        numerator = torch.logsumexp(
            is_left + torch.where(
                is_action(actions, ITER_ACTION_LEFT_BRACKET_BIT, True, torch.bool),
                0.,
                torch.finfo(x.dtype).min
            ),
            dim=2
        ) * attention_mask
        denominator = torch.logsumexp(is_left, dim=2) * attention_mask
        return denominator.sub(numerator).mul(attention_mask).sum()

    def _is_span_forward(
        self,
        x: Tensor,
        actions: Tensor,
        lr_pair_flag: Tensor,
    ) -> Tensor:
        bs, seq_len, dim = x.shape
        num_l = lr_pair_flag.size(2)
        device = x.device

        # [B, N]
        attention_mask = ~is_action(actions, ITER_ACTION_PAD_BIT)
        is_left = is_action(actions, ITER_ACTION_LEFT_BRACKET_BIT)
        is_right = is_action(actions, ITER_ACTION_RIGHT_BRACKET_BIT)

        # [B, num_l]
        left_positions = x.new_empty((bs, num_l), dtype=torch.long)
        left_positions, left_mask = batched_index_padded(is_left, 0, out=left_positions, return_mask=True)

        # [B, N, num_l]
        distance_to_previous_left = inclusive_cumsum(attention_mask.long(), 1).unsqueeze(2) - left_positions.unsqueeze(1)
        is_after_or_at_left = distance_to_previous_left >= 0
        is_after_or_at_left = is_after_or_at_left & attention_mask.unsqueeze(2) & left_mask.unsqueeze(1)

        nest_depth = min(self.max_nest_depth, num_l)
        if nest_depth == 0:
            return x.new_zeros(tuple(), dtype=x.dtype)

        # [B, N, nest_depth]
        closest_left_weights: Tensor
        closest_left_indices: Tensor
        closest_left_weights, closest_left_indices = torch.where(
            (distance_to_previous_left >= 0) & is_after_or_at_left,
            distance_to_previous_left,
            torch.iinfo(distance_to_previous_left.dtype).max
        ).topk(nest_depth, dim=2, largest=False)
        closest_left_mask = closest_left_weights < seq_len

        # [B, N, nest_depth]
        closest_left_positions = torch.gather(
            expand_dim(left_positions.unsqueeze(1), 1, seq_len),
            dim=2,
            index=closest_left_indices
        )
        # [B, N, nest_depth]
        closest_left_is_after_or_at_left = torch.gather(
            is_after_or_at_left,
            dim=2,
            index=closest_left_indices
        )
        # [B, N, nest_depth, num_types]
        closest_left_lr_pair_flag = torch.gather(
            lr_pair_flag,
            dim=2,
            index=expand_dim(closest_left_indices.unsqueeze(3), 3, self.num_types)
        ) * closest_left_mask.unsqueeze(3)

        x = expand_dim(x.unsqueeze(2), 2, nest_depth)
        # prepare closest_left_positions such that we can gather from x in dim=1
        # [B, N, nest_depth, H]
        closest_left_hidden = torch.gather(
            x,
            dim=1,
            index=expand_dim(closest_left_positions.unsqueeze(3), 3, dim).clamp_min(0)
        )

        width = torch.arange(seq_len, device=x.device).view(1, -1, 1) - closest_left_positions
        # [B, N, nest_depth, num_types]
        logits = self._score_span_logits(closest_left_hidden, x, width)
        logits = logits + (~closest_left_is_after_or_at_left.unsqueeze(3) * NEG_INF)

        mask = closest_left_is_after_or_at_left.any(dim=2, keepdim=True)
        nested_training = self.is_feature_nest_depth_gt_1
        if nested_training:
            assert_true(self.is_feature_extra_lr_class, "For datasets with nest_depth > 1, you need to add an "
                                                        "additional NONE class to the num_types dimension.")
            mask = mask & closest_left_lr_pair_flag.any(dim=3)
            denominator = torch.logsumexp(
                logits,
                dim=3
            ) * mask
            numerator = torch.logsumexp(
                logits + (~closest_left_lr_pair_flag * NEG_INF),
                dim=3
            ) * mask

            denominator = denominator.unsqueeze(3)
            numerator = numerator.unsqueeze(3)
        else:
            denominator = torch.logsumexp(
                logits,
                dim=(2, 3)
            ).unsqueeze(2) * mask
            numerator = torch.logsumexp(
                logits + (~closest_left_lr_pair_flag * NEG_INF),
                dim=(2, 3)
            ).unsqueeze(2) * mask

        no_action = torch.zeros_like(denominator)
        denominator = torch.logsumexp(
            torch.cat((no_action, denominator), dim=2 + nested_training),
            dim=2 + nested_training
        )

        numerator = torch.logsumexp(
            torch.cat((
                no_action + torch.where(
                    ~unsqueeze_dims(is_right, (2,) * (1 + nested_training)),
                    0,
                    torch.finfo(no_action.dtype).min
                ),
                numerator
            ), dim=2 + nested_training),
            dim=2 + nested_training
        )
        sub_mask = attention_mask
        if nested_training:
            sub_mask = attention_mask.unsqueeze(2)

        return denominator.sub(numerator).mul(sub_mask).sum()

    def _merge_representations(self, head: Tensor, tail: Tensor):
        if self.is_feature_sum_representations:
            raise NotImplementedError

        torch._check(head.dim() == tail.dim())
        return torch.cat((head, tail), dim=-1)


class ITERForRelationExtraction(ITER):
    def __init__(self, config: ITERConfig):
        random_state = torch.get_rng_state()
        super().__init__(config)
        torch.set_rng_state(random_state)

        self.threshold = config.threshold
        self.num_links = config.num_links

        model_cls = config.guess_model_class()
        model_kwargs = config.model_kwargs
        try:
            self.model = model_cls.from_pretrained(
                config.transformer_config.name_or_path, **model_kwargs,
            )
        except (OSError, EnvironmentError):
            self.model = model_cls(config.transformer_config, **model_kwargs)
        self.is_l = IsLeft(config)
        self.rr_score = IsLink(config)
        self.lr_score = IsSpan(config)

        # --- Initialize new Wenyanwen Geo-NER modules ---
        hidden_dim = config.d_model  # [B, N, H] hidden dimension

        # Fact Decomposition Module (JEDI-inspired two-stage design)
        # Enabled via either explicit config flag OR feature bit 22
        use_fact = config.use_fact_decomposition or config.is_feature_fact_decomposition
        if use_fact:
            self.fact_module = FactDecompositionModule(
                hidden_dim=hidden_dim,
                num_entity_types=config.num_types,
                num_relation_types=config.num_links,
                num_bilinear_groups=config.num_bilinear_groups,
                dropout=config.dropout,
                max_span_length=config.max_span_length,
            )
        else:
            self.fact_module = None

        # Multi-Scale Geographic Representation Module
        # Enabled via either explicit config flag OR feature bit 23
        use_multiscale = config.use_multiscale_geo or config.is_feature_multiscale_geo
        if use_multiscale:
            self.multiscale_module = MultiScaleGeoModule(
                hidden_dim=hidden_dim,
                kernel_sizes=config.cnn_kernel_sizes,
                num_filters_per_kernel=128,
                num_attention_heads=8,
                max_span_length=min(config.max_span_length, 10),
                dropout=config.dropout,
            )
        else:
            self.multiscale_module = None

        # --- Task-Specific Projections for Parallel Branches ---
        # Each module gets its own projection from the shared encoder output,
        # preventing gradient interference and allowing each module to learn
        # its own "view" of the shared representation.
        # Enabled when BOTH modules are active (to address negative transfer).
        self.use_parallel_branches = (
            self.multiscale_module is not None and self.fact_module is not None
        )
        if self.use_parallel_branches:
            # Independent linear projections from encoder space to task-specific subspaces
            self.proj_multiscale = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            self.proj_fact = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            # Learned gating for fusing parallel branch outputs with original encoder output
            # gate = softmax over three learned scalars: [multiscale, fact, skip]
            self.fusion_gate = nn.Parameter(torch.zeros(3))
            # Layer norm for the fused output
            self.fusion_layer_norm = nn.LayerNorm(hidden_dim)

        # Number of iterative refinement steps
        self.num_iterations = config.num_iterations

        # Loss weights
        self.fact_loss_weight = config.fact_loss_weight
        self.multiscale_loss_weight = config.multiscale_loss_weight
        # --- End new module initialization ---

    def resize_num_links(self, new_num_links: int):
        self.rr_score.resize(new_num_links)
        self.num_links = new_num_links

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        actions: Tensor,
        lr_pair_flag: Tensor,
        rr_pair_flag: Tensor = None,
    ) -> ITEROutput:
        """
        Args:
            input_ids (Tensor): The tokens the model receives as input of shape [B, N]
            attention_mask (Tensor): The attention mask corresponding to the ``input_ids``
            actions (Tensor): Target actions for each position in a sequence. Shape: [B, N]
                There are two actions the model can perform at a given position: left `[` and right `]`
                bracket pairings. Each action is encoded by a different bit in ``actions`` being set: bit ``0`` for
                left and bit ``1`` for right bracket pairings.
            lr_pair_flag (Tensor): A Boolean tensor of shape [B, N, num_left, num_types], where a `True` value at
                position (b, 1<=n<=N, l, t) indicates a pairing between the l-th bracket and position n of type t.
            rr_pair_flag (Tensor): A Boolean tensor of shape [B, N, num_right, num_links], where a `True` value at
                position (b, 1<=n<=N, r, t) indicates a link between the r-th span and the span ending at n of type t.

        Returns: ITEROutput with loss, hidden_state, l_loss, lr_loss, rr_loss,
                 plus fact_loss and multiscale_loss when corresponding modules are enabled.
        """
        # [B, N, H] — Base encoder forward pass
        x = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )[0]

        x = self._apply_long_context_refiner(x, attention_mask)
        x = self._apply_ner_target_adapter(x, attention_mask)

        # --- Multi-Scale Geographic + Fact Decomposition ---
        multiscale_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        fact_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        branch_outputs = None

        has_multiscale = self.multiscale_module is not None
        has_fact = self.fact_module is not None

        if has_multiscale and has_fact and self.use_parallel_branches:
            # =================================================================
            # PARALLEL BRANCH MODE: Each module gets its own projected view of
            # the shared encoder output. Gradients from each module do NOT flow
            # through the other module. A learned gate fuses all outputs.
            #
            # This addresses negative transfer by:
            #   1. Task-specific projections → independent feature subspaces
            #   2. Gradient isolation (detach) → no cross-module gradient interference
            #   3. Learned fusion gate → model decides optimal feature mix
            # =================================================================

            # --- Branch A: Multiscale Geo ---
            x_ms = self.proj_multiscale(x)
            enhanced_x, branch_outputs = self.multiscale_module(
                x_ms, attention_mask, return_branches=True
            )
            if self.is_feature_multiscale_consistency_loss and branch_outputs is not None:
                multiscale_loss = self.multiscale_module.compute_consistency_loss(branch_outputs)

            # --- Branch B: Fact Decomposition ---
            # When gradient isolation is enabled, detach() prevents fact loss
            # gradients from flowing back into the shared encoder and multiscale branch.
            # This is the strongest defense against negative transfer.
            use_isolation = self.is_feature_gradient_isolation and self.training
            x_fact_input = x.detach() if use_isolation else x
            x_fact = self.proj_fact(x_fact_input)
            fact_outputs = self.fact_module(x_fact, attention_mask)

            if actions is not None and lr_pair_flag is not None:
                is_left = actions.bitwise_and(1).ne(0)
                is_right = actions.bitwise_and(2).ne(0)
                left_positions, left_mask = batched_index_padded(is_left, 0, return_mask=True)
                right_positions, right_mask = batched_index_padded(is_right, 0, return_mask=True)
                target_types = torch.argmax(lr_pair_flag.float(), dim=-1)
                target_types_flat = target_types.amax(dim=2)
                _, _, fact_loss = self.fact_module.compute_fact_loss(
                    span_outputs=fact_outputs['span_outputs'],
                    target_starts=left_positions,
                    target_ends=right_positions,
                    target_types=target_types_flat,
                    target_relations=None,
                    relation_logits=fact_outputs['relation_logits'],
                )

            # --- Learned Gated Fusion ---
            # softmax over three learned scalars → positive weights that sum to 1
            gate_weights = F.softmax(self.fusion_gate, dim=0)  # [3]
            # enhanced_x already includes residual inside multiscale_module;
            # we fuse: gate_ms * enhanced_ms + gate_fact * projected_fact + gate_skip * original_x
            fused = (
                gate_weights[0] * enhanced_x +
                gate_weights[1] * x_fact +
                gate_weights[2] * x
            )
            x = self.fusion_layer_norm(fused)

        elif has_multiscale:
            # --- Sequential: Multiscale only ---
            enhanced_x, branch_outputs = self.multiscale_module(
                x, attention_mask, return_branches=True
            )
            if self.is_feature_multiscale_consistency_loss and branch_outputs is not None:
                multiscale_loss = self.multiscale_module.compute_consistency_loss(branch_outputs)
            x = enhanced_x

        elif has_fact:
            # --- Sequential: Fact decomposition only ---
            fact_outputs = self.fact_module(x, attention_mask)
            if actions is not None and lr_pair_flag is not None:
                is_left = actions.bitwise_and(1).ne(0)
                is_right = actions.bitwise_and(2).ne(0)
                left_positions, left_mask = batched_index_padded(is_left, 0, return_mask=True)
                right_positions, right_mask = batched_index_padded(is_right, 0, return_mask=True)
                target_types = torch.argmax(lr_pair_flag.float(), dim=-1)
                target_types_flat = target_types.amax(dim=2)
                _, _, fact_loss = self.fact_module.compute_fact_loss(
                    span_outputs=fact_outputs['span_outputs'],
                    target_starts=left_positions,
                    target_ends=right_positions,
                    target_types=target_types_flat,
                    target_relations=None,
                    relation_logits=fact_outputs['relation_logits'],
                )

        # --- ITER Original Losses ---
        lr_loss = self._is_span_forward(
            x,
            actions,
            lr_pair_flag
        )
        rr_loss = self._is_link_forward(
            x,
            actions,
            rr_pair_flag
        )
        l_loss = self._is_left_forward(x, actions)
        ner_type_loss = self._ner_type_auxiliary_loss(x, actions, lr_pair_flag)
        ner_boundary_loss = self._ner_boundary_auxiliary_loss(x, actions, attention_mask)
        ner_span_margin_loss = self._ner_span_margin_loss(x, actions, lr_pair_flag, attention_mask)

        # --- Combined Loss ---
        # L = L_iter + λ1 * L_fact + λ2 * L_multiscale
        loss = l_loss + lr_loss + rr_loss + ner_type_loss + ner_boundary_loss + ner_span_margin_loss
        loss = loss + self.fact_loss_weight * fact_loss
        loss = loss + self.multiscale_loss_weight * multiscale_loss

        if self.is_feature_batch_average_loss:
            loss = loss / input_ids.size(0)

        # Store additional losses in the output for monitoring
        output = ITEROutput(loss, x, l_loss, lr_loss, rr_loss)
        # Attach additional losses for monitoring (not part of the dataclass)
        output.fact_loss = fact_loss
        output.multiscale_loss = multiscale_loss
        output.ner_type_loss = ner_type_loss
        output.ner_boundary_loss = ner_boundary_loss
        output.ner_span_margin_loss = ner_span_margin_loss

        return output

    def iterative_refine(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        num_iterations: int = 3,
    ) -> dict:
        """
        Iterative fact-aware refinement of entity boundaries and relations.

        At each iteration:
        1. Select candidate facts (entity spans + relation hypotheses) from current state
        2. Update entity memory with fact-aware representations
        3. Refine entity boundaries using boundary refiner

        This method leverages both fact decomposition outputs and multi-scale
        geographic representations for improved accuracy.

        Args:
            hidden_states: [B, N, H] encoded token representations
            attention_mask: [B, N] attention mask
            num_iterations: Number of refinement iterations

        Returns:
            dict with refined entity/relation predictions
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        # Initialize entity memory: [B, K, H] where K is max candidate entities
        max_entities = min(seq_len // 2, 50)  # Reasonable upper bound

        # Get initial entity candidates using the ITER bracket approach
        # Use the base class's _generate_forward for initial predictions
        base_output = super()._generate_forward(hidden_states, attention_mask)
        actions = base_output.actions
        lr_pair_flag = base_output.lr_pair_flag

        # Initialize entity memory from initial predictions
        is_left = actions.bitwise_and(1).ne(0)
        is_right = actions.bitwise_and(2).ne(0)

        left_positions, left_mask = batched_index_padded(is_left, 0, return_mask=True)
        right_positions, right_mask = batched_index_padded(is_right, 0, return_mask=True)

        # [B, K, H] entity memory: store representations of entity spans
        entity_memory = torch.zeros(batch_size, max_entities, hidden_dim, device=device, dtype=hidden_states.dtype)

        # Populate initial entity memory with span representations
        for b in range(batch_size):
            for k in range(min(left_positions.size(1), max_entities)):
                if left_mask[b, k]:
                    start = left_positions[b, k].item()
                    # Find corresponding right bracket
                    # For simplicity, use mean pooling of span tokens
                    if k < right_positions.size(1) and right_mask[b, k]:
                        end = right_positions[b, k].item()
                        if end >= start:
                            span_tokens = hidden_states[b, start:end + 1, :]  # [span_len, H]
                            entity_memory[b, k] = span_tokens.mean(dim=0)

        # Iterative refinement loop
        for iteration in range(num_iterations):
            # Step 1: Select candidate facts using current entity memory
            # For each position, compute attention over entity memory
            # [B, N, K]: attention weights from each position to each entity
            entity_attention = torch.einsum(
                'bnh,bkh->bnk',
                hidden_states,
                entity_memory
            ) / (hidden_dim ** 0.5)  # Scaled dot-product

            entity_attention = F.softmax(entity_attention, dim=-1)  # [B, N, K]

            # Step 2: Update entity memory with context-aware representations
            # [B, K, H]: aggregate context per entity
            context_per_entity = torch.einsum(
                'bnk,bnh->bkh',
                entity_attention,
                hidden_states
            )
            # Update memory with gated residual
            gate = torch.sigmoid(
                torch.einsum('bkh,bkh->bk', entity_memory, context_per_entity).unsqueeze(-1)
            )
            entity_memory = gate * entity_memory + (1 - gate) * context_per_entity

            # Step 3: Refine entity boundaries (if fact module is available)
            if self.fact_module is not None:
                # Use fact module to refine boundaries
                refined_starts, refined_ends = self.fact_module.refine_boundaries(
                    hidden_states,
                    left_positions.clamp(min=0),
                    right_positions.clamp(min=0),
                )
                left_positions = torch.where(left_mask, refined_starts, left_positions)
                right_positions = torch.where(right_mask, refined_ends, right_positions)

        return {
            'entity_memory': entity_memory,
            'refined_starts': left_positions,
            'refined_ends': right_positions,
            'actions': actions,
            'lr_pair_flag': lr_pair_flag,
        }

    def _merge_link_representations(self, head: Tensor, tail: Tensor):
        torch._check(head.dim() == tail.dim())
        return torch.cat((head, tail), dim=-1)

    def _is_link_forward(
        self,
        x: Tensor,
        actions: Tensor,
        rr_pair_flag: Tensor
    ) -> Tensor:
        assert not self.is_feature_ner_only
        if self.is_feature_use_lse_rr_loss:
            raise NotImplementedError()

        nested_training = self.is_feature_nest_depth_gt_1
        bs, seq_len, dim = x.shape
        attention_mask = actions.bitwise_and(4).eq(0)
        is_right = actions.bitwise_and(2).ne(0)
        is_left = actions.bitwise_and(1).ne(0)

        is_right_positions, is_right_mask = batched_index_padded(is_right, 0, return_mask=True)
        if is_right_positions.size(1) == 0:
            return x.new_zeros(tuple())

        # [B, num_right, dim]
        is_right_hidden = torch.gather(
            x,
            dim=1,
            index=unsqueeze_and_expand_dims(
                is_right_positions,
                (2,), (2, dim)
            )
        ) * is_right_mask.unsqueeze(2)

        if nested_training:
            assert self.is_feature_extra_lr_class

            is_left_positions, is_left_mask = batched_index_padded(is_left, 0, return_mask=True)

            distance_to_previous_left = inclusive_cumsum(attention_mask.long(), 1).unsqueeze(2) - is_left_positions.unsqueeze(1)
            distance_to_previous_left.masked_fill_(~is_left_mask.unsqueeze(1), -1)

            is_after_or_at_left = distance_to_previous_left >= 0
            is_after_or_at_left = is_after_or_at_left & attention_mask.unsqueeze(2) & is_left_mask.unsqueeze(1)

            nest_depth = min(self.max_nest_depth, is_left_positions.size(1))
            if nest_depth == 0:
                return x.new_zeros(tuple())
            num_right = is_right_positions.size(1)
            num_left = is_left_positions.size(1)

            # [B, N, nest_depth]
            closest_left_weights: Tensor
            closest_left_indices: Tensor
            closest_left_weights, closest_left_indices = torch.where(
                (distance_to_previous_left >= 0) & is_after_or_at_left,
                distance_to_previous_left,
                distance_to_previous_left + (torch.iinfo(distance_to_previous_left.dtype).max >> 4)
            ).topk(nest_depth, dim=2, largest=False)
            closest_left_mask = closest_left_weights < seq_len

            # [B, N, nest_depth]
            closest_left_positions = torch.gather(
                expand_dim(is_left_positions.unsqueeze(1), 1, seq_len),
                dim=2,
                index=closest_left_indices
            )

            x = expand_dim(x.unsqueeze(2), 2, nest_depth)
            # prepare closest_left_positions such that we can gather from x in dim=1
            # [B, N, nest_depth, H]
            closest_left_hidden = torch.gather(
                x,
                dim=1,
                index=expand_dim(closest_left_positions.unsqueeze(3), 3, dim).clamp_min(0)
            )

            # [B, N, nest_depth, H]
            # for 0 <= n <= N: out[n] = x[n] + x[closest_left[x]]
            closest_span_hidden = torch.add(x, closest_left_hidden)

            # [B, num_right, nest_depth]
            is_right_closest_left_mask = torch.gather(
                closest_left_mask,
                dim=1,
                index=unsqueeze_and_expand_dims(is_right_positions, 2, (2, nest_depth))
            )
            is_right_closest_left_indices = torch.gather(
                closest_left_indices,
                dim=1,
                index=unsqueeze_and_expand_dims(is_right_positions, 2, (2, nest_depth))
            )
            # [B, num_right, nest_depth, H]
            is_right_closest_left_hidden = torch.gather(
                closest_span_hidden,
                dim=1,
                index=unsqueeze_and_expand_dims(
                    is_right_positions,
                    (2, 3),
                    (2, nest_depth), (3, dim),
                )
            )

            if rr_pair_flag.size(3) == 0:
                return x.new_zeros(tuple())
            # rr_pair_flag has dim [B, N, num_left, num_right, num_left, T]
            # dim0 = B
            # dim2 = num_left (sequence)
            # dim4 = num_left (span)
            # [B, N, nest_depth, num_right, num_left, T]
            num_left_to_nest_depth_index = unsqueeze_and_expand_dims(
                # [B, N, nest_depth] -> [B, N, nest_depth, num_right, num_left, T]
                closest_left_indices,
                (3, 4, 5),
                (3, num_right), (4, num_left), (5, self.num_links),
            )
            rr_pair_flag = torch.gather(
                rr_pair_flag,
                dim=2,
                index=num_left_to_nest_depth_index
            )
            # [B, N, nest_depth, num_right, nest_depth, T]
            # here is the problem
            rr_pair_flag = torch.gather(
                rr_pair_flag,
                dim=4,
                index=unsqueeze_and_expand_dims(
                    is_right_closest_left_indices,
                    (1, 2, 5),
                    (1, seq_len), (2, nest_depth), (5, self.num_links)
                )
            )

            # [B, N, nest_depth, num_right, nest_depth, 2*dim]
            logits = self._merge_link_representations(
                unsqueeze_and_expand_dims(is_right_closest_left_hidden, (1, 2,), (1, seq_len), (2, nest_depth)),
                unsqueeze_and_expand_dims(closest_span_hidden, (3, 4), (3, num_right), (4, nest_depth))
            )
            # [B, N, nest_depth, num_right, nest_depth, T]
            logits = self.rr_score(logits)
            # [B, N, is_right]
            logits_mask = is_right_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
            # closest_left_mask: [B, N, nest_depth]
            # is_right_closest_left_mask: [B, num_right, nest_depth]
            logits_mask = unsqueeze_dims(logits_mask, 2, 4)
            logits_mask = logits_mask & unsqueeze_dims(closest_left_mask, 3, 4) & unsqueeze_dims(is_right_closest_left_mask, 1, 2)
        else:
            num_r = rr_pair_flag.size(2)

            # [B, N, num_right, 2*dim]
            logits = self._merge_link_representations(
                unsqueeze_and_expand_dims(is_right_hidden, (1,), (1, seq_len)),
                unsqueeze_and_expand_dims(x, (2,), (2, num_r))
            )
            # [B, N, num_right, num_links]
            logits = self.rr_score(logits)
            logits_mask = is_right_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
            if self.is_feature_in_development:
                logits_mask = is_right_mask.unsqueeze(1) & is_right.unsqueeze(2)

        if not logits_mask.any():
            return logits.new_zeros(tuple())

        input = logits[logits_mask]
        target = rr_pair_flag[logits_mask].float()
        if self.is_feature_extra_rr_class:
            return F.cross_entropy(input, target, reduction="sum")
        else:
            return F.binary_cross_entropy_with_logits(input, target, reduction="sum")

    def _generate_forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        return_cache: Optional[bool] = False,
    ) -> ITERGenerateOutput:
        base_output = super()._generate_forward(x, attention_mask, return_cache=True)

        bs, seq_len, dim = x.shape
        nested_inference = self.is_feature_nest_depth_gt_1
        assert not nested_inference or self.is_feature_extra_lr_class

        # [B, N]
        actions = base_output.actions
        is_left = actions.bitwise_and(1).ne(0)
        is_right = actions.bitwise_and(2).ne(0)

        is_left_positions, is_left_mask = batched_index_padded(is_left, 0, return_mask=True)
        num_left = is_left_positions.size(1)

        # [B, num_right]
        is_right_positions, is_right_mask = batched_index_padded(is_right, 0, return_mask=True)
        num_right = is_right_positions.size(1)

        dims = (bs, seq_len, num_right, self.num_links)
        if nested_inference:
            dims = (bs, seq_len, num_left, num_right, num_left, self.num_links)
        rr_pair_flag = x.new_zeros(dims, dtype=torch.bool)
        rr_pair_probabilities = x.new_zeros(dims)

        if num_right <= 0:
            return ITERGenerateOutput(
                actions=base_output.actions,
                lr_pair_flag=base_output.lr_pair_flag,
                cache=base_output.cache,
                lr_pair_probabilities=base_output.lr_pair_probabilities,
                lr_span_probabilities=base_output.lr_span_probabilities,
                rr_pair_flag=rr_pair_flag,
                rr_pair_probabilities=rr_pair_probabilities,
            )

        # [B, num_r, dim]
        is_right_hidden = torch.gather(
            x,
            dim=1,
            index=expand_dim(is_right_positions.unsqueeze(2), 2, dim),
        )

        closest_left_mask = None
        if nested_inference:
            # [B, num_right, nest_depth, dim]
            is_right_hidden = expand_dim(
                is_right_hidden.unsqueeze(2), 2, self.max_nest_depth
            )
            closest_left_hidden = base_output.cache[2]

            # [B, num_right, nest_depth, dim]
            closest_left_hidden_at_right_positions = torch.gather(
                closest_left_hidden,
                dim=1,
                index=unsqueeze_dims(is_right_positions, 2, 3).expand(-1, -1, self.max_nest_depth, dim)
            )
            is_right_hidden = is_right_hidden + closest_left_hidden_at_right_positions


            # [B, num_right, nest_depth]
            closest_left_mask = torch.gather(
                # [B, seq_len, nest_depth]
                base_output.cache.closest_left_mask,
                dim=1,
                index=unsqueeze_and_expand_dims(
                    is_right_positions,
                    2,
                    (2, self.max_nest_depth)
                )
            )
        pass

        if not nested_inference:
            # [B, 1, num_right, dim]
            is_right_hidden = is_right_hidden

            def proj(inp: Tensor, proj_dim: int):
                return unsqueeze_and_expand_dims(inp, (proj_dim,), (proj_dim, num_right))

            # [B, num_right, num_right, dim]
            inner = proj(is_right_hidden, 1)
            outer = proj(is_right_hidden, 2)
            # [B, num_right, num_right]
            inner_mask = proj(is_right_mask, 1)
            outer_mask = proj(is_right_mask, 2)
        else:
            def proj(inp: Tensor, is_inner: bool = False, is_mask: bool = False):
                if is_inner:
                    return unsqueeze_and_expand_dims(
                        inp,
                        (1, 2),
                        (1, num_right), (2, self.max_nest_depth),
                    )
                else:
                    return unsqueeze_and_expand_dims(
                        inp,
                        (3, 4),
                        (3, num_right), (4, self.max_nest_depth),
                    )
            # [B, num_right, nest_depth, num_right, nest_depth, dim]
            inner = proj(is_right_hidden, True, False)
            outer = proj(is_right_hidden, False, False)
            # [B, num_right, nest_depth, num_right, nest_depth]
            inner_outer_mask = unsqueeze_dims(is_right_mask, 2) * closest_left_mask

            inner_mask = proj(inner_outer_mask, True, True)
            outer_mask = proj(inner_outer_mask, False, True)

        compare_mask = inner_mask & outer_mask
        compare = self._merge_link_representations(inner, outer)
        logits = self.rr_score(compare)
        logits.masked_fill_(~compare_mask.unsqueeze(compare_mask.dim()), float("-inf"))
        # Long-context fusion needs calibrated local scores even below the
        # discrete Stage-1 threshold. For the standard non-nested relation
        # layout, map every valid compact right-bracket pair back to its token
        # position. The Boolean rr_pair_flag below remains thresholded.
        if not nested_inference and logits.size(-1) == self.num_links:
            full_probabilities = logits.sigmoid()
            batch_indices = torch.arange(bs, device=x.device).unsqueeze(1).expand_as(is_right_positions)
            valid_batches = batch_indices[is_right_mask]
            valid_head_positions = is_right_positions[is_right_mask]
            # AMP can produce FP16 logits while ``x`` (and therefore the
            # output cache) remains FP32.  Index assignment requires an exact
            # dtype match, so cast probabilities to the cache dtype explicitly.
            rr_pair_probabilities[valid_batches, valid_head_positions] = (
                full_probabilities[is_right_mask].to(dtype=rr_pair_probabilities.dtype)
            )
        if self.is_feature_use_lse_rr_loss or self.is_feature_ce_loss:
            assert self.is_feature_extra_rr_class
            outcome = F.one_hot(logits.argmax(dim=-1), num_classes=self.num_links)
            outcome = outcome[..., :-1]
        else:
            assert not self.is_feature_extra_rr_class
            outcome = logits.sigmoid() > self.threshold

        head_nest, tail_nest = None, None
        head_left_index, tail_left_index = None, None
        # [B, N] | [B, N, nest_depth]
        closest_left_indices = base_output.cache.closest_left_indices
        closest_left_mask = base_output.cache.closest_left_mask

        for indices in outcome.nonzero():  # ugly, causes host<->device synchronization
            link_probability = logits[indices.unbind()].sigmoid()
            if nested_inference:
                batch, head, head_nest, tail, tail_nest, link_type = indices
            else:
                batch, head, tail, link_type = indices

            head_position = is_right_positions[batch, head]
            tail_position = is_right_positions[batch, tail]
            if nested_inference:
                head_left_index = closest_left_indices[batch, head_position, head_nest]
                tail_left_index = closest_left_indices[batch, tail_position, tail_nest]
                if VERIFY_GENERATION:
                    assert closest_left_mask[batch, head_position, head_nest]
                    assert closest_left_mask[batch, tail_position, tail_nest]

            # PL Marker trained their models to predict relationships in both ways
            # ... so head -> link -> tail + tail -> inverse link -> head and in case one is missing,
            # the model still scores correctly because of the set-based F1 eval strategy
            reverse = link_type >= self.num_links
            if reverse:
                if nested_inference:
                    dims = (batch, tail_position, tail_left_index, head, head_left_index, link_type % self.num_links)
                else:
                    dims = (batch, tail_position, head, link_type % self.num_links)
            else:
                if nested_inference:
                    dims = (batch, head_position, head_left_index, tail, tail_left_index, link_type)
                else:
                    dims = (batch, head_position, tail, link_type)
            rr_pair_flag[dims] = True
            rr_pair_probabilities[dims] = link_probability.to(dtype=rr_pair_probabilities.dtype)

        return ITERGenerateOutput(
                actions=base_output.actions,
                lr_pair_flag=base_output.lr_pair_flag,
                cache=base_output.cache,
                lr_pair_probabilities=base_output.lr_pair_probabilities,
                lr_span_probabilities=base_output.lr_span_probabilities,
                rr_pair_flag=rr_pair_flag,
                rr_pair_probabilities=rr_pair_probabilities,
            )
