#!/usr/bin/env python3
"""Train the second-stage document-level relation evidence model.

The local ITER checkpoint is frozen. Its label-independent overlapping windows
are encoded once, entity views are pooled across windows, and a small global
relation head is trained with evidence supervision and non-negative PU risk.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
from pathlib import Path

if "--offline" in sys.argv:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from tqdm import tqdm

from iter import ITER, ITERConfig
from iter.datasets import CoNLL
from iter.misc.document_metrics import safe_prf
from iter.modeling_iter2 import ITERForRelationExtraction
from iter.modules.document_relation import (
    DocumentRelationConfig,
    DocumentRelationModel,
    compute_relation_statistics,
    distance_weighted_positive_loss,
    distance_expert_pu_loss,
    long_distance_pair_ranking_loss,
    pair_type_compatibility_loss,
    relation_positive_focal_loss,
    relation_recall_margin_loss,
    relation_balanced_positive_loss,
    encode_document_features,
    evidence_kl_loss,
    non_negative_pu_loss,
    silver_evidence_distribution,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _relation_targets(
    raw_document: dict,
    pairs: torch.Tensor,
    relation_types: list[str],
    symmetric_relation_ids: set[int],
    entities: list[dict] | None = None,
) -> torch.Tensor:
    relation_to_id = {name: idx for idx, name in enumerate(relation_types)}
    pair_to_row = {tuple(pair): row for row, pair in enumerate(pairs.tolist())}
    source_entities = raw_document.get("entities", [])
    target_entities = entities if entities is not None else source_entities
    boundary_to_ids: dict[tuple[int, int], list[int]] = {}
    for entity_id, entity in enumerate(target_entities):
        boundary_to_ids.setdefault((int(entity["start"]), int(entity["end"])), []).append(entity_id)
    targets = torch.zeros((pairs.size(0), len(relation_types)), dtype=torch.float)
    for relation in raw_document.get("relations", []):
        relation_id = relation_to_id[relation["type"]]
        gold_head = source_entities[int(relation["head"])]
        gold_tail = source_entities[int(relation["tail"])]
        head_ids = boundary_to_ids.get((int(gold_head["start"]), int(gold_head["end"])), [])
        tail_ids = boundary_to_ids.get((int(gold_tail["start"]), int(gold_tail["end"])), [])
        for head_id in head_ids:
            for tail_id in tail_ids:
                head_tail = (head_id, tail_id)
                if head_tail in pair_to_row:
                    targets[pair_to_row[head_tail], relation_id] = 1.0
                if relation_id in symmetric_relation_ids:
                    tail_head = (tail_id, head_id)
                    if tail_head in pair_to_row:
                        targets[pair_to_row[tail_head], relation_id] = 1.0
    return targets


def _restrict_pairs_to_source_documents(
    pairs: torch.Tensor, entities: list[dict]
) -> torch.Tensor:
    """Remove candidate pairs whose entities came from different sources.

    Cross-window super-documents concatenate independent annotated examples.
    Such pairs are neither positive nor reliable negatives, so they must not
    participate in Stage-2 training.  Legacy datasets without provenance keep
    the original all-pairs behavior.
    """
    # Legacy datasets have no provenance key at all and retain the original
    # all-pairs behavior.  Cross-window datasets do carry the key; in that
    # case an unresolved span (None, e.g. crossing a separator) must not be
    # paired with anything because its source is unknown.
    if not entities or not any("source_document_id" in entity for entity in entities):
        return pairs
    source_ids = [entity.get("source_document_id") for entity in entities]
    keep = torch.tensor(
        [
            source_ids[int(head)] is not None
            and source_ids[int(tail)] is not None
            and source_ids[int(head)] == source_ids[int(tail)]
            for head, tail in pairs.tolist()
        ],
        dtype=torch.bool,
        device=pairs.device,
    )
    return pairs[keep]


def _load_prediction_records(predictions_dir: Path, split: str, expected_documents: int) -> list[dict]:
    """Load Stage-1 entities and optional dense local RE scores."""
    path = predictions_dir / f"{split}_predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {path}")
    records: dict[int, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            document_id = int(record["document_id"])
            prediction = record.get("prediction", {})
            records[document_id] = {
                "entities": prediction.get("entities", []),
                "local_relation_scores": record.get("local_relation_scores"),
            }
    missing = sorted(set(range(expected_documents)) - set(records))
    if missing:
        raise ValueError(
            f"{path} is missing {len(missing)} document predictions; first ids: {missing[:5]}"
        )
    return [records[idx] for idx in range(expected_documents)]


def _local_relation_tensors(
    pairs: torch.Tensor,
    local_scores: list[dict] | None,
    relation_types: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.zeros((pairs.size(0), len(relation_types)), dtype=torch.float)
    available = torch.zeros_like(probabilities, dtype=torch.bool)
    if local_scores is None:
        return probabilities, available

    pair_to_row = {tuple(pair): row for row, pair in enumerate(pairs.tolist())}
    relation_to_id = {name: idx for idx, name in enumerate(relation_types)}
    for score in local_scores:
        if isinstance(score, dict):
            head, tail = score["head"], score["tail"]
            type_name, raw_probability = score["type"], score["probability"]
        else:
            head, tail, type_name, raw_probability = score
        pair = (int(head), int(tail))
        if pair not in pair_to_row or type_name not in relation_to_id:
            continue
        row = pair_to_row[pair]
        relation_id = relation_to_id[type_name]
        probability = float(raw_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Local relation probability is outside [0,1]: {probability}")
        probabilities[row, relation_id] = max(
            float(probabilities[row, relation_id]), probability
        )
        available[row, relation_id] = True
    return probabilities, available


def _extract_split_features(
    local_model,
    raw_documents: list[dict],
    dataset: CoNLL,
    device: str,
    window_batch_size: int,
    limit: int | None,
    description: str,
    entity_overrides: list[list[dict]] | None = None,
    local_score_overrides: list[list[dict] | None] | None = None,
) -> list[dict[str, torch.Tensor]]:
    documents = raw_documents[:limit] if limit is not None else raw_documents
    symmetric_ids = set(dataset.symmetric_link_types)
    result = []
    for document_id, raw_document in enumerate(tqdm(documents, desc=description)):
        entities = (
            entity_overrides[document_id]
            if entity_overrides is not None
            else raw_document.get("entities", [])
        )
        features = encode_document_features(
            local_model=local_model,
            raw_document=raw_document,
            entities=entities,
            entity_types=dataset.entity_types,
            max_length=dataset.max_length,
            overlap_words=dataset.window_overlap,
            device=device,
            window_batch_size=window_batch_size,
        )
        # Relation pairs must use the same entity set that produced entity_repr.
        # In predicted-entity mode this differs from the gold entity count.
        pairs = DocumentRelationModel.all_ordered_pairs(len(entities), device="cpu")
        pairs = _restrict_pairs_to_source_documents(pairs, entities)
        if pairs.numel() and int(pairs.max().item()) >= len(entities):
            raise RuntimeError(
                f"Pair index exceeds entity count for document {document_id}: "
                f"max_pair_index={int(pairs.max().item())}, entities={len(entities)}"
            )
        features = {
            name: (value.detach().float().cpu() if value.is_floating_point() else value.detach().cpu())
            for name, value in features.items()
        }
        features["pairs"] = pairs
        features["targets"] = _relation_targets(
            raw_document, pairs, dataset.link_types, symmetric_ids, entities=entities
        )
        if local_score_overrides is not None:
            local_probabilities, local_available = _local_relation_tensors(
                pairs, local_score_overrides[document_id], dataset.link_types
            )
            features["local_probabilities"] = local_probabilities
            features["local_available"] = local_available
        result.append(features)
    return result


def _sample_rows(
    targets: torch.Tensor,
    max_unlabeled_pairs: int,
    generator: torch.Generator,
    pairs: torch.Tensor | None = None,
    entity_positions: torch.Tensor | None = None,
    long_distance_boundary: int = 128,
    long_distance_sampling_weight: float = 0.0,
) -> torch.Tensor:
    """Keep all positives and sample informative unlabeled pairs.

    Random negatives dominate the quadratic candidate set in long documents.
    When enabled, sampling reserves probability mass for long-distance pairs,
    which makes the long-distance head see useful negatives without changing
    the PU interpretation of an unlabeled relation.
    """
    positive = targets.any(dim=-1).nonzero(as_tuple=False).flatten()
    unlabeled = (~targets.any(dim=-1)).nonzero(as_tuple=False).flatten()
    if unlabeled.numel() > max_unlabeled_pairs:
        weights = torch.ones(unlabeled.numel(), dtype=torch.float)
        if (
            pairs is not None
            and entity_positions is not None
            and long_distance_sampling_weight > 0.0
        ):
            sampled_pairs = pairs[unlabeled]
            distances = (
                entity_positions[sampled_pairs[:, 0]]
                - entity_positions[sampled_pairs[:, 1]]
            ).abs()
            weights += float(long_distance_sampling_weight) * (
                distances >= long_distance_boundary
            ).to(weights.dtype)
        chosen = torch.multinomial(
            weights, max_unlabeled_pairs, replacement=False, generator=generator
        )
        unlabeled = unlabeled[chosen]
    selected = torch.cat((positive, unlabeled))
    if selected.numel() == 0:
        return selected
    return selected[torch.randperm(selected.numel(), generator=generator)]


@torch.no_grad()
def _evaluate(
    model: DocumentRelationModel,
    features: list[dict[str, torch.Tensor]],
    device: str,
    symmetric_relation_ids: set[int],
    fixed_threshold: float | None = None,
) -> tuple[dict, float]:
    model.eval()
    probability_rows = []
    target_rows = []
    for document in features:
        pairs = document["pairs"].to(device)
        output = model(
            document["entity_repr"].to(device),
            document["entity_type_probs"].to(device),
            document["entity_positions"].to(device),
            document["window_repr"].to(device),
            document["window_ranges"].to(device),
            pairs=pairs,
            local_probabilities=(
                document["local_probabilities"].to(device)
                if model.config.fusion_mode == "learned"
                else None
            ),
            local_available=(
                document["local_available"].to(device)
                if model.config.fusion_mode == "learned"
                else None
            ),
        )
        probabilities = (
            output.fused_probabilities
            if output.fused_probabilities is not None
            else output.probabilities
        )
        if output.pair_link_logits is not None:
            # Match inference-time pair-level suppression so checkpoint and
            # threshold selection reflect the actual IND decision rule.
            probabilities = model.apply_pair_link_gate(
                probabilities, output.pair_link_logits
            )
        probabilities = probabilities.cpu()
        targets = document["targets"].bool()
        valid = torch.ones_like(targets, dtype=torch.bool)
        for relation_id in symmetric_relation_ids:
            valid[:, relation_id] = document["pairs"][:, 0] < document["pairs"][:, 1]
        probability_rows.append(probabilities[valid])
        target_rows.append(targets[valid])
    probabilities = torch.cat(probability_rows) if probability_rows else torch.empty(0)
    targets = torch.cat(target_rows).bool() if target_rows else torch.empty(0, dtype=torch.bool)

    best_score = safe_prf(0, 0, int(targets.sum().item()))
    best_threshold = 0.5 if fixed_threshold is None else float(fixed_threshold)
    thresholds = (
        [float(fixed_threshold)]
        if fixed_threshold is not None
        else torch.arange(0.10, 0.901, 0.02).tolist()
    )
    for threshold in thresholds:
        predictions = probabilities >= threshold
        tp = int((predictions & targets).sum().item())
        fp = int((predictions & ~targets).sum().item())
        fn = int((~predictions & targets).sum().item())
        score = safe_prf(tp, fp, fn)
        if score["f1"] > best_score["f1"]:
            best_score = score
            best_threshold = float(threshold)
    return best_score, best_threshold


def train(args: argparse.Namespace) -> Path:
    _set_seed(args.seed)
    normalized_focus_pairs = []
    for pair in args.pair_type_focus_pairs:
        if isinstance(pair, str):
            parts = pair.split(":", 1)
        else:
            parts = list(pair)
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                "pair_type_focus_pairs must contain TYPE_A:TYPE_B or [TYPE_A, TYPE_B] entries"
            )
        normalized_focus_pairs.append([str(parts[0]), str(parts[1])])
    args.pair_type_focus_pairs = normalized_focus_pairs
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Local ITER checkpoint does not exist: {args.checkpoint}")
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    load_kwargs = {"local_files_only": True} if args.offline else {}
    local_config = ITERConfig.from_pretrained(args.checkpoint, **load_kwargs)
    local_class = ITER if local_config.is_feature_ner_only else ITERForRelationExtraction
    local_model = local_class.from_pretrained(args.checkpoint, **load_kwargs).to(device).eval()
    for parameter in local_model.parameters():
        parameter.requires_grad_(False)

    dataset = CoNLL.from_name(
        args.config,
        tokenizer=local_model.tokenizer,
        tokenizer_config=local_model.config.transformer_config,
    )
    if not dataset.enable_windowing:
        raise ValueError("Document relation training requires enable_windowing=true")
    if local_config.entity_types and local_config.entity_types != dataset.entity_types:
        raise ValueError("Checkpoint entity types do not match dataset config")
    if local_config.link_types and local_config.link_types != dataset.link_types:
        raise ValueError("Checkpoint relation types do not match dataset config")

    raw_train = dataset.open_split("train", dataset.splits["train"])
    raw_eval = dataset.open_split("eval", dataset.splits["eval"])
    stats_documents = raw_train[:args.max_train_documents] if args.max_train_documents else raw_train
    compatibility, class_priors = compute_relation_statistics(
        stats_documents,
        dataset.entity_types,
        dataset.link_types,
        set(dataset.symmetric_link_types),
    )

    train_entity_overrides = None
    eval_entity_overrides = None
    train_local_score_overrides = None
    eval_local_score_overrides = None
    if args.entity_source == "predicted":
        if args.predictions_dir is None:
            raise ValueError("--predictions_dir is required when --entity_source=predicted")
        train_prediction_records = _load_prediction_records(
            args.predictions_dir, "train", len(raw_train)
        )
        eval_prediction_records = _load_prediction_records(
            args.predictions_dir, "eval", len(raw_eval)
        )
        train_entity_overrides = [record["entities"] for record in train_prediction_records]
        eval_entity_overrides = [record["entities"] for record in eval_prediction_records]
        train_local_score_overrides = [
            record["local_relation_scores"] for record in train_prediction_records
        ]
        eval_local_score_overrides = [
            record["local_relation_scores"] for record in eval_prediction_records
        ]
        if args.fusion_mode == "learned":
            missing_train = sum(scores is None for scores in train_local_score_overrides)
            missing_eval = sum(scores is None for scores in eval_local_score_overrides)
            if missing_train or missing_eval:
                raise ValueError(
                    "learned fusion requires local_relation_scores exported for every prediction "
                    f"record (missing train={missing_train}, eval={missing_eval})"
                )
        print(f"Using predicted entities from {args.predictions_dir.resolve()}")
    elif args.fusion_mode == "learned":
        raise ValueError("learned fusion requires --entity_source predicted")

    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.bfloat16 and device.startswith("cuda")
        else contextlib.nullcontext()
    )
    with amp_context:
        train_features = _extract_split_features(
            local_model,
            raw_train,
            dataset,
            device,
            args.window_batch_size,
            args.max_train_documents,
            "Encoding train documents",
            entity_overrides=train_entity_overrides,
            local_score_overrides=train_local_score_overrides,
        )
        eval_features = _extract_split_features(
            local_model,
            raw_eval,
            dataset,
            device,
            args.window_batch_size,
            args.max_eval_documents,
            "Encoding eval documents",
            entity_overrides=eval_entity_overrides,
            local_score_overrides=eval_local_score_overrides,
        )
    del local_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    output_dir = args.output_dir or args.checkpoint / "document_relation"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = DocumentRelationConfig(
        hidden_size=local_config.d_model,
        entity_types=dataset.entity_types,
        relation_types=dataset.link_types,
        dropout=args.dropout,
        type_prior_weight=args.type_prior_weight,
        evidence_top_k=args.evidence_top_k,
        fusion_mode=args.fusion_mode,
        fusion_hidden_size=args.fusion_hidden_size,
        fusion_relation_embedding_size=args.fusion_relation_embedding_size,
        distance_conditioned_fusion=args.distance_conditioned_fusion,
        fusion_long_distance_boundary=args.fusion_long_distance_boundary,
        fusion_distance_hidden_size=args.fusion_distance_hidden_size,
        fusion_long_distance_bias=args.fusion_long_distance_bias,
        long_distance_head=args.long_distance_head,
        long_distance_boundary=args.long_distance_boundary,
        long_distance_head_weight=args.long_distance_head_weight,
        distance_experts=args.distance_experts,
        distance_expert_boundaries=args.distance_expert_boundaries,
        distance_expert_temperature=args.distance_expert_temperature,
        distance_expert_weight=args.distance_expert_weight,
        distance_expert_active_buckets=args.distance_expert_active_buckets,
        pair_type_compatibility=args.pair_type_compatibility,
        pair_type_compatibility_bias_weight=args.pair_type_compatibility_bias_weight,
        pair_type_focus_pairs=args.pair_type_focus_pairs,
        pair_type_focus_weight=args.pair_type_focus_weight,
        use_direction_embedding=args.use_direction_embedding,
        use_window_position_encoding=args.use_window_position_encoding,
        positive_focal_loss_weight=args.positive_focal_loss_weight,
        positive_focal_gamma=args.positive_focal_gamma,
        long_distance_sampling_weight=args.long_distance_sampling_weight,
        focus_relation_names=args.focus_relation_names,
        focus_relation_positive_weight=args.focus_relation_positive_weight,
        relation_adapter=args.relation_adapter,
        relation_adapter_size=args.relation_adapter_size,
        relation_adapter_weight=args.relation_adapter_weight,
        recall_head=args.recall_head,
        recall_head_size=args.recall_head_size,
        recall_head_weight=args.recall_head_weight,
        pair_link_head=args.pair_link_head,
        pair_link_bias=args.pair_link_bias,
        pair_link_gate_weight=args.pair_link_gate_weight,
        pair_link_positive_weight=args.pair_link_positive_weight,
        local_checkpoint=str(args.checkpoint.resolve()),
        dataset_config=args.config,
    )
    model = DocumentRelationModel(config).to(device)
    model.set_statistics(compatibility.to(device), class_priors.to(device))
    if args.load_document_checkpoint is not None:
        checkpoint_path = args.load_document_checkpoint
        if not (checkpoint_path / "document_relation.safetensors").exists():
            raise FileNotFoundError(
                f"Document relation checkpoint does not exist: {checkpoint_path}"
            )
        previous = DocumentRelationModel.from_pretrained(checkpoint_path)
        previous_state = previous.state_dict()
        current_state = model.state_dict()
        derived_buffers = {
            "compatibility_bias",
            "class_priors",
            "pair_type_focus_matrix",
            "relation_focus_weights",
        }
        compatible = {
            name: value
            for name, value in previous_state.items()
            if name in current_state
            and current_state[name].shape == value.shape
            and name not in derived_buffers
        }
        required = {
            name
            for name, _ in model.named_parameters()
            if not name.startswith("distance_expert_")
            and not name.startswith("relation_adapter_")
            and not name.startswith("recall_head.")
        }
        absent_required = required - set(compatible)
        if absent_required:
            raise RuntimeError(
                "Continuation checkpoint is missing required parameters: "
                f"{sorted(absent_required)}"
            )
        model.load_state_dict(compatible, strict=False)
        del previous
        print(
            f"Initialized {len(compatible)} document relation tensors from "
            f"{checkpoint_path.resolve()}"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)

    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    indices = list(range(len(train_features)))
    for epoch in range(1, args.epochs + 1):
        random.shuffle(indices)
        model.train()
        epoch_loss = 0.0
        epoch_pu_loss = 0.0
        epoch_evidence_loss = 0.0
        epoch_distance_loss = 0.0
        epoch_long_distance_ranking_loss = 0.0
        epoch_pair_type_compatibility_loss = 0.0
        epoch_positive_focal_loss = 0.0
        epoch_recall_margin_loss = 0.0
        epoch_pair_link_loss = 0.0
        epoch_distance_expert_loss = 0.0
        epoch_steps = 0
        for batch_start in tqdm(
            range(0, len(indices), args.documents_per_batch),
            desc=f"Document RE epoch {epoch}/{args.epochs}",
        ):
            optimizer.zero_grad(set_to_none=True)
            logits_rows = []
            global_logits_rows = []
            target_rows = []
            evidence_losses = []
            distance_losses = []
            type_balance_losses = []
            long_distance_ranking_losses = []
            pair_type_compatibility_losses = []
            positive_focal_losses = []
            recall_margin_losses = []
            pair_link_losses = []
            distance_expert_losses = []
            for document_id in indices[batch_start:batch_start + args.documents_per_batch]:
                document = train_features[document_id]
                selected = _sample_rows(
                    document["targets"],
                    args.max_unlabeled_pairs,
                    generator,
                    pairs=document["pairs"],
                    entity_positions=document["entity_positions"],
                    long_distance_boundary=args.long_distance_boundary,
                    long_distance_sampling_weight=args.long_distance_sampling_weight,
                )
                if selected.numel() == 0:
                    continue
                pairs = document["pairs"][selected].to(device)
                targets = document["targets"][selected].to(device)
                entity_type_probs = document["entity_type_probs"].to(device)
                output = model(
                    document["entity_repr"].to(device),
                    entity_type_probs,
                    document["entity_positions"].to(device),
                    document["window_repr"].to(device),
                    document["window_ranges"].to(device),
                    pairs=pairs,
                    local_probabilities=(
                        document["local_probabilities"][selected].to(device)
                        if model.config.fusion_mode == "learned"
                        else None
                    ),
                    local_available=(
                        document["local_available"][selected].to(device)
                        if model.config.fusion_mode == "learned"
                        else None
                    ),
                )
                decision_logits = (
                    output.fused_logits if output.fused_logits is not None else output.logits
                )
                logits_rows.append(decision_logits)
                if output.fused_logits is not None:
                    global_logits_rows.append(output.logits)
                target_rows.append(targets)
                if model.config.pair_link_head:
                    if output.pair_link_logits is None:
                        raise RuntimeError("pair-link head did not return logits")
                    link_targets = targets.any(dim=-1).to(output.pair_link_logits.dtype)
                    # IND is treated as a closed-world relation annotation for
                    # this optional gate: sampled unlabeled pairs are hard
                    # negatives, while every annotated pair is positive.
                    pair_link_losses.append(
                        F.binary_cross_entropy_with_logits(
                            output.pair_link_logits,
                            link_targets,
                            pos_weight=output.pair_link_logits.new_tensor(
                                args.pair_link_positive_weight
                            ),
                        )
                    )
                silver = silver_evidence_distribution(
                    pairs,
                    document["entity_positions"].to(device),
                    document["window_ranges"].to(device),
                )
                evidence_losses.append(
                    evidence_kl_loss(output.evidence_attention, silver, targets.any(dim=-1))
                )
                distance_losses.append(
                    distance_weighted_positive_loss(
                        decision_logits,
                        targets,
                        document["entity_positions"].to(device),
                        pairs,
                        boundaries=args.distance_loss_boundaries,
                        weights=args.distance_loss_weights,
                    )
                )
                type_balance_losses.append(
                    relation_balanced_positive_loss(
                        decision_logits,
                        targets,
                        model.class_priors,
                        max_weight=args.type_balance_max_weight,
                    )
                )
                if args.positive_focal_loss_weight > 0.0:
                    positive_focal_losses.append(
                        relation_positive_focal_loss(
                            decision_logits,
                            targets,
                            document["entity_positions"].to(device),
                            pairs,
                            long_distance_boundary=args.long_distance_boundary,
                            long_distance_weight=max(1.0, args.long_distance_sampling_weight),
                            gamma=args.positive_focal_gamma,
                            relation_weights=model.relation_focus_weights,
                        )
                    )
                if args.recall_margin_loss_weight > 0.0:
                    recall_margin_losses.append(
                        relation_recall_margin_loss(
                            decision_logits,
                            targets,
                            document["entity_positions"].to(device),
                            pairs,
                            threshold=args.recall_threshold,
                            margin=args.recall_margin,
                            short_distance_boundary=args.recall_short_distance_boundary,
                            short_distance_weight=args.recall_short_distance_weight,
                            relation_weights=model.relation_focus_weights,
                        )
                    )
                if model.config.distance_experts:
                    if output.distance_expert_logits is None:
                        raise RuntimeError("distance experts did not return logits")
                    distance_expert_losses.append(
                        distance_expert_pu_loss(
                            output.distance_expert_logits,
                            targets,
                            document["entity_positions"].to(device),
                            pairs,
                            args.distance_expert_boundaries,
                            model.class_priors,
                            args.distance_expert_active_buckets,
                        )
                    )
                if model.config.long_distance_head:
                    if output.long_distance_logits is None:
                        raise RuntimeError("long-distance head did not return logits")
                    long_distance_ranking_losses.append(
                        long_distance_pair_ranking_loss(
                            output.long_distance_logits,
                            targets,
                            document["entity_positions"].to(device),
                            pairs,
                            boundary=args.long_distance_boundary,
                            margin=args.long_distance_ranking_margin,
                            max_negatives=args.long_distance_ranking_max_negatives,
                        )
                    )
                if model.config.pair_type_compatibility:
                    if output.pair_type_logits is None:
                        raise RuntimeError("pair-type compatibility did not return logits")
                    pair_type_compatibility_losses.append(
                        pair_type_compatibility_loss(
                            output.pair_type_logits,
                            targets,
                            entity_type_probs[pairs[:, 0]],
                            entity_type_probs[pairs[:, 1]],
                            model.pair_type_focus_matrix,
                            model.class_priors,
                        )
                    )
            if not logits_rows:
                continue
            logits = torch.cat(logits_rows)
            targets = torch.cat(target_rows)
            pu_loss = non_negative_pu_loss(logits, targets, model.class_priors)
            if global_logits_rows:
                global_logits = torch.cat(global_logits_rows)
                pu_loss = pu_loss + args.fusion_global_loss_weight * non_negative_pu_loss(
                    global_logits, targets, model.class_priors
                )
            evidence_loss = torch.stack(evidence_losses).mean()
            distance_loss = torch.stack(distance_losses).mean()
            type_balance_loss = torch.stack(type_balance_losses).mean()
            long_distance_ranking_loss_value = (
                torch.stack(long_distance_ranking_losses).mean()
                if long_distance_ranking_losses
                else logits.sum() * 0.0
            )
            pair_type_compatibility_loss_value = (
                torch.stack(pair_type_compatibility_losses).mean()
                if pair_type_compatibility_losses
                else logits.sum() * 0.0
            )
            positive_focal_loss_value = (
                torch.stack(positive_focal_losses).mean()
                if positive_focal_losses
                else logits.sum() * 0.0
            )
            recall_margin_loss_value = (
                torch.stack(recall_margin_losses).mean()
                if recall_margin_losses
                else logits.sum() * 0.0
            )
            pair_link_loss_value = (
                torch.stack(pair_link_losses).mean()
                if pair_link_losses
                else logits.sum() * 0.0
            )
            distance_expert_loss_value = (
                torch.stack(distance_expert_losses).mean()
                if distance_expert_losses
                else logits.sum() * 0.0
            )
            loss = (
                pu_loss
                + args.evidence_loss_weight * evidence_loss
                + args.distance_positive_loss_weight * distance_loss
                + args.type_balance_positive_loss_weight * type_balance_loss
                + args.long_distance_ranking_loss_weight * long_distance_ranking_loss_value
                + args.pair_type_compatibility_loss_weight * pair_type_compatibility_loss_value
                + args.positive_focal_loss_weight * positive_focal_loss_value
                + args.recall_margin_loss_weight * recall_margin_loss_value
                + args.distance_expert_loss_weight * distance_expert_loss_value
                + args.pair_link_loss_weight * pair_link_loss_value
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            epoch_pu_loss += float(pu_loss.detach().cpu())
            epoch_evidence_loss += float(evidence_loss.detach().cpu())
            epoch_distance_loss += float(distance_loss.detach().cpu())
            epoch_long_distance_ranking_loss += float(
                long_distance_ranking_loss_value.detach().cpu()
            )
            epoch_pair_type_compatibility_loss += float(
                pair_type_compatibility_loss_value.detach().cpu()
            )
            epoch_positive_focal_loss += float(positive_focal_loss_value.detach().cpu())
            epoch_recall_margin_loss += float(recall_margin_loss_value.detach().cpu())
            epoch_distance_expert_loss += float(distance_expert_loss_value.detach().cpu())
            epoch_pair_link_loss += float(pair_link_loss_value.detach().cpu())
            epoch_steps += 1

        eval_score, threshold = _evaluate(
            model,
            eval_features,
            device,
            set(dataset.symmetric_link_types),
            fixed_threshold=args.fixed_eval_threshold,
        )
        mean_loss = epoch_loss / max(epoch_steps, 1)
        history.append({
            "epoch": epoch,
            "train_loss": mean_loss,
            "train_pu_loss": epoch_pu_loss / max(epoch_steps, 1),
            "train_evidence_loss": epoch_evidence_loss / max(epoch_steps, 1),
            "train_distance_positive_loss": epoch_distance_loss / max(epoch_steps, 1),
            "train_long_distance_ranking_loss": epoch_long_distance_ranking_loss
            / max(epoch_steps, 1),
            "train_pair_type_compatibility_loss": epoch_pair_type_compatibility_loss
            / max(epoch_steps, 1),
            "train_positive_focal_loss": epoch_positive_focal_loss / max(epoch_steps, 1),
            "train_recall_margin_loss": epoch_recall_margin_loss / max(epoch_steps, 1),
            "train_distance_expert_loss": epoch_distance_expert_loss / max(epoch_steps, 1),
            "train_pair_link_loss": epoch_pair_link_loss / max(epoch_steps, 1),
            "threshold": threshold,
            **eval_score,
        })
        print(json.dumps(history[-1], ensure_ascii=False))
        if eval_score["f1"] > best_f1:
            best_f1 = eval_score["f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            model.config.threshold = threshold
            model.save_pretrained(output_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    summary = {
        "local_checkpoint": str(args.checkpoint.resolve()),
        "document_relation_checkpoint": str(output_dir.resolve()),
        "config": args.config,
        "device": device,
        "seed": args.seed,
        "entity_source": args.entity_source,
        "fusion_mode": args.fusion_mode,
        "fusion_global_loss_weight": args.fusion_global_loss_weight,
        "distance_positive_loss_weight": args.distance_positive_loss_weight,
        "distance_loss_boundaries": args.distance_loss_boundaries,
        "distance_loss_weights": args.distance_loss_weights,
        "distance_conditioned_fusion": args.distance_conditioned_fusion,
        "fusion_long_distance_boundary": args.fusion_long_distance_boundary,
        "fusion_distance_hidden_size": args.fusion_distance_hidden_size,
        "fusion_long_distance_bias": args.fusion_long_distance_bias,
        "long_distance_head": args.long_distance_head,
        "long_distance_boundary": args.long_distance_boundary,
        "long_distance_head_weight": args.long_distance_head_weight,
        "distance_experts": args.distance_experts,
        "distance_expert_boundaries": args.distance_expert_boundaries,
        "distance_expert_temperature": args.distance_expert_temperature,
        "distance_expert_weight": args.distance_expert_weight,
        "distance_expert_loss_weight": args.distance_expert_loss_weight,
        "pair_link_head": args.pair_link_head,
        "pair_link_loss_weight": args.pair_link_loss_weight,
        "pair_link_bias": args.pair_link_bias,
        "pair_link_gate_weight": args.pair_link_gate_weight,
        "pair_link_positive_weight": args.pair_link_positive_weight,
        "distance_expert_active_buckets": args.distance_expert_active_buckets,
        "long_distance_ranking_loss_weight": args.long_distance_ranking_loss_weight,
        "long_distance_ranking_margin": args.long_distance_ranking_margin,
        "long_distance_ranking_max_negatives": args.long_distance_ranking_max_negatives,
        "pair_type_compatibility": args.pair_type_compatibility,
        "pair_type_compatibility_loss_weight": args.pair_type_compatibility_loss_weight,
        "pair_type_compatibility_bias_weight": args.pair_type_compatibility_bias_weight,
        "pair_type_focus_pairs": args.pair_type_focus_pairs,
        "pair_type_focus_weight": args.pair_type_focus_weight,
        "use_direction_embedding": args.use_direction_embedding,
        "use_window_position_encoding": args.use_window_position_encoding,
        "positive_focal_loss_weight": args.positive_focal_loss_weight,
        "positive_focal_gamma": args.positive_focal_gamma,
        "long_distance_sampling_weight": args.long_distance_sampling_weight,
        "focus_relation_names": args.focus_relation_names,
        "focus_relation_positive_weight": args.focus_relation_positive_weight,
        "relation_adapter": args.relation_adapter,
        "relation_adapter_size": args.relation_adapter_size,
        "relation_adapter_weight": args.relation_adapter_weight,
        "recall_head": args.recall_head,
        "recall_head_size": args.recall_head_size,
        "recall_head_weight": args.recall_head_weight,
        "recall_margin_loss_weight": args.recall_margin_loss_weight,
        "recall_threshold": args.recall_threshold,
        "recall_margin": args.recall_margin,
        "recall_short_distance_boundary": args.recall_short_distance_boundary,
        "recall_short_distance_weight": args.recall_short_distance_weight,
        "fixed_eval_threshold": args.fixed_eval_threshold,
        "load_document_checkpoint": (
            str(args.load_document_checkpoint.resolve())
            if args.load_document_checkpoint is not None
            else None
        ),
        "type_balance_positive_loss_weight": args.type_balance_positive_loss_weight,
        "type_balance_max_weight": args.type_balance_max_weight,
        "predictions_dir": str(args.predictions_dir.resolve()) if args.predictions_dir else None,
        "train_documents": len(train_features),
        "eval_documents": len(eval_features),
        "best_epoch": best_epoch,
        "best_eval_f1": best_f1,
        "class_priors": {
            name: float(class_priors[idx]) for idx, name in enumerate(dataset.link_types)
        },
        "history": history,
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Best document relation checkpoint: {output_dir.resolve()}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training_config",
        type=Path,
        default=None,
        help="Optional JSON file providing defaults for document RE training arguments",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained local ITER checkpoint")
    parser.add_argument("--config", default="wenyanwen/wenyanwen_long_context_512")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--documents_per_batch", type=int, default=8)
    parser.add_argument("--window_batch_size", type=int, default=8)
    parser.add_argument("--max_unlabeled_pairs", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--type_prior_weight", type=float, default=0.5)
    parser.add_argument("--evidence_loss_weight", type=float, default=0.2)
    parser.add_argument("--evidence_top_k", type=int, default=2)
    parser.add_argument("--fusion_mode", choices=("fixed", "learned"), default="fixed")
    parser.add_argument("--fusion_hidden_size", type=int, default=32)
    parser.add_argument("--fusion_relation_embedding_size", type=int, default=8)
    parser.add_argument("--fusion_global_loss_weight", type=float, default=0.25)
    parser.add_argument("--distance_conditioned_fusion", action="store_true")
    parser.add_argument("--fusion_long_distance_boundary", type=int, default=128)
    parser.add_argument("--fusion_distance_hidden_size", type=int, default=16)
    parser.add_argument("--fusion_long_distance_bias", type=float, default=0.5)
    parser.add_argument("--long_distance_head", action="store_true")
    parser.add_argument("--long_distance_boundary", type=int, default=128)
    parser.add_argument("--long_distance_head_weight", type=float, default=0.5)
    parser.add_argument("--distance_experts", action="store_true")
    parser.add_argument(
        "--distance_expert_boundaries", nargs="+", type=int, default=[32, 64, 128]
    )
    parser.add_argument("--distance_expert_temperature", type=float, default=0.75)
    parser.add_argument("--distance_expert_weight", type=float, default=0.5)
    parser.add_argument("--distance_expert_loss_weight", type=float, default=0.15)
    parser.add_argument("--distance_expert_active_buckets", nargs="*", type=int, default=[])
    parser.add_argument("--long_distance_ranking_loss_weight", type=float, default=0.0)
    parser.add_argument("--long_distance_ranking_margin", type=float, default=0.2)
    parser.add_argument("--long_distance_ranking_max_negatives", type=int, default=64)
    parser.add_argument("--pair_type_compatibility", action="store_true")
    parser.add_argument("--pair_type_compatibility_loss_weight", type=float, default=0.0)
    parser.add_argument("--pair_type_compatibility_bias_weight", type=float, default=0.25)
    parser.add_argument(
        "--pair_type_focus_pairs",
        nargs="+",
        default=[["学校", "园林"], ["行政区划", "水系"], ["城市内部区划", "街道"]],
    )
    parser.add_argument("--pair_type_focus_weight", type=float, default=2.0)
    parser.add_argument("--use_direction_embedding", action="store_true")
    parser.add_argument("--use_window_position_encoding", action="store_true")
    parser.add_argument("--positive_focal_loss_weight", type=float, default=0.0)
    parser.add_argument("--positive_focal_gamma", type=float, default=1.5)
    parser.add_argument("--long_distance_sampling_weight", type=float, default=0.0)
    parser.add_argument("--focus_relation_names", nargs="+", default=[])
    parser.add_argument("--focus_relation_positive_weight", type=float, default=1.0)
    parser.add_argument("--relation_adapter", action="store_true")
    parser.add_argument("--relation_adapter_size", type=int, default=128)
    parser.add_argument("--relation_adapter_weight", type=float, default=0.25)
    parser.add_argument("--recall_head", action="store_true")
    parser.add_argument("--recall_head_size", type=int, default=128)
    parser.add_argument("--recall_head_weight", type=float, default=0.25)
    parser.add_argument("--recall_margin_loss_weight", type=float, default=0.0)
    parser.add_argument("--recall_threshold", type=float, default=0.24)
    parser.add_argument("--recall_margin", type=float, default=0.35)
    parser.add_argument("--recall_short_distance_boundary", type=int, default=64)
    parser.add_argument("--recall_short_distance_weight", type=float, default=1.5)
    parser.add_argument("--pair_link_head", action="store_true",
                        help="Add an optional pair-existence gate (use for IND-only RE experiments).")
    parser.add_argument("--pair_link_loss_weight", type=float, default=0.0)
    parser.add_argument("--pair_link_bias", type=float, default=0.0)
    parser.add_argument("--pair_link_gate_weight", type=float, default=1.0)
    parser.add_argument("--pair_link_positive_weight", type=float, default=1.0)
    parser.add_argument(
        "--fixed_eval_threshold",
        type=float,
        default=None,
        help="Evaluate and select checkpoints at one fixed RE threshold; disables threshold search.",
    )
    parser.add_argument("--load_document_checkpoint", type=Path, default=None)
    parser.add_argument("--distance_positive_loss_weight", type=float, default=0.0)
    parser.add_argument(
        "--distance_loss_boundaries", nargs="+", type=int, default=[32, 64, 128]
    )
    parser.add_argument(
        "--distance_loss_weights", nargs="+", type=float, default=[1.0, 1.25, 1.5, 2.0]
    )
    parser.add_argument("--type_balance_positive_loss_weight", type=float, default=0.0)
    parser.add_argument("--type_balance_max_weight", type=float, default=4.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--entity_source",
        choices=("gold", "predicted"),
        default="gold",
        help="Entities used to train Stage 2; predicted requires train/eval JSONL files.",
    )
    parser.add_argument(
        "--predictions_dir",
        type=Path,
        default=None,
        help="Directory containing train_predictions.jsonl and eval_predictions.jsonl.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bfloat16", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max_train_documents", type=int, default=None, help="Debug-only limit")
    parser.add_argument("--max_eval_documents", type=int, default=None, help="Debug-only limit")
    return parser


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--training_config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    parser = build_parser()
    if pre_args.training_config is not None:
        with pre_args.training_config.open(encoding="utf-8") as handle:
            defaults = json.load(handle)
        unknown = set(defaults) - {action.dest for action in parser._actions}
        if unknown:
            parser.error(f"Unknown keys in training config: {sorted(unknown)}")
        parser.set_defaults(**defaults)
    parsed = parser.parse_args()
    if parsed.fixed_eval_threshold is not None and not 0.0 <= parsed.fixed_eval_threshold <= 1.0:
        parser.error("--fixed_eval_threshold must be between 0 and 1")
    train(parsed)


if __name__ == "__main__":
    main()
