#!/usr/bin/env python3
"""Windowed document inference, global-offset merging, and long-context evaluation."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import contextlib
import csv
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from iter import ITER, ITERConfig
from iter.datasets import CoNLL
from iter.datasets.training import Hparams
from iter.misc.document_metrics import entity_key, evaluate_documents, relation_key
from iter.modeling_iter2 import ITERForRelationExtraction
from iter.modules.document_relation import DocumentRelationModel, encode_document_features
from iter.tools.dataloaders import create_primary_dataloader


PROJECT_ROOT = Path(__file__).resolve().parent


def _synchronize_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _parameter_counts(model) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def _directory_size_bytes(directory: Path, *, recursive: bool) -> int:
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    return sum(path.stat().st_size for path in iterator if path.is_file())


def _summarize_benchmark_runs(runs: list[dict]) -> dict:
    if not runs:
        return {}
    fields = (
        "stage1_seconds",
        "stage2_seconds",
        "pure_inference_seconds",
        "milliseconds_per_document",
        "documents_per_second",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
    )
    summary = {}
    for field in fields:
        values = [float(run[field]) for run in runs]
        summary[field] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def _jsonable(value):
    """Convert tuple-heavy inference evidence into JSON-safe primitives."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _decode_window(
    model,
    generation,
    batch_index: int,
    raw_document: dict,
    window: dict,
    *,
    ner_span_threshold: float = 0.5,
    ner_type_threshold: float = 0.0,
    include_relations: bool = True,
) -> tuple[list, list, list]:
    """Decode model tensors to global word offsets for one non-nested window."""
    if model.is_feature_nest_depth_gt_1:
        raise NotImplementedError("Global-offset inference currently supports entity_nest_depth=1")

    start = window["word_start"]
    end = window["word_end"]
    tokens = raw_document["tokens"][start:end]
    encoding = model.tokenizer(tokens, is_split_into_words=True, add_special_tokens=True)
    word_ids = encoding.word_ids()
    actions = generation.actions[batch_index]
    left_positions = actions.bitwise_and(1).ne(0).nonzero(as_tuple=False).flatten().tolist()
    right_positions = actions.bitwise_and(2).ne(0).nonzero(as_tuple=False).flatten().tolist()

    # Prefer continuous model posteriors so dev calibration can change the
    # candidate threshold without rerunning the encoder.  The fallback keeps
    # compatibility with checkpoints/inference objects that do not expose the
    # new confidence tensors.
    candidates = []
    pair_probabilities = getattr(generation, "lr_pair_probabilities", None)
    span_probabilities = getattr(generation, "lr_span_probabilities", None)
    if pair_probabilities is not None and span_probabilities is not None:
        span_values = span_probabilities[batch_index]
        pair_values = pair_probabilities[batch_index]
        valid = (span_values >= ner_span_threshold) & (span_values > 0)
        for right_token, left_index in valid.nonzero(as_tuple=False).tolist():
            type_values = pair_values[right_token, left_index]
            type_id = int(type_values.argmax().item())
            type_confidence = float(type_values[type_id].item())
            if type_values.numel() > 1:
                top2 = torch.topk(type_values, k=2).values
                type_margin = float((top2[0] - top2[1]).item())
            else:
                type_margin = type_confidence
            if type_confidence < ner_type_threshold:
                continue
            candidates.append(
                (
                    right_token,
                    left_index,
                    type_id,
                float(span_values[right_token, left_index].item()),
                type_confidence,
                type_margin,
            )
            )
    else:
        candidates = [
            (right_token, left_index, type_id, None, None, None)
            for right_token, left_index, type_id in generation.lr_pair_flag[batch_index]
            .nonzero(as_tuple=False)
            .tolist()
        ]

    entities = []
    entities_by_right = {}
    for right_token, left_index, type_id, span_confidence, type_confidence, type_margin in candidates:
        if left_index >= len(left_positions) or type_id >= len(model.config.entity_types):
            continue
        left_token = left_positions[left_index]
        if left_token >= len(word_ids) or right_token >= len(word_ids):
            continue
        local_start, local_end = word_ids[left_token], word_ids[right_token]
        if local_start is None or local_end is None:
            continue
        global_start, global_end = start + local_start, start + local_end + 1
        key = (global_start, global_end, model.config.entity_types[type_id])
        centrality = min(global_start - start, end - global_end)
        entity = {
            "key": key,
            "start": global_start,
            "end": global_end,
            "type": model.config.entity_types[type_id],
            "text": "".join(raw_document["tokens"][global_start:global_end]),
            "window_id": window["window_id"],
            "centrality": centrality,
            "span_confidence": span_confidence,
            "type_confidence": type_confidence,
            "type_margin": type_margin,
            "confidence": (
                span_confidence * type_confidence
                if span_confidence is not None and type_confidence is not None
                else None
            ),
        }
        entities.append(entity)
        entities_by_right[right_token] = entity

    if not include_relations:
        return entities, [], []

    relations = []
    relation_scores = []
    rr_flags = generation.rr_pair_flag
    if rr_flags is not None:
        probabilities = generation.rr_pair_probabilities
        for right_token, other_right_index, type_id in rr_flags[batch_index].nonzero(as_tuple=False).tolist():
            if other_right_index >= len(right_positions) or type_id >= len(model.config.link_types):
                continue
            head = entities_by_right.get(right_token)
            tail = entities_by_right.get(right_positions[other_right_index])
            if head is None or tail is None:
                continue
            probability = None
            if probabilities is not None:
                probability = float(probabilities[batch_index, right_token, other_right_index, type_id].item())
            relations.append({
                "head_key": head["key"],
                "tail_key": tail["key"],
                "type": model.config.link_types[type_id],
                "window_id": window["window_id"],
                "probability": probability,
            })
        if probabilities is not None:
            for head_right, head in entities_by_right.items():
                for tail_index, tail_right in enumerate(right_positions):
                    tail = entities_by_right.get(tail_right)
                    if tail is None or head["key"] == tail["key"]:
                        continue
                    for type_id, type_name in enumerate(model.config.link_types):
                        relation_scores.append({
                            "head_key": head["key"],
                            "tail_key": tail["key"],
                            "type": type_name,
                            "window_id": window["window_id"],
                            "probability": float(
                                probabilities[batch_index, head_right, tail_index, type_id].item()
                            ),
                        })
    return entities, relations, relation_scores


def _entity_overlap(left: dict, right: dict) -> bool:
    """Return whether two half-open entity spans share at least one token."""
    return left["start"] < right["end"] and right["start"] < left["end"]


def _entity_selection_score(
    support: list[dict],
    *,
    joint_rerank_weight: float = 0.0,
    type_margin_weight: float = 0.0,
    boundary_rerank_weight: float = 0.0,
    type_rerank_weight: float = 0.0,
) -> tuple[float, float, float, float]:
    """Aggregate per-window confidence while retaining a vote fallback."""
    confidences = [
        float(item["confidence"])
        for item in support
        if item.get("confidence") is not None
    ]
    type_confidences = [
        float(item["type_confidence"])
        for item in support
        if item.get("type_confidence") is not None
    ]
    centrality = max(float(item.get("centrality", 0.0)) for item in support)
    if confidences:
        # A high-confidence central vote dominates, while repeated windows
        # provide a small stabilizing bonus rather than overwhelming it.
        mean_confidence = sum(confidences) / len(confidences)
        selection_score = 0.7 * max(confidences) + 0.3 * mean_confidence
        selection_score += min(0.05, 0.01 * max(0, len(support) - 1))
        type_confidence = sum(type_confidences) / len(type_confidences) if type_confidences else 0.0
        span_confidences = [
            float(item["span_confidence"])
            for item in support
            if item.get("span_confidence") is not None
        ]
        span_confidence = (
            sum(span_confidences) / len(span_confidences)
            if span_confidences else 0.0
        )
        margins = [
            float(item["type_margin"])
            for item in support
            if item.get("type_margin") is not None
        ]
        type_margin = sum(margins) / len(margins) if margins else type_confidence
        boundary_score = span_confidence
        type_score = 0.7 * type_confidence + 0.3 * type_margin
        # Legacy aliases are retained for compatibility with the first
        # reranking implementation.  The explicit two-stage weights below
        # make boundary and type errors independently controllable.
        if joint_rerank_weight:
            joint = (1.0 - joint_rerank_weight) * span_confidence + joint_rerank_weight * type_confidence
            selection_score = selection_score + joint_rerank_weight * (joint - selection_score)
        if type_margin_weight:
            selection_score += type_margin_weight * type_margin
        total_weight = boundary_rerank_weight + type_rerank_weight
        if total_weight > 0.0:
            legacy_weight = max(0.0, 1.0 - total_weight)
            selection_score = (
                legacy_weight * selection_score
                + boundary_rerank_weight * boundary_score
                + type_rerank_weight * type_score
            ) / max(1.0, legacy_weight + total_weight)
        return min(1.0, selection_score), type_confidence, centrality, type_margin

    # Old checkpoints only expose discrete flags.  Keep their behavior
    # deterministic and let the explicit ``none`` strategy reproduce the old
    # vote-based merger exactly.
    return float(len(support)), 0.0, centrality, 0.0


def _source_document_id(raw: dict, start: int, end: int) -> str | None:
    """Resolve a predicted span to its concatenated source document.

    This uses only the source packing manifest and token offsets, never gold
    relation labels.  Spans crossing a separator are left unassigned and are
    consequently excluded from cross-source relation candidates.
    """
    for source in raw.get("source_documents", []):
        lower = int(source.get("word_start", -1))
        upper = int(source.get("word_end", -1))
        if lower <= start and end <= upper:
            source_id = source.get("source_document_id")
            if source_id is not None:
                return str(source_id)
            if source.get("source_split") is not None and source.get("source_index") is not None:
                return f"{source['source_split']}:{source['source_index']}"
    return None


def _select_flat_entities(
    candidates: list[dict],
    strategy: str,
    *,
    boundary_shift_penalty: float = 0.0,
) -> list[dict]:
    """Apply an optional flat-NER non-overlap decoder to merged candidates."""
    if strategy == "none" or len(candidates) < 2:
        return candidates

    if boundary_shift_penalty:
        # Penalize a candidate when another candidate shares its start or end
        # but has a stronger span score.  This targets boundary-offset errors
        # without using gold labels.
        for candidate in candidates:
            same_endpoint = [
                other for other in candidates
                if other is not candidate and (
                    (other["start"] == candidate["start"] and other["end"] != candidate["end"])
                    or (other["end"] == candidate["end"] and other["start"] != candidate["start"])
                )
            ]
            stronger = max(
                [float(other.get("boundary_score", other.get("span_confidence", 0.0))) for other in same_endpoint],
                default=0.0,
            )
            own = float(candidate.get("boundary_score", candidate.get("span_confidence", 0.0)) or 0.0)
            candidate["boundary_shift_penalty"] = max(0.0, stronger - own) * boundary_shift_penalty
            candidate["selection_score"] = float(candidate.get("selection_score", 0.0)) - candidate["boundary_shift_penalty"]

    if strategy == "nms":
        def numeric(item: dict, field: str) -> float:
            value = item.get(field)
            return float(value) if value is not None else 0.0

        ranked = sorted(
            candidates,
            key=lambda item: (
                -float(item.get("selection_score", 0.0)),
                -numeric(item, "type_confidence"),
                -(item["end"] - item["start"]),
                item["start"],
                item["end"],
            ),
        )
        selected = []
        for candidate in ranked:
            if all(not _entity_overlap(candidate, other) for other in selected):
                selected.append(candidate)
        return sorted(selected, key=lambda item: (item["start"], item["end"], item["type"]))

    if strategy != "weighted_interval":
        raise ValueError(f"Unknown NER overlap strategy: {strategy}")

    # Weighted interval scheduling finds the globally best non-overlapping
    # subset instead of making a local suppression decision.
    ordered = sorted(candidates, key=lambda item: (item["end"], item["start"], item["type"]))
    ends = [item["end"] for item in ordered]
    # ``ends`` is sorted with ``ordered``.  The rightmost end that does not
    # overlap the current interval can therefore be found in O(log n),
    # instead of scanning all earlier candidates (O(n^2) per document).
    predecessor = [
        bisect_right(ends, item["start"], 0, index) - 1
        for index, item in enumerate(ordered)
    ]

    best: list[tuple[float, float, int, tuple[int, ...]]] = [(0.0, 0.0, 0, tuple())]
    for index, item in enumerate(ordered, start=1):
        without = best[index - 1]
        previous = best[predecessor[index - 1] + 1]
        weight = float(item.get("selection_score", 0.0))
        raw_confidence = item.get("confidence")
        confidence = float(raw_confidence) if raw_confidence is not None else weight
        with_item = (
            previous[0] + weight,
            previous[1] + confidence,
            previous[2] + 1,
            previous[3] + (index - 1,),
        )
        # Prefer higher total score, then confidence, then more retained gold
        # candidates when scores tie.
        if with_item[:3] > without[:3]:
            best.append(with_item)
        else:
            best.append(without)
    selected = [ordered[index] for index in best[-1][3]]
    return sorted(selected, key=lambda item: (item["start"], item["end"], item["type"]))


def _merge_document(
    raw: dict,
    window_records: list[dict],
    evidence: dict,
    symmetric_names: set[str],
    *,
    ner_overlap_strategy: str = "none",
    ner_joint_rerank_weight: float = 0.0,
    ner_type_margin_weight: float = 0.0,
    ner_boundary_rerank_weight: float = 0.0,
    ner_type_rerank_weight: float = 0.0,
    ner_boundary_shift_penalty: float = 0.0,
    ner_type_confusion_penalty: float = 0.0,
    include_relations: bool = True,
) -> dict:
    # Resolve type conflicts at identical boundaries using confidence when
    # available, with votes/centrality as a compatibility fallback.
    by_boundary = defaultdict(lambda: defaultdict(list))
    for entity in evidence["entities"]:
        by_boundary[(entity["start"], entity["end"])][entity["type"]].append(entity)

    boundary_candidates = []
    for (start, end), candidates in by_boundary.items():
        scored_candidates = {
            type_name: _entity_selection_score(
                support,
                joint_rerank_weight=ner_joint_rerank_weight,
                type_margin_weight=ner_type_margin_weight,
                boundary_rerank_weight=ner_boundary_rerank_weight,
                type_rerank_weight=ner_type_rerank_weight,
            )
            for type_name, support in candidates.items()
        }
        rank_scores = {name: values[0] for name, values in scored_candidates.items()}
        if ner_type_confusion_penalty and len(scored_candidates) > 1:
            max_margin = max(values[3] for values in scored_candidates.values())
            rank_scores = {
                name: values[0] - ner_type_confusion_penalty * max(0.0, max_margin - values[3])
                for name, values in scored_candidates.items()
            }
        if ner_overlap_strategy == "none":
            # Exact legacy behavior for reproducible baseline/HCBR numbers.
            type_name, _ = max(
                candidates.items(),
                key=lambda item: (
                    len(item[1]),
                    max(x.get("centrality", 0.0) for x in item[1]),
                item[0],
            ),
        )
        else:
            type_name, _ = max(
                candidates.items(),
                key=lambda item: (
                    scored_candidates[item[0]][0],
                    len(item[1]),
                    scored_candidates[item[0]][2],
                    item[0],
                ),
            )
            if ner_type_confusion_penalty and len(candidates) > 1:
                type_name = max(
                    candidates,
                    key=lambda name: (
                        rank_scores[name],
                        len(candidates[name]),
                        scored_candidates[name][2],
                        name,
                    ),
                )
        support = candidates[type_name]
        selection_score, type_confidence, centrality, type_margin = scored_candidates[type_name]
        selection_score = rank_scores[type_name]
        evidence_scores = (
            {name: float(len(items)) for name, items in candidates.items()}
            if ner_overlap_strategy == "none"
            else {name: rank_scores[name] for name in scored_candidates}
        )
        total_score = sum(evidence_scores.values()) or 1.0
        confidences = [x.get("confidence") for x in support if x.get("confidence") is not None]
        span_confidences = [x.get("span_confidence") for x in support if x.get("span_confidence") is not None]
        candidate = {
            "start": start,
            "end": end,
            "type": type_name,
            "text": "".join(raw["tokens"][start:end]),
            "window_votes": len(support),
            "window_ids": sorted({x["window_id"] for x in support}),
            "type_probabilities": {
                name: score / total_score for name, score in sorted(evidence_scores.items())
            },
            "span_confidence": max(span_confidences) if span_confidences else None,
            "type_confidence": type_confidence if confidences else None,
            "type_margin": type_margin if confidences else None,
            "boundary_score": max(span_confidences) if span_confidences else 0.0,
            "type_score": 0.7 * (type_confidence if confidences else 0.0) + 0.3 * (type_margin if confidences else 0.0),
            "confidence": max(confidences) if confidences else None,
            "selection_score": selection_score,
            "centrality": centrality,
        }
        source_id = _source_document_id(raw, start, end)
        if source_id is not None:
            candidate["source_document_id"] = source_id
        boundary_candidates.append(candidate)

    predictions = _select_flat_entities(
        boundary_candidates,
        ner_overlap_strategy,
        boundary_shift_penalty=ner_boundary_shift_penalty,
    )

    # Calibration searches only the NER decoder.  Avoid scanning relation and
    # relation-score evidence for every threshold/strategy combination while
    # preserving the regular inference path by keeping this opt-in.
    if not include_relations:
        gold_entities = [
            {
                "start": entity["start"],
                "end": entity["end"],
                "type": entity["type"],
                "text": "".join(raw["tokens"][entity["start"]:entity["end"]]),
            }
            for entity in raw.get("entities", [])
        ]
        return {
            "gold": {"entities": gold_entities, "relations": []},
            "prediction": {"entities": predictions, "relations": []},
            "_local_score_map": {},
        }

    selected_keys = {entity_key(entity) for entity in predictions}
    entity_indices = {entity_key(entity): idx for idx, entity in enumerate(predictions)}

    relation_evidence = defaultdict(list)
    for relation in evidence["relations"]:
        head, tail = tuple(relation["head_key"]), tuple(relation["tail_key"])
        if head not in selected_keys or tail not in selected_keys:
            continue
        if relation["type"] in symmetric_names and tail < head:
            head, tail = tail, head
        relation_evidence[(head, relation["type"], tail)].append(relation)

    predicted_relations = []
    for (head, type_name, tail), support in relation_evidence.items():
        probabilities = [x["probability"] for x in support if x["probability"] is not None]
        predicted_relations.append({
            "head": entity_indices[head],
            "tail": entity_indices[tail],
            "type": type_name,
            "symmetric": type_name in symmetric_names,
            "window_votes": len(support),
            "max_probability": max(probabilities) if probabilities else None,
            "window_ids": sorted({x["window_id"] for x in support}),
        })
    predicted_relations.sort(key=lambda x: (x["head"], x["tail"], x["type"]))

    local_score_map = {}
    for relation in evidence.get("relation_scores", []):
        head, tail = tuple(relation["head_key"]), tuple(relation["tail_key"])
        if head not in selected_keys or tail not in selected_keys:
            continue
        if relation["type"] in symmetric_names and tail < head:
            head, tail = tail, head
        key = (head, relation["type"], tail)
        local_score_map[key] = max(local_score_map.get(key, 0.0), relation["probability"])

    gold_entities = [
        {
            "start": entity["start"],
            "end": entity["end"],
            "type": entity["type"],
            "text": "".join(raw["tokens"][entity["start"]:entity["end"]]),
        }
        for entity in raw.get("entities", [])
    ]
    observable_relation_ids = {
        relation_id
        for record in window_records
        for relation_id in record.get("source_relation_ids", [])
    }
    gold_relations = [
        {
            "head": relation["head"],
            "tail": relation["tail"],
            "type": relation["type"],
            "symmetric": relation["type"] in symmetric_names,
            "window_observable": idx in observable_relation_ids,
        }
        for idx, relation in enumerate(raw.get("relations", []))
    ]
    return {
        "gold": {"entities": gold_entities, "relations": gold_relations},
        "prediction": {"entities": predictions, "relations": predicted_relations},
        "_local_score_map": local_score_map,
    }


def _serialize_local_relation_scores(
    prediction: dict,
    local_score_map: dict,
    symmetric_names: set[str],
) -> list[list]:
    """Convert entity-keyed local scores to compact prediction-entity indices."""
    entity_indices = {
        entity_key(entity): entity_id
        for entity_id, entity in enumerate(prediction["entities"])
    }
    rows = []
    for (head_key, type_name, tail_key), probability in local_score_map.items():
        if head_key not in entity_indices or tail_key not in entity_indices:
            continue
        head_id = entity_indices[head_key]
        tail_id = entity_indices[tail_key]
        rows.append([head_id, tail_id, type_name, float(probability)])
        if type_name in symmetric_names:
            rows.append([tail_id, head_id, type_name, float(probability)])
    return sorted(rows, key=lambda item: (item[0], item[1], item[2]))


def _fusion_branch_weights(fusion_gate: float | None) -> tuple[float | None, float | None]:
    """Return the local/global branch weights used by learned fusion."""
    if fusion_gate is None:
        return None, None
    local_weight = float(fusion_gate)
    return local_weight, 1.0 - local_weight


def _gold_relation_fusion_metadata(
    gold: dict | None,
    symmetric_names: set[str],
) -> dict[tuple, dict]:
    metadata = {}
    if gold is None:
        return metadata
    gold_entities = gold.get("entities", [])
    for relation_id, relation in enumerate(gold.get("relations", [])):
        head = gold_entities[relation["head"]]
        tail = gold_entities[relation["tail"]]
        head_key = entity_key(head)
        tail_key = entity_key(tail)
        if relation["type"] in symmetric_names and tail_key < head_key:
            head_key, tail_key = tail_key, head_key
        metadata[(head_key, relation["type"], tail_key)] = {
            "gold_relation_id": relation_id,
            "relation_type": relation["type"],
            "head_start": head["start"],
            "head_end": head["end"],
            "head_type": head["type"],
            "tail_start": tail["start"],
            "tail_end": tail["end"],
            "tail_type": tail["type"],
            "distance": abs(
                (head["start"] + head["end"]) // 2
                - (tail["start"] + tail["end"]) // 2
            ),
            "same_window": bool(relation.get("window_observable", True)),
        }
    return metadata


def _missing_fusion_trace(metadata: dict) -> dict:
    return {
        **metadata,
        "endpoint_match": False,
        "local_available": False,
        "local_gate_weight": None,
        "global_gate_weight": None,
        "local_probability": None,
        "global_probability": None,
        "final_probability": None,
        "relation_threshold": None,
        "is_recalled": False,
    }


@torch.no_grad()
def _apply_document_relation(
    local_model,
    document_model: DocumentRelationModel,
    raw: dict,
    prediction: dict,
    local_score_map: dict,
    symmetric_names: set[str],
    max_length: int,
    overlap_words: int,
    device: str,
    threshold: float,
    review_margin: float,
    local_global_alpha: float,
    max_entities: int,
    relation_thresholds: dict[str, float] | None = None,
    gold: dict | None = None,
    collect_fusion_trace: bool = False,
) -> None:
    entities = prediction["entities"]
    gold_relation_metadata = (
        _gold_relation_fusion_metadata(gold, symmetric_names)
        if collect_fusion_trace
        else {}
    )
    if len(entities) < 2:
        prediction["relations"] = []
        if collect_fusion_trace:
            prediction["fusion_trace"] = sorted(
                (_missing_fusion_trace(item) for item in gold_relation_metadata.values()),
                key=lambda item: item["gold_relation_id"],
            )
        return

    ranked_indices = sorted(
        range(len(entities)),
        key=lambda idx: (
            entities[idx].get("window_votes", 1),
            max(entities[idx].get("type_probabilities", {entities[idx]["type"]: 1.0}).values()),
            -(entities[idx]["end"] - entities[idx]["start"]),
        ),
        reverse=True,
    )[:max_entities]
    ranked_indices.sort()
    global_entities = [entities[idx] for idx in ranked_indices]
    features = encode_document_features(
        local_model=local_model,
        raw_document=raw,
        entities=global_entities,
        entity_types=document_model.config.entity_types,
        max_length=max_length,
        overlap_words=overlap_words,
        device=device,
    )
    pairs = DocumentRelationModel.all_ordered_pairs(len(global_entities), device=device)
    # Apply provenance filtering whenever this is a provenance-aware
    # cross-window document. Unknown spans (for example a span crossing the
    # separator) are excluded rather than becoming cross-source candidates.
    if global_entities and any("source_document_id" in entity for entity in global_entities):
        source_ids = [entity.get("source_document_id") for entity in global_entities]
        keep = torch.tensor(
            [
                source_ids[head] is not None
                and source_ids[tail] is not None
                and source_ids[head] == source_ids[tail]
                for head, tail in pairs.tolist()
            ],
            dtype=torch.bool,
            device=device,
        )
        pairs = pairs[keep]
    local_probabilities = None
    local_available = None
    if document_model.config.fusion_mode == "learned":
        local_probabilities = torch.zeros(
            (pairs.size(0), len(document_model.config.relation_types)),
            device=device,
            dtype=features["entity_repr"].dtype,
        )
        local_available = torch.zeros_like(local_probabilities, dtype=torch.bool)
        for pair_row, pair in enumerate(pairs.tolist()):
            head_key = entity_key(global_entities[pair[0]])
            tail_key = entity_key(global_entities[pair[1]])
            for relation_id, type_name in enumerate(document_model.config.relation_types):
                canonical_head, canonical_tail = head_key, tail_key
                if type_name in symmetric_names and canonical_tail < canonical_head:
                    canonical_head, canonical_tail = canonical_tail, canonical_head
                probability = local_score_map.get((canonical_head, type_name, canonical_tail))
                if probability is not None:
                    local_probabilities[pair_row, relation_id] = float(probability)
                    local_available[pair_row, relation_id] = True
    output = document_model(
        **features,
        pairs=pairs,
        local_probabilities=local_probabilities,
        local_available=local_available,
    )
    candidates = {}
    review_candidates = {}
    fusion_trace = {}
    fusion_diagnostics = {
        "available_scores": 0,
        "local_weight_sum": 0.0,
        "global_suppressed_local": 0,
        "global_supplemented_local": 0,
        "distance_buckets": {
            name: {"count": 0, "local_weight_sum": 0.0}
            for name in ("0-31", "32-63", "64-127", ">=128")
        },
    }
    top_k = min(document_model.config.evidence_top_k, features["window_repr"].size(0))
    for pair_row, pair in enumerate(output.pairs.tolist()):
        source_head, source_tail = ranked_indices[pair[0]], ranked_indices[pair[1]]
        head_key = entity_key(entities[source_head])
        tail_key = entity_key(entities[source_tail])
        evidence_scores, evidence_ids = output.evidence_attention[pair_row].topk(top_k)
        evidence = [
            {
                "window_id": int(window_id),
                "score": float(score),
                "word_start": int(features["window_ranges"][window_id, 0].item()),
                "word_end": int(features["window_ranges"][window_id, 1].item()),
            }
            for score, window_id in zip(evidence_scores.tolist(), evidence_ids.tolist())
        ]
        for relation_id, type_name in enumerate(document_model.config.relation_types):
            output_head, output_tail = source_head, source_tail
            canonical_head, canonical_tail = head_key, tail_key
            if type_name in symmetric_names and canonical_tail < canonical_head:
                canonical_head, canonical_tail = canonical_tail, canonical_head
                output_head, output_tail = output_tail, output_head
            local_probability = local_score_map.get((canonical_head, type_name, canonical_tail))
            global_probability = float(output.probabilities[pair_row, relation_id].item())
            fusion_gate = None
            if output.fused_probabilities is not None:
                final_probability = float(
                    output.fused_probabilities[pair_row, relation_id].item()
                )
                fusion_gate = float(output.fusion_gate[pair_row, relation_id].item())
                source = "learned_local_global" if local_probability is not None else "global"
                if local_probability is not None:
                    distance = abs(
                        float(features["entity_positions"][pair[0]].item())
                        - float(features["entity_positions"][pair[1]].item())
                    )
                    distance_bucket = (
                        "0-31" if distance < 32 else
                        "32-63" if distance < 64 else
                        "64-127" if distance < 128 else
                        ">=128"
                    )
                    fusion_diagnostics["available_scores"] += 1
                    fusion_diagnostics["local_weight_sum"] += fusion_gate
                    bucket = fusion_diagnostics["distance_buckets"][distance_bucket]
                    bucket["count"] += 1
                    bucket["local_weight_sum"] += fusion_gate
                    if final_probability < float(local_probability) - 1e-8:
                        fusion_diagnostics["global_suppressed_local"] += 1
                    elif final_probability > float(local_probability) + 1e-8:
                        fusion_diagnostics["global_supplemented_local"] += 1
            elif local_probability is None:
                final_probability = global_probability
                source = "global"
            else:
                blended = (
                    local_global_alpha * global_probability
                    + (1.0 - local_global_alpha) * local_probability
                )
                final_probability = max(local_probability, blended)
                source = "local_global"
            # Optional IND-specific pair-existence gate.  It is disabled for
            # the long-context v8 checkpoints, so their predictions remain
            # byte-for-byte compatible.  When enabled, relation confidence is
            # discounted by the probability that this entity pair has any
            # annotated relation at all, reducing high-confidence pair-level
            # false positives before relation-type thresholding.
            pair_link_probability = None
            if output.pair_link_logits is not None:
                pair_link_probability = float(
                    torch.sigmoid(output.pair_link_logits[pair_row]).item()
                )
                gate_weight = float(document_model.config.pair_link_gate_weight)
                effective_link = (1.0 - gate_weight) + gate_weight * pair_link_probability
                final_probability *= effective_link
            key = (output_head, type_name, output_tail)
            candidate = {
                "head": output_head,
                "tail": output_tail,
                "type": type_name,
                "symmetric": type_name in symmetric_names,
                "source": source,
                "final_probability": final_probability,
                "global_probability": global_probability,
                "local_probability": local_probability,
                "fusion_gate": fusion_gate,
                "pair_link_probability": pair_link_probability,
                "pair_link_gate_weight": (
                    document_model.config.pair_link_gate_weight
                    if output.pair_link_logits is not None else None
                ),
                "evidence": evidence,
            }
            relation_cutoff = (
                relation_thresholds.get(type_name, threshold)
                if relation_thresholds is not None
                else threshold
            )
            candidate["relation_threshold"] = relation_cutoff
            gold_key = (canonical_head, type_name, canonical_tail)
            if gold_key in gold_relation_metadata:
                metadata = gold_relation_metadata[gold_key]
                local_weight, global_weight = _fusion_branch_weights(fusion_gate)
                trace_item = {
                    **metadata,
                    "endpoint_match": True,
                    "local_available": local_probability is not None,
                    "local_gate_weight": local_weight,
                    "global_gate_weight": global_weight,
                    "local_probability": local_probability,
                    "global_probability": global_probability,
                    "final_probability": final_probability,
                    "relation_threshold": relation_cutoff,
                    "is_recalled": final_probability >= relation_cutoff,
                }
                previous = fusion_trace.get(gold_key)
                if previous is None or final_probability > previous["final_probability"]:
                    fusion_trace[gold_key] = trace_item
            if final_probability < relation_cutoff:
                if final_probability >= max(0.0, relation_cutoff - review_margin):
                    candidate["review_reason"] = "below_relation_threshold"
                    if key not in review_candidates or final_probability > review_candidates[key]["final_probability"]:
                        review_candidates[key] = candidate
                continue
            if key not in candidates or final_probability > candidates[key]["final_probability"]:
                candidates[key] = candidate
    prediction["relations"] = sorted(
        candidates.values(), key=lambda item: (item["head"], item["tail"], item["type"])
    )
    prediction["review_candidates"] = sorted(
        review_candidates.values(), key=lambda item: item["final_probability"], reverse=True
    )
    if output.fused_probabilities is not None:
        count = fusion_diagnostics["available_scores"]
        fusion_diagnostics["mean_local_weight"] = (
            fusion_diagnostics["local_weight_sum"] / count if count else 0.0
        )
        for bucket in fusion_diagnostics["distance_buckets"].values():
            bucket["mean_local_weight"] = (
                bucket["local_weight_sum"] / bucket["count"] if bucket["count"] else 0.0
            )
        prediction["fusion_diagnostics"] = fusion_diagnostics
    if collect_fusion_trace:
        for key, metadata in gold_relation_metadata.items():
            if key not in fusion_trace:
                fusion_trace[key] = _missing_fusion_trace(metadata)
        prediction["fusion_trace"] = sorted(
            fusion_trace.values(), key=lambda item: item["gold_relation_id"]
        )


def _comparison(document: dict) -> dict:
    gold, pred = document["gold"], document["prediction"]
    ge = {entity_key(e): e for e in gold["entities"]}
    pe = {entity_key(e): e for e in pred["entities"]}
    gold_by_boundary = {(e["start"], e["end"]): e for e in gold["entities"]}
    pred_by_boundary = {(e["start"], e["end"]): e for e in pred["entities"]}
    type_errors = []
    for boundary in gold_by_boundary.keys() & pred_by_boundary.keys():
        if gold_by_boundary[boundary]["type"] != pred_by_boundary[boundary]["type"]:
            type_errors.append({"gold": gold_by_boundary[boundary], "predicted": pred_by_boundary[boundary]})

    gr = {relation_key(r, gold["entities"], True): r for r in gold["relations"]}
    pr = {relation_key(r, pred["entities"], True): r for r in pred["relations"]}
    return {
        "entity_tp": [ge[k] for k in sorted(ge.keys() & pe.keys())],
        "entity_fn": [ge[k] for k in sorted(ge.keys() - pe.keys())],
        "entity_fp": [pe[k] for k in sorted(pe.keys() - ge.keys())],
        "entity_type_errors": type_errors,
        "relation_tp": [list(k) for k in sorted(gr.keys() & pr.keys())],
        "relation_fn": [list(k) for k in sorted(gr.keys() - pr.keys())],
        "relation_fp": [list(k) for k in sorted(pr.keys() - gr.keys())],
    }


def _comparison_rows(documents: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    document_rows = []
    entity_rows = []
    relation_rows = []
    for document in documents:
        document_id = document["document_id"]
        gold = document["gold"]
        pred = document["prediction"]
        gold_by_boundary = {(e["start"], e["end"]): e for e in gold["entities"]}
        pred_by_boundary = {(e["start"], e["end"]): e for e in pred["entities"]}
        entity_status_counts = defaultdict(int)
        for boundary in sorted(gold_by_boundary.keys() | pred_by_boundary.keys()):
            gold_entity = gold_by_boundary.get(boundary)
            predicted_entity = pred_by_boundary.get(boundary)
            if gold_entity and predicted_entity:
                status = "TP" if gold_entity["type"] == predicted_entity["type"] else "TYPE_ERROR"
            elif gold_entity:
                status = "FN"
            else:
                status = "FP"
            entity_status_counts[status] += 1
            entity_rows.append({
                "document_id": document_id,
                "status": status,
                "start": boundary[0],
                "end": boundary[1],
                "text": (gold_entity or predicted_entity).get("text", ""),
                "gold_type": gold_entity["type"] if gold_entity else None,
                "predicted_type": predicted_entity["type"] if predicted_entity else None,
                "window_votes": predicted_entity.get("window_votes") if predicted_entity else None,
                "window_ids": json.dumps(predicted_entity.get("window_ids", [])) if predicted_entity else None,
                "type_probabilities": json.dumps(
                    predicted_entity.get("type_probabilities", {}), ensure_ascii=False
                ) if predicted_entity else None,
                "span_confidence": predicted_entity.get("span_confidence") if predicted_entity else None,
                "type_confidence": predicted_entity.get("type_confidence") if predicted_entity else None,
                "confidence": predicted_entity.get("confidence") if predicted_entity else None,
                "selection_score": predicted_entity.get("selection_score") if predicted_entity else None,
            })

        gold_relations = {relation_key(r, gold["entities"], True): r for r in gold["relations"]}
        pred_relations = {relation_key(r, pred["entities"], True): r for r in pred["relations"]}
        relation_status_counts = defaultdict(int)
        for key in sorted(gold_relations.keys() | pred_relations.keys()):
            gold_relation = gold_relations.get(key)
            predicted_relation = pred_relations.get(key)
            status = "TP" if gold_relation and predicted_relation else ("FN" if gold_relation else "FP")
            relation_status_counts[status] += 1
            head, relation_type, tail = key
            relation_rows.append({
                "document_id": document_id,
                "status": status,
                "head_start": head[0],
                "head_end": head[1],
                "head_type": head[2],
                "head_text": "".join(document["tokens"][head[0]:head[1]]),
                "relation_type": relation_type,
                "tail_start": tail[0],
                "tail_end": tail[1],
                "tail_type": tail[2],
                "tail_text": "".join(document["tokens"][tail[0]:tail[1]]),
                "window_observable": gold_relation.get("window_observable") if gold_relation else None,
                "source": predicted_relation.get("source", "local") if predicted_relation else None,
                "local_probability": predicted_relation.get("local_probability") if predicted_relation else None,
                "global_probability": predicted_relation.get("global_probability") if predicted_relation else None,
                "fusion_gate": predicted_relation.get("fusion_gate") if predicted_relation else None,
                "final_probability": (
                    predicted_relation.get("final_probability", predicted_relation.get("max_probability"))
                    if predicted_relation else None
                ),
                "evidence": json.dumps(
                    predicted_relation.get("evidence", []), ensure_ascii=False
                ) if predicted_relation else None,
            })
        document_rows.append({
            "document_id": document_id,
            "word_count": document["word_count"],
            "is_long": document["is_long"],
            "gold_entities": len(gold["entities"]),
            "predicted_entities": len(pred["entities"]),
            "entity_tp": entity_status_counts["TP"],
            "entity_type_error": entity_status_counts["TYPE_ERROR"],
            "entity_fn": entity_status_counts["FN"],
            "entity_fp": entity_status_counts["FP"],
            "gold_relations": len(gold["relations"]),
            "predicted_relations": len(pred["relations"]),
            "relation_tp": relation_status_counts["TP"],
            "relation_fn": relation_status_counts["FN"],
            "relation_fp": relation_status_counts["FP"],
        })
    return document_rows, entity_rows, relation_rows


def _write_table(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison_workbook(
    path: Path, document_rows: list[dict], entity_rows: list[dict], relation_rows: list[dict]
) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError:
        return
    workbook = Workbook()
    workbook.remove(workbook.active)
    for title, rows in (
        ("documents", document_rows),
        ("entities", entity_rows),
        ("relations", relation_rows),
    ):
        sheet = workbook.create_sheet(title)
        if not rows:
            continue
        headers = list(rows[0])
        sheet.append(headers)
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for row in rows:
            sheet.append([row[name] for name in headers])
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
    workbook.save(path)


def _write_outputs(documents: list[dict], metrics: dict, output_dir: Path, split: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{split}_predictions.jsonl"
    errors_path = output_dir / f"{split}_errors.jsonl"
    with predictions_path.open("w", encoding="utf-8") as predictions_file, errors_path.open(
        "w", encoding="utf-8"
    ) as errors_file:
        for document in documents:
            predictions_file.write(json.dumps(document, ensure_ascii=False) + "\n")
            comparison = document["comparison"]
            if comparison["entity_fn"] or comparison["entity_fp"] or comparison["relation_fn"] or comparison["relation_fp"]:
                error_document = {
                    key: value
                    for key, value in document.items()
                    if key != "local_relation_scores"
                }
                errors_file.write(json.dumps(error_document, ensure_ascii=False) + "\n")

    with (output_dir / f"{split}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (output_dir / f"{split}_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slice", "task", "precision", "recall", "f1", "tp", "fp", "fn", "documents"])
        slices = {"overall": metrics["overall"], "long_documents": metrics["long_documents"], **metrics["length_buckets"]}
        for slice_name, values in slices.items():
            for task in (
                "ner_strict", "ner_boundary", "re_strict_end_to_end",
                "re_boundary_end_to_end", "re_on_matched_entities",
            ):
                score = values[task]
                writer.writerow([
                    slice_name, task, score["precision"], score["recall"], score["f1"],
                    score["tp"], score["fp"], score["fn"], values["documents"],
                ])
    with (output_dir / f"{split}_per_type_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "type", "precision", "recall", "f1", "tp", "fp", "fn", "support"])
        for task, values in (("ner", metrics["ner_per_type"]), ("re", metrics["re_per_type"])):
            for type_name, score in values.items():
                writer.writerow([
                    task, type_name, score["precision"], score["recall"], score["f1"],
                    score["tp"], score["fp"], score["fn"], score["support"],
                ])
    document_rows, entity_rows, relation_rows = _comparison_rows(documents)
    _write_table(output_dir / f"{split}_document_comparison.csv", document_rows)
    _write_table(output_dir / f"{split}_entity_comparison.csv", entity_rows)
    _write_table(output_dir / f"{split}_relation_comparison.csv", relation_rows)
    _write_comparison_workbook(
        output_dir / f"{split}_comparison.xlsx", document_rows, entity_rows, relation_rows
    )
    review_rows = []
    for document in documents:
        entities = document["prediction"]["entities"]
        for relation in document["prediction"].get("review_candidates", []):
            head = entities[relation["head"]]
            tail = entities[relation["tail"]]
            review_rows.append({
                "document_id": document["document_id"],
                "head_text": head["text"],
                "head_start": head["start"],
                "head_end": head["end"],
                "head_type": head["type"],
                "relation_type": relation["type"],
                "tail_text": tail["text"],
                "tail_start": tail["start"],
                "tail_end": tail["end"],
                "tail_type": tail["type"],
                "final_probability": relation["final_probability"],
                "local_probability": relation["local_probability"],
                "global_probability": relation["global_probability"],
                "fusion_gate": relation.get("fusion_gate"),
                "evidence": json.dumps(relation["evidence"], ensure_ascii=False),
                "review_reason": relation["review_reason"],
            })
    _write_table(output_dir / f"{split}_review_candidates.csv", review_rows)


def main() -> None:
    process_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained ITER checkpoint directory")
    parser.add_argument(
        "--document_re_checkpoint",
        type=Path,
        default=None,
        help="Optional second-stage document relation checkpoint",
    )
    parser.add_argument("--config", default="wenyanwen/wenyanwen_long_context_512")
    parser.add_argument("--split", default="test", choices=("train", "eval", "test"))
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--long_threshold", type=int, default=128)
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max_documents", type=int, default=None, help="Debug-only source document limit")
    parser.add_argument(
        "--ner_span_threshold",
        type=float,
        default=0.5,
        help="Minimum model posterior for an entity span; calibrate on eval/dev only",
    )
    parser.add_argument(
        "--ner_type_threshold",
        type=float,
        default=0.0,
        help="Minimum posterior for the selected entity type; calibrate on eval/dev only",
    )
    parser.add_argument(
        "--ner_overlap_strategy",
        choices=("none", "nms", "weighted_interval"),
        default="none",
        help="Document-level flat-NER overlap handling (none preserves the legacy decoder)",
    )
    parser.add_argument(
        "--ner_joint_rerank_weight",
        type=float,
        default=0.0,
        help="Blend span and type confidence during document NER reranking; select on eval/dev only",
    )
    parser.add_argument(
        "--ner_type_margin_weight",
        type=float,
        default=0.0,
        help="Extra weight for top1-vs-top2 type confidence margin during NER reranking",
    )
    parser.add_argument("--ner_boundary_rerank_weight", type=float, default=0.0)
    parser.add_argument("--ner_type_rerank_weight", type=float, default=0.0)
    parser.add_argument("--ner_boundary_shift_penalty", type=float, default=0.0)
    parser.add_argument("--ner_type_confusion_penalty", type=float, default=0.0)
    parser.add_argument(
        "--dump_ner_evidence",
        type=Path,
        default=None,
        help="Dump raw eval evidence for offline dev threshold calibration",
    )
    parser.add_argument("--relation_threshold", type=float, default=None)
    parser.add_argument(
        "--relation_thresholds_file",
        type=Path,
        default=None,
        help="Optional eval-calibrated per-relation threshold JSON",
    )
    parser.add_argument(
        "--review_margin",
        type=float,
        default=0.10,
        help="Export rejected relation candidates within this distance below the threshold",
    )
    parser.add_argument("--local_global_alpha", type=float, default=0.65)
    parser.add_argument(
        "--export_local_relation_scores",
        action="store_true",
        help="Persist dense Stage-1 local RE scores for learned-fusion training",
    )
    parser.add_argument(
        "--export_fusion_trace",
        action="store_true",
        help=(
            "Persist relation-specific local/global gate values for gold relations; "
            "intended for analysis only and does not change predictions"
        ),
    )
    parser.add_argument("--max_document_entities", type=int, default=64)
    parser.add_argument(
        "--benchmark_output",
        type=Path,
        default=None,
        help="Optional JSON report with parameter counts and synchronized inference timings",
    )
    parser.add_argument(
        "--benchmark_warmup_runs",
        type=int,
        default=0,
        help="Unmeasured in-process inference runs before benchmark measurement",
    )
    parser.add_argument(
        "--benchmark_repeats",
        type=int,
        default=1,
        help="Measured in-process inference runs (requires --benchmark_output)",
    )
    args = parser.parse_args()

    if not 0.0 <= args.local_global_alpha <= 1.0:
        parser.error("--local_global_alpha must be between 0 and 1")
    if args.relation_threshold is not None and not 0.0 <= args.relation_threshold <= 1.0:
        parser.error("--relation_threshold must be between 0 and 1")
    if not 0.0 <= args.review_margin <= 1.0:
        parser.error("--review_margin must be between 0 and 1")
    if not 0.0 <= args.ner_span_threshold <= 1.0:
        parser.error("--ner_span_threshold must be between 0 and 1")
    if not 0.0 <= args.ner_type_threshold <= 1.0:
        parser.error("--ner_type_threshold must be between 0 and 1")
    if not 0.0 <= args.ner_joint_rerank_weight <= 1.0:
        parser.error("--ner_joint_rerank_weight must be between 0 and 1")
    if not 0.0 <= args.ner_type_margin_weight <= 1.0:
        parser.error("--ner_type_margin_weight must be between 0 and 1")
    for name in (
        "ner_boundary_rerank_weight",
        "ner_type_rerank_weight",
        "ner_boundary_shift_penalty",
        "ner_type_confusion_penalty",
    ):
        if getattr(args, name) < 0.0:
            parser.error(f"--{name} must be non-negative")
    if args.dump_ner_evidence is not None and args.split not in {"eval", "dev"}:
        parser.error("--dump_ner_evidence is only allowed with the eval/dev split, never test")
    if args.max_document_entities < 2:
        parser.error("--max_document_entities must be at least 2")
    if args.benchmark_warmup_runs < 0:
        parser.error("--benchmark_warmup_runs must be non-negative")
    if args.benchmark_repeats < 1:
        parser.error("--benchmark_repeats must be at least 1")
    if args.benchmark_output is None and (
        args.benchmark_warmup_runs != 0 or args.benchmark_repeats != 1
    ):
        parser.error("Benchmark warm-up/repeat options require --benchmark_output")
    if args.benchmark_output is not None and args.dump_ner_evidence is not None:
        parser.error("Benchmark mode cannot be combined with --dump_ner_evidence")

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    if not args.checkpoint.exists():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    _synchronize_device(device)
    model_load_started = time.perf_counter()
    load_kwargs = {"local_files_only": True} if args.offline else {}
    config = ITERConfig.from_pretrained(args.checkpoint, **load_kwargs)
    model_class = ITER if config.is_feature_ner_only else ITERForRelationExtraction
    model = model_class.from_pretrained(args.checkpoint, **load_kwargs)
    model = model.to(device).eval()
    document_model = None
    if args.document_re_checkpoint is not None:
        if not args.document_re_checkpoint.exists():
            parser.error(f"Document RE checkpoint does not exist: {args.document_re_checkpoint}")
        document_model = DocumentRelationModel.from_pretrained(
            args.document_re_checkpoint, device=device
        ).eval()
    _synchronize_device(device)
    model_load_seconds = time.perf_counter() - model_load_started
    relation_thresholds = None
    if args.relation_thresholds_file is not None:
        if not args.relation_thresholds_file.exists():
            parser.error(f"Relation thresholds file does not exist: {args.relation_thresholds_file}")
        with args.relation_thresholds_file.open(encoding="utf-8") as handle:
            threshold_payload = json.load(handle)
        if threshold_payload.get("source_split") not in {"eval", "dev"}:
            parser.error(
                "Per-relation thresholds must come from eval/dev, never test "
                f"(got {threshold_payload.get('source_split')!r})"
            )
        relation_thresholds = threshold_payload.get("thresholds")
        if not isinstance(relation_thresholds, dict):
            parser.error("Per-relation threshold file must contain a 'thresholds' object")
        relation_thresholds = {str(name): float(value) for name, value in relation_thresholds.items()}
        if any(not 0.0 <= value <= 1.0 for value in relation_thresholds.values()):
            parser.error("Per-relation thresholds must be between 0 and 1")

    dataset_prepare_started = time.perf_counter()
    dataset = CoNLL.from_name(
        args.config,
        tokenizer=model.tokenizer,
        tokenizer_config=model.config.transformer_config,
        reinitialize_dataset=args.rebuild_cache,
    )
    hparams = Hparams.from_name(args.config, eval_batch_size=args.batch_size)
    if not model.config.entity_types:
        model.config.entity_types = dataset.entity_types
    if not model.config.link_types:
        model.config.link_types = dataset.link_types
    dataset.setup_dataset()
    raw_documents = dataset.open_split(args.split, dataset.splits[args.split])
    manifest = dataset.window_manifests.get(args.split, [])
    if len(manifest) != len(dataset[args.split]):
        raise RuntimeError(
            f"Window manifest has {len(manifest)} rows but tensor dataset has {len(dataset[args.split])}. "
            "Run again with --rebuild_cache."
        )

    dataloader = create_primary_dataloader([dataset], args.split, False, hparams, 1, 0, 0, False)
    dataset_prepare_seconds = time.perf_counter() - dataset_prepare_started
    decode_span_threshold = 0.0 if args.dump_ner_evidence is not None else args.ner_span_threshold
    decode_type_threshold = 0.0 if args.dump_ner_evidence is not None else args.ner_type_threshold
    symmetric_names = {dataset.link_types[idx] for idx in dataset.symmetric_link_types}
    source_limit = min(len(raw_documents), args.max_documents) if args.max_documents is not None else len(raw_documents)
    if source_limit == 0:
        raise RuntimeError(f"Split {args.split!r} contains no source documents")

    def run_stage1(run_label: str):
        grouped_records = defaultdict(list)
        evidence = defaultdict(lambda: {"entities": [], "relations": [], "relation_scores": []})
        cursor = 0
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.startswith("cuda")
            else contextlib.nullcontext()
        )
        with torch.no_grad(), amp_context:
            for batch in tqdm(dataloader, desc=run_label):
                batch_size = batch.input_ids.size(0)
                device_batch = batch.to(device)
                generation = model.generate(
                    device_batch.input_ids, attention_mask=device_batch.attention_mask
                )
                for batch_index in range(batch_size):
                    record = manifest[cursor + batch_index]
                    document_id = record["document_id"]
                    if args.max_documents is not None and document_id >= args.max_documents:
                        continue
                    entities, relations, relation_scores = _decode_window(
                        model,
                        generation,
                        batch_index,
                        raw_documents[document_id],
                        record,
                        ner_span_threshold=decode_span_threshold,
                        ner_type_threshold=decode_type_threshold,
                        include_relations=args.dump_ner_evidence is None,
                    )
                    evidence[document_id]["entities"].extend(entities)
                    evidence[document_id]["relations"].extend(relations)
                    evidence[document_id]["relation_scores"].extend(relation_scores)
                    grouped_records[document_id].append(record)
                cursor += batch_size
        return grouped_records, evidence

    def merge_documents(grouped_records, evidence):
        documents = []
        for document_id in range(source_limit):
            raw = raw_documents[document_id]
            merged = _merge_document(
                raw,
                grouped_records[document_id],
                evidence[document_id],
                symmetric_names,
                ner_overlap_strategy=args.ner_overlap_strategy,
                ner_joint_rerank_weight=args.ner_joint_rerank_weight,
                ner_type_margin_weight=args.ner_type_margin_weight,
                ner_boundary_rerank_weight=args.ner_boundary_rerank_weight,
                ner_type_rerank_weight=args.ner_type_rerank_weight,
                ner_boundary_shift_penalty=args.ner_boundary_shift_penalty,
                ner_type_confusion_penalty=args.ner_type_confusion_penalty,
            )
            local_score_map = merged.pop("_local_score_map")
            if args.export_local_relation_scores:
                merged["local_relation_scores"] = _serialize_local_relation_scores(
                    merged["prediction"], local_score_map, symmetric_names
                )
            if document_model is not None:
                if document_model.config.entity_types != dataset.entity_types:
                    raise ValueError("Document RE checkpoint entity types do not match dataset")
                if document_model.config.relation_types != dataset.link_types:
                    raise ValueError("Document RE checkpoint relation types do not match dataset")
                threshold = (
                    args.relation_threshold
                    if args.relation_threshold is not None
                    else document_model.config.threshold
                )
                _apply_document_relation(
                    local_model=model,
                    document_model=document_model,
                    raw=raw,
                    prediction=merged["prediction"],
                    local_score_map=local_score_map,
                    symmetric_names=symmetric_names,
                    max_length=dataset.max_length,
                    overlap_words=dataset.window_overlap,
                    device=device,
                    threshold=threshold,
                    review_margin=args.review_margin,
                    local_global_alpha=args.local_global_alpha,
                    max_entities=args.max_document_entities,
                    relation_thresholds=relation_thresholds,
                    gold=merged["gold"],
                    collect_fusion_trace=args.export_fusion_trace,
                )
            document = {
                "document_id": document_id,
                "text": "".join(raw["tokens"]),
                "tokens": raw["tokens"],
                "word_count": len(raw["tokens"]),
                "is_long": len(raw["tokens"]) >= args.long_threshold,
                "windows": grouped_records[document_id],
                **merged,
            }
            document["comparison"] = _comparison(document)
            documents.append(document)
        return documents

    grouped_records = evidence = None
    if args.benchmark_output is None:
        grouped_records, evidence = run_stage1(f"Inferring {args.split}")
    if args.dump_ner_evidence is not None:
        assert grouped_records is not None and evidence is not None
        payload = {
            "version": 1,
            "evidence_kind": "ner_only",
            "split": args.split,
            "source_thresholds": {
                "ner_span_threshold": decode_span_threshold,
                "ner_type_threshold": decode_type_threshold,
                "ner_joint_rerank_weight": args.ner_joint_rerank_weight,
                "ner_type_margin_weight": args.ner_type_margin_weight,
                "ner_boundary_rerank_weight": args.ner_boundary_rerank_weight,
                "ner_type_rerank_weight": args.ner_type_rerank_weight,
                "ner_boundary_shift_penalty": args.ner_boundary_shift_penalty,
                "ner_type_confusion_penalty": args.ner_type_confusion_penalty,
            },
            "raw_documents": raw_documents[:source_limit],
            "window_records": {
                str(document_id): grouped_records[document_id]
                for document_id in range(source_limit)
            },
            # NER calibration does not need RE candidates.  Omitting them
            # keeps the evidence dump small and avoids serializing a dense
            # relation-score tensor for every window.
            "evidence": {
                str(document_id): {
                    "entities": [
                        {
                            field: entity.get(field)
                            for field in (
                                "start",
                                "end",
                                "type",
                                "window_id",
                                "centrality",
                                "span_confidence",
                                "type_confidence",
                                "type_margin",
                                "confidence",
                            )
                        }
                        for entity in evidence[document_id]["entities"]
                    ],
                    "relations": [],
                    "relation_scores": [],
                }
                for document_id in range(source_limit)
            },
            "symmetric_names": sorted(symmetric_names),
        }
        args.dump_ner_evidence.parent.mkdir(parents=True, exist_ok=True)
        args.dump_ner_evidence.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"NER evidence written to: {args.dump_ner_evidence.resolve()}")
        return

    benchmark_runs = []
    if args.benchmark_output is None:
        assert grouped_records is not None and evidence is not None
        documents = merge_documents(grouped_records, evidence)
    else:
        total_runs = args.benchmark_warmup_runs + args.benchmark_repeats
        documents = None
        for run_index in range(total_runs):
            measured = run_index >= args.benchmark_warmup_runs
            if measured and device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(torch.device(device))
            _synchronize_device(device)
            stage1_started = time.perf_counter()
            current_grouped, current_evidence = run_stage1(
                f"Benchmark {run_index + 1}/{total_runs}: Stage 1"
            )
            _synchronize_device(device)
            stage1_seconds = time.perf_counter() - stage1_started
            _synchronize_device(device)
            stage2_started = time.perf_counter()
            current_documents = merge_documents(current_grouped, current_evidence)
            _synchronize_device(device)
            stage2_seconds = time.perf_counter() - stage2_started
            documents = current_documents
            if measured:
                pure_seconds = stage1_seconds + stage2_seconds
                if device.startswith("cuda"):
                    peak_allocated = torch.cuda.max_memory_allocated(torch.device(device))
                    peak_reserved = torch.cuda.max_memory_reserved(torch.device(device))
                else:
                    peak_allocated = 0
                    peak_reserved = 0
                benchmark_runs.append({
                    "run": len(benchmark_runs) + 1,
                    "stage1_seconds": stage1_seconds,
                    "stage2_seconds": stage2_seconds,
                    "pure_inference_seconds": pure_seconds,
                    "milliseconds_per_document": pure_seconds * 1000.0 / source_limit,
                    "documents_per_second": source_limit / pure_seconds,
                    "peak_cuda_allocated_bytes": peak_allocated,
                    "peak_cuda_reserved_bytes": peak_reserved,
                })
        assert documents is not None

    evaluation_started = time.perf_counter()
    metrics = evaluate_documents(documents, long_threshold=args.long_threshold)
    if document_model is not None and document_model.config.fusion_mode == "learned":
        diagnostics = [
            document["prediction"].get("fusion_diagnostics", {})
            for document in documents
        ]
        total_count = sum(item.get("available_scores", 0) for item in diagnostics)
        metrics["fusion_diagnostics"] = {
            "available_scores": total_count,
            "mean_local_weight": (
                sum(item.get("local_weight_sum", 0.0) for item in diagnostics) / total_count
                if total_count else 0.0
            ),
            "global_suppressed_local": sum(
                item.get("global_suppressed_local", 0) for item in diagnostics
            ),
            "global_supplemented_local": sum(
                item.get("global_supplemented_local", 0) for item in diagnostics
            ),
            "distance_buckets": {},
        }
        for bucket_name in ("0-31", "32-63", "64-127", ">=128"):
            count = sum(
                item.get("distance_buckets", {}).get(bucket_name, {}).get("count", 0)
                for item in diagnostics
            )
            weight_sum = sum(
                item.get("distance_buckets", {}).get(bucket_name, {}).get("local_weight_sum", 0.0)
                for item in diagnostics
            )
            metrics["fusion_diagnostics"]["distance_buckets"][bucket_name] = {
                "count": count,
                "mean_local_weight": weight_sum / count if count else 0.0,
            }
    evaluation_seconds = time.perf_counter() - evaluation_started
    metrics["run"] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": args.config,
        "split": args.split,
        "device": device,
        "source_documents": len(documents),
        "model_windows": len(manifest),
        "document_re_checkpoint": (
            str(args.document_re_checkpoint.resolve()) if args.document_re_checkpoint else None
        ),
        "relation_threshold": (
            args.relation_threshold
            if args.relation_threshold is not None
            else (document_model.config.threshold if document_model is not None else None)
        ),
        "relation_thresholds_file": (
            str(args.relation_thresholds_file.resolve())
            if args.relation_thresholds_file is not None else None
        ),
        "ner_span_threshold": args.ner_span_threshold,
        "ner_type_threshold": args.ner_type_threshold,
        "ner_overlap_strategy": args.ner_overlap_strategy,
        "ner_joint_rerank_weight": args.ner_joint_rerank_weight,
        "ner_type_margin_weight": args.ner_type_margin_weight,
        "ner_boundary_rerank_weight": args.ner_boundary_rerank_weight,
        "ner_type_rerank_weight": args.ner_type_rerank_weight,
        "ner_boundary_shift_penalty": args.ner_boundary_shift_penalty,
        "ner_type_confusion_penalty": args.ner_type_confusion_penalty,
        "max_document_entities": args.max_document_entities,
        "document_re_fusion_mode": (
            document_model.config.fusion_mode if document_model is not None else None
        ),
        "document_re_extensions": (
            {
                "distance_conditioned_fusion": document_model.config.distance_conditioned_fusion,
                "fusion_long_distance_boundary": document_model.config.fusion_long_distance_boundary,
                "long_distance_head": document_model.config.long_distance_head,
                "long_distance_boundary": document_model.config.long_distance_boundary,
                "distance_experts": document_model.config.distance_experts,
                "distance_expert_boundaries": document_model.config.distance_expert_boundaries,
                "distance_expert_active_buckets": document_model.config.distance_expert_active_buckets,
                "pair_type_compatibility": document_model.config.pair_type_compatibility,
                "pair_link_head": document_model.config.pair_link_head,
                "pair_link_bias": document_model.config.pair_link_bias,
                "pair_link_gate_weight": document_model.config.pair_link_gate_weight,
                "pair_link_positive_weight": document_model.config.pair_link_positive_weight,
                "pair_type_focus_pairs": document_model.config.pair_type_focus_pairs,
                "focus_relation_names": document_model.config.focus_relation_names,
                "focus_relation_positive_weight": document_model.config.focus_relation_positive_weight,
                "relation_adapter": document_model.config.relation_adapter,
                "relation_adapter_size": document_model.config.relation_adapter_size,
                "relation_adapter_weight": document_model.config.relation_adapter_weight,
            }
            if document_model is not None
            else None
        ),
        "local_global_alpha": (
            args.local_global_alpha
            if document_model is not None and document_model.config.fusion_mode == "fixed"
            else None
        ),
        "review_margin": args.review_margin if document_model is not None else None,
        "export_fusion_trace": args.export_fusion_trace,
        "dataset_card": str(
            (PROJECT_ROOT / "datasets" / dataset.data_dir / "dataset_card.json").resolve()
        ),
        "benchmark_output": (
            str(args.benchmark_output.resolve()) if args.benchmark_output is not None else None
        ),
        "benchmark_warmup_runs": args.benchmark_warmup_runs,
        "benchmark_repeats": args.benchmark_repeats,
    }
    output_dir = args.output_dir or PROJECT_ROOT / "artifacts" / "inference" / args.checkpoint.name
    serialization_started = time.perf_counter()
    _write_outputs(documents, metrics, output_dir, args.split)
    serialization_seconds = time.perf_counter() - serialization_started
    if args.benchmark_output is not None:
        stage1_parameters = _parameter_counts(model)
        stage2_parameters = (
            _parameter_counts(document_model)
            if document_model is not None
            else {"total": 0, "trainable": 0}
        )
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        )
        if document_model is not None:
            parameter_bytes += sum(
                parameter.numel() * parameter.element_size()
                for parameter in document_model.parameters()
            )
        hardware = {
            "device": device,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": None,
            "gpu_total_memory_bytes": None,
        }
        if device.startswith("cuda") and torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(torch.device(device))
            hardware["gpu_name"] = properties.name
            hardware["gpu_total_memory_bytes"] = properties.total_memory
        benchmark_payload = {
            "schema_version": 1,
            "method": "LongGeoIE",
            "timing_scope": {
                "main_metric": "pure_inference_seconds",
                "included": (
                    "cached test-batch collation and device transfer, Stage-1 window model and "
                    "decoding, document NER merging/NMS, and Stage-2 document RE"
                ),
                "excluded": (
                    "checkpoint loading, dataset/cache preparation, metric computation, and "
                    "artifact serialization"
                ),
                "cuda_synchronized": device.startswith("cuda"),
            },
            "dataset": {
                "config": args.config,
                "split": args.split,
                "documents": source_limit,
                "windows": len(manifest),
                "batch_size": args.batch_size,
                "max_document_entities": args.max_document_entities,
            },
            "decoding": {
                "ner_span_threshold": args.ner_span_threshold,
                "ner_type_threshold": args.ner_type_threshold,
                "ner_overlap_strategy": args.ner_overlap_strategy,
                "relation_threshold": metrics["run"]["relation_threshold"],
            },
            "parameters": {
                "stage1": stage1_parameters,
                "stage2": stage2_parameters,
                "total_unique": {
                    "total": stage1_parameters["total"] + stage2_parameters["total"],
                    "trainable": stage1_parameters["trainable"] + stage2_parameters["trainable"],
                },
                "in_memory_bytes": parameter_bytes,
            },
            "checkpoint_size_bytes": {
                "stage1_direct_files": _directory_size_bytes(args.checkpoint, recursive=False),
                "stage2": (
                    _directory_size_bytes(args.document_re_checkpoint, recursive=True)
                    if args.document_re_checkpoint is not None
                    else 0
                ),
            },
            "hardware": hardware,
            "warmup_runs": args.benchmark_warmup_runs,
            "benchmark_repeats": args.benchmark_repeats,
            "measured_runs": benchmark_runs,
            "summary": _summarize_benchmark_runs(benchmark_runs),
            "other_timing_seconds": {
                "model_load": model_load_seconds,
                "dataset_prepare": dataset_prepare_seconds,
                "evaluation": evaluation_seconds,
                "artifact_serialization": serialization_seconds,
                "process_through_serialization": time.perf_counter() - process_started,
            },
            "accuracy": {
                "ner_strict_f1": metrics["overall"]["ner_strict"]["f1"],
                "re_strict_end_to_end_f1": metrics["overall"]["re_strict_end_to_end"]["f1"],
            },
            "output_dir": str(output_dir.resolve()),
        }
        args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_output.write_text(
            json.dumps(benchmark_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Benchmark written to: {args.benchmark_output.resolve()}")
    print(json.dumps(metrics["overall"], ensure_ascii=False, indent=2))
    print(f"Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
