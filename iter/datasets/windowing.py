"""Tokenizer-aware overlapping windows for document-level NER/RE datasets.

The raw dataset uses word offsets (``start`` inclusive, ``end`` exclusive).
This module keeps those semantics while producing window-local training labels
and a sidecar manifest that can map predictions back to the source document.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _encoding_length(tokenizer, tokens: list[str]) -> int:
    encoded = tokenizer(tokens, is_split_into_words=True, add_special_tokens=True)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return len(input_ids)


def _largest_fitting_end(tokenizer, tokens: list[str], start: int, max_length: int) -> int:
    """Find the largest exclusive word end whose tokenization fits max_length."""
    low, high = start + 1, len(tokens)
    best = start
    while low <= high:
        mid = (low + high) // 2
        if _encoding_length(tokenizer, tokens[start:mid]) <= max_length:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    if best == start:
        raise ValueError(
            f"A single source token at word offset {start} exceeds max_length={max_length}."
        )
    return best


def build_document_windows(
    item: dict[str, Any],
    tokenizer,
    max_length: int,
    overlap_words: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split one raw document and return (window_items, manifest, coverage).

    Window boundaries depend only on tokens and tokenizer length. Gold entity
    boundaries must never influence the windows because doing so leaks labels
    into evaluation. A relation is attached to a window only when both endpoint
    entities occur in that window. The manifest records global offsets and
    coverage, so entities/relations that no window can observe remain measurable
    instead of silently vanishing.
    """
    tokens = item["tokens"]
    entities = item.get("entities", [])
    relations = item.get("relations", [])
    if max_length < 4:
        raise ValueError("max_length must leave room for content and special tokens")
    if overlap_words < 0:
        raise ValueError("overlap_words must be non-negative")

    windows: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    covered_entity_ids: set[int] = set()
    covered_relation_ids: set[int] = set()
    start = 0

    while start < len(tokens):
        end = _largest_fitting_end(tokenizer, tokens, start, max_length)

        entity_ids = [
            idx for idx, entity in enumerate(entities)
            if start <= entity["start"] and entity["end"] <= end
        ]
        old_to_new = {old: new for new, old in enumerate(entity_ids)}
        window_entities = []
        for entity_id in entity_ids:
            entity = deepcopy(entities[entity_id])
            entity["start"] -= start
            entity["end"] -= start
            window_entities.append(entity)

        relation_ids = [
            idx for idx, relation in enumerate(relations)
            if relation["head"] in old_to_new and relation["tail"] in old_to_new
        ]
        window_relations = []
        for relation_id in relation_ids:
            relation = deepcopy(relations[relation_id])
            relation["head"] = old_to_new[relation["head"]]
            relation["tail"] = old_to_new[relation["tail"]]
            window_relations.append(relation)

        window_item = {
            "tokens": tokens[start:end],
            "entities": window_entities,
            "relations": window_relations,
        }
        windows.append(window_item)
        manifest.append({
            "window_id": len(windows) - 1,
            "word_start": start,
            "word_end": end,
            "word_count": end - start,
            "tokenized_length": _encoding_length(tokenizer, tokens[start:end]),
            "source_entity_ids": entity_ids,
            "source_relation_ids": relation_ids,
        })
        covered_entity_ids.update(entity_ids)
        covered_relation_ids.update(relation_ids)

        if end >= len(tokens):
            break
        next_start = max(start + 1, end - overlap_words)
        start = min(next_start, end)

    coverage = {
        "source_word_count": len(tokens),
        "source_entity_count": len(entities),
        "source_relation_count": len(relations),
        "covered_entity_ids": sorted(covered_entity_ids),
        "covered_relation_ids": sorted(covered_relation_ids),
        "uncovered_entity_ids": sorted(set(range(len(entities))) - covered_entity_ids),
        "uncovered_relation_ids": sorted(set(range(len(relations))) - covered_relation_ids),
    }
    return windows, manifest, coverage
