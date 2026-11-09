"""Build a semantic super-document corpus with relations crossing 512 windows.

The source corpus contains short documents, so simply changing a model window
does not test cross-window relation evidence.  This utility concatenates whole
annotated documents, shifts entity offsets, preserves only their original
relations, and searches packing orders that maximize relations not jointly
observable in a 512-token window with 128-token overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any


SPLITS = ("train", "dev", "test")


def entity_center(entity: dict[str, Any]) -> float:
    return (int(entity["start"]) + int(entity["end"]) - 1) / 2.0


def relation_distance(document: dict[str, Any], relation: dict[str, Any]) -> float:
    entities = document.get("entities", [])
    return abs(
        entity_center(entities[int(relation["head"])])
        - entity_center(entities[int(relation["tail"])])
    )


def window_starts(length: int, window_size: int, overlap: int) -> list[int]:
    if window_size <= 0 or overlap < 0 or overlap >= window_size:
        raise ValueError("window_size must be positive and overlap must be smaller")
    stride = window_size - overlap
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    last = length - window_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def relation_is_window_observable(
    document: dict[str, Any],
    relation: dict[str, Any],
    window_size: int,
    overlap: int,
) -> bool:
    entities = document.get("entities", [])
    head = entities[int(relation["head"])]
    tail = entities[int(relation["tail"])]
    lower = min(int(head["start"]), int(tail["start"]))
    upper = max(int(head["end"]), int(tail["end"]))
    for start in window_starts(len(document.get("tokens", [])), window_size, overlap):
        if start <= lower and upper <= min(start + window_size, len(document["tokens"])):
            return True
    return False


def document_cross_window_stats(
    document: dict[str, Any], window_size: int, overlap: int
) -> dict[str, int]:
    relations = document.get("relations", [])
    unobservable = sum(
        not relation_is_window_observable(document, relation, window_size, overlap)
        for relation in relations
    )
    long_relations = sum(
        relation_distance(document, relation) >= 128 for relation in relations
    )
    return {
        "relations": len(relations),
        "cross_window_relations": unobservable,
        "long_distance_relations": long_relations,
    }


def pack_documents(
    entries: list[tuple[str, int, dict[str, Any]]],
    target_tokens: int,
    max_tokens: int,
    separator_token: str,
) -> list[list[tuple[str, int, dict[str, Any]]]]:
    """Greedily form bins without splitting a source document."""
    if target_tokens < 1 or max_tokens < target_tokens:
        raise ValueError("max_tokens must be at least target_tokens")
    bins: list[list[tuple[str, int, dict[str, Any]]]] = []
    current: list[tuple[str, int, dict[str, Any]]] = []
    current_length = 0
    for entry in entries:
        source_length = len(entry[2].get("tokens", []))
        if source_length > max_tokens:
            raise ValueError(
                f"Source document {entry[0]}:{entry[1]} has {source_length} tokens, "
                f"which exceeds max_tokens={max_tokens}"
            )
        added_length = source_length + (1 if current else 0)
        if current and current_length + added_length > max_tokens:
            bins.append(current)
            current = []
            current_length = 0
            added_length = source_length
        current.append(entry)
        current_length += added_length
        if current_length >= target_tokens:
            bins.append(current)
            current = []
            current_length = 0
    if current:
        if bins and current_length < target_tokens:
            previous_length = sum(
                len(item[2].get("tokens", [])) for item in bins[-1]
            ) + max(0, len(bins[-1]) - 1)
            if previous_length + 1 + current_length <= max_tokens:
                bins[-1].extend(current)
                current = []
        if current:
            bins.append(current)
    return bins


def merge_bin(
    entries: list[tuple[str, int, dict[str, Any]]], separator_token: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tokens: list[str] = []
    entities: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for source_split, source_index, source in entries:
        if tokens:
            tokens.append(separator_token)
        offset = len(tokens)
        entity_offset = len(entities)
        source_document_id = f"{source_split}:{source_index}"
        for entity in source.get("entities", []):
            shifted = dict(entity)
            shifted["start"] = int(entity["start"]) + offset
            shifted["end"] = int(entity["end"]) + offset
            # Keep provenance separate from the entity label.  Stage-2 must
            # not turn entities from two concatenated source documents into
            # artificial negative relation pairs.
            shifted["source_document_id"] = source_document_id
            entities.append(shifted)
        for relation in source.get("relations", []):
            shifted_relation = dict(relation)
            shifted_relation["head"] = int(relation["head"]) + entity_offset
            shifted_relation["tail"] = int(relation["tail"]) + entity_offset
            relations.append(shifted_relation)
        tokens.extend(source.get("tokens", []))
        manifest.append(
            {
                "source_split": source_split,
                "source_index": source_index,
                "source_document_id": source_document_id,
                "word_start": offset,
                "word_end": offset + len(source.get("tokens", [])),
                "source_tokens": len(source.get("tokens", [])),
                "source_entities": len(source.get("entities", [])),
                "source_relations": len(source.get("relations", [])),
            }
        )
    return {"tokens": tokens, "entities": entities, "relations": relations}, manifest


def optimize_bin(
    entries: list[tuple[str, int, dict[str, Any]]],
    separator_token: str,
    window_size: int,
    overlap: int,
    attempts: int,
    rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    best_document: dict[str, Any] | None = None
    best_manifest: list[dict[str, Any]] | None = None
    best_score = (-1, -1, -1)
    candidates = [list(entries)]
    if len(entries) > 1:
        candidates.append(list(reversed(entries)))
    for _ in range(max(0, attempts - len(candidates))):
        candidate = list(entries)
        rng.shuffle(candidate)
        candidates.append(candidate)
    for candidate in candidates:
        document, manifest = merge_bin(candidate, separator_token)
        stats = document_cross_window_stats(document, window_size, overlap)
        score = (
            stats["cross_window_relations"],
            stats["long_distance_relations"],
            len(document.get("tokens", [])),
        )
        if score > best_score:
            best_score = score
            best_document = document
            best_manifest = manifest
    assert best_document is not None and best_manifest is not None
    return best_document, best_manifest


def load_split(source_dir: Path, split: str) -> list[dict[str, Any]]:
    path = source_dir / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def write_config(
    config_root: Path,
    config_name: str,
    dataset_name: str,
    data_dir: str,
    base_config: str,
) -> Path:
    config_root.mkdir(parents=True, exist_ok=True)
    path = config_root / f"{config_name}.json"
    path.write_text(
        json.dumps(
            {
                "base": base_config,
                "dataset": {
                    "name": dataset_name,
                    "data_dir": data_dir,
                    "reinitialize_dataset": False,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("datasets/ind_distance_gt28_rebalanced"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/ind_distance_gt28_rebalanced_cross_window"))
    parser.add_argument("--config-root", type=Path, default=Path("cfg/wenyanwen"))
    parser.add_argument("--config-name", default="wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window")
    parser.add_argument("--dataset-name", default="wenyanwen_long_context_type_aux_ind_distance_gt28_rebalanced_cross_window")
    parser.add_argument("--base-config", default="wenyanwen/wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced")
    parser.add_argument("--target-tokens", type=int, default=768)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--window-overlap", type=int, default=128)
    parser.add_argument("--separator-token", default="。")
    parser.add_argument("--attempts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    if args.window_overlap >= args.window_size:
        parser.error("--window-overlap must be smaller than --window-size")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    types_path = args.source_dir / "types.json"
    if not types_path.exists():
        raise FileNotFoundError(types_path)
    shutil.copy2(types_path, args.output_dir / "types.json")
    rng = random.Random(args.seed)
    dataset_stats: dict[str, Any] = {}
    manifest_lines: list[str] = []

    for split in SPLITS:
        source_documents = load_split(args.source_dir, split)
        entries = [(split, index, document) for index, document in enumerate(source_documents)]
        rng.shuffle(entries)
        bins = pack_documents(entries, args.target_tokens, args.max_tokens, args.separator_token)
        output_documents: list[dict[str, Any]] = []
        cross_relations = 0
        long_relations = 0
        unobservable_relations = 0
        lengths: list[int] = []
        for document_id, entries_in_bin in enumerate(bins):
            document, source_manifest = optimize_bin(
                entries_in_bin,
                args.separator_token,
                args.window_size,
                args.window_overlap,
                args.attempts,
                rng,
            )
            stats = document_cross_window_stats(
                document, args.window_size, args.window_overlap
            )
            document["source_documents"] = source_manifest
            output_documents.append(document)
            lengths.append(len(document["tokens"]))
            cross_relations += stats["cross_window_relations"]
            long_relations += stats["long_distance_relations"]
            unobservable_relations += stats["cross_window_relations"]
            manifest_lines.append(
                json.dumps(
                    {
                        "split": split,
                        "document_id": document_id,
                        "token_count": len(document["tokens"]),
                        "source_documents": source_manifest,
                        **stats,
                    },
                    ensure_ascii=False,
                )
            )
        (args.output_dir / f"{split}.json").write_text(
            json.dumps(output_documents, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        dataset_stats[split] = {
            "source_documents": len(source_documents),
            "documents": len(output_documents),
            "min_tokens": min(lengths, default=0),
            "median_tokens": sorted(lengths)[len(lengths) // 2] if lengths else 0,
            "max_tokens": max(lengths, default=0),
            "relations": sum(len(document.get("relations", [])) for document in output_documents),
            "cross_window_relations": cross_relations,
            "long_distance_relations_ge128": long_relations,
            "unobservable_relations": unobservable_relations,
            "documents_with_cross_window_relations": sum(
                document_cross_window_stats(document, args.window_size, args.window_overlap)[
                    "cross_window_relations"
                ]
                > 0
                for document in output_documents
            ),
        }

    (args.output_dir / "document_manifest.jsonl").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )
    card = {
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "construction": {
            "rule": "concatenate whole annotated source documents without adding cross-source relations",
            "target_tokens": args.target_tokens,
            "max_tokens": args.max_tokens,
            "separator_token": args.separator_token,
            "window_size": args.window_size,
            "window_overlap": args.window_overlap,
            "cross_window_definition": "no 512-token window with 128-token overlap contains both relation endpoints",
        },
        "splits": dataset_stats,
    }
    (args.output_dir / "dataset_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    config_path = write_config(
        args.config_root,
        args.config_name,
        args.dataset_name,
        args.output_dir.name,
        args.base_config,
    )
    print(json.dumps({"dataset": str(args.output_dir), "config": str(config_path), "splits": dataset_stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
