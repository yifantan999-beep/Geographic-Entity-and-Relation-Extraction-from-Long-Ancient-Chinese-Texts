"""Freeze a train-matched 50/50 dev/test repartition of an existing holdout.

The training split is copied without modification.  Existing dev and test
documents are pooled and repartitioned using annotation statistics only; no
model predictions or evaluation scores are read.  The selected test half is
optimized to resemble train in entity types, relation types, relation-distance
buckets, length, density, and cross-window relation coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


DISTANCE_BUCKETS = ("0-31", "32-63", "64-127", ">=128")
LENGTH_BUCKETS = ("0-511", "512-767", "768-895", ">=896")
WINDOW_SIZE = 512
WINDOW_OVERLAP = 128


def canonical_payload(document: dict[str, Any]) -> bytes:
    return json.dumps(
        {
            "tokens": document.get("tokens", []),
            "entities": document.get("entities", []),
            "relations": document.get("relations", []),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def document_hash(document: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_payload(document)).hexdigest()


def distance_bucket(distance: float) -> str:
    if distance < 32:
        return "0-31"
    if distance < 64:
        return "32-63"
    if distance < 128:
        return "64-127"
    return ">=128"


def length_bucket(length: int) -> str:
    if length < 512:
        return "0-511"
    if length < 768:
        return "512-767"
    if length < 896:
        return "768-895"
    return ">=896"


def entity_center(entity: dict[str, Any]) -> float:
    return (int(entity["start"]) + int(entity["end"]) - 1) / 2.0


def window_starts(length: int) -> list[int]:
    if length <= WINDOW_SIZE:
        return [0]
    stride = WINDOW_SIZE - WINDOW_OVERLAP
    starts = list(range(0, length - WINDOW_SIZE + 1, stride))
    final_start = length - WINDOW_SIZE
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def relation_observable(document: dict[str, Any], relation: dict[str, Any]) -> bool:
    entities = document["entities"]
    head = entities[int(relation["head"])]
    tail = entities[int(relation["tail"])]
    lower = min(int(head["start"]), int(tail["start"]))
    upper = max(int(head["end"]), int(tail["end"]))
    length = len(document["tokens"])
    return any(
        start <= lower and upper <= min(start + WINDOW_SIZE, length)
        for start in window_starts(length)
    )


def validate_document(document: dict[str, Any], label: str) -> None:
    tokens = document.get("tokens", [])
    entities = document.get("entities", [])
    for entity_index, entity in enumerate(entities):
        start, end = int(entity["start"]), int(entity["end"])
        if not 0 <= start < end <= len(tokens):
            raise ValueError(f"{label}: invalid entity {entity_index}: {entity}")
    for relation_index, relation in enumerate(document.get("relations", [])):
        head, tail = int(relation["head"]), int(relation["tail"])
        if not 0 <= head < len(entities) or not 0 <= tail < len(entities):
            raise ValueError(f"{label}: invalid relation {relation_index}: {relation}")


def document_features(document: dict[str, Any]) -> tuple[Counter[str], dict[str, float]]:
    features: Counter[str] = Counter()
    entities = document.get("entities", [])
    relations = document.get("relations", [])
    for entity in entities:
        features[f"entity:{entity['type']}"] += 1

    distances: list[float] = []
    cross_window = 0
    for relation in relations:
        features[f"relation:{relation['type']}"] += 1
        head = entities[int(relation["head"])]
        tail = entities[int(relation["tail"])]
        distance = abs(entity_center(head) - entity_center(tail))
        distances.append(distance)
        features[f"distance:{distance_bucket(distance)}"] += 1
        if not relation_observable(document, relation):
            cross_window += 1

    features[f"length:{length_bucket(len(document.get('tokens', [])))}"] += 1
    features[f"cross_window:{'yes' if cross_window else 'no'}"] += 1
    max_distance = max(distances, default=0.0)
    features[f"max_distance:{distance_bucket(max_distance)}"] += 1
    scalars = {
        "tokens": float(len(document.get("tokens", []))),
        "entities": float(len(entities)),
        "relations": float(len(relations)),
        "cross_window_relations": float(cross_window),
        "long_distance_relations": float(sum(distance >= 128 for distance in distances)),
    }
    return features, scalars


def total_variation(left: Counter[str], right: Counter[str], prefix: str) -> float:
    keys = {key for key in left.keys() | right.keys() if key.startswith(prefix)}
    left_total = sum(left[key] for key in keys)
    right_total = sum(right[key] for key in keys)
    if not left_total or not right_total:
        return 0.0 if left_total == right_total else 1.0
    return 0.5 * sum(
        abs(left[key] / left_total - right[key] / right_total) for key in keys
    )


def mean_scalars(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals: Counter[str] = Counter()
    for row in rows:
        totals.update(row["scalars"])
    return {key: totals[key] / max(len(rows), 1) for key in totals}


def selection_objective(
    train_features: Counter[str],
    train_scalars: dict[str, float],
    records: list[dict[str, Any]],
    selected: set[int],
) -> tuple[float, dict[str, float]]:
    test_features: Counter[str] = Counter()
    selected_rows = [records[index] for index in sorted(selected)]
    for record in selected_rows:
        test_features.update(record["features"])
    test_scalars = mean_scalars(selected_rows)

    components = {
        "entity_type_tv": total_variation(train_features, test_features, "entity:"),
        "relation_type_tv": total_variation(train_features, test_features, "relation:"),
        "relation_distance_tv": total_variation(train_features, test_features, "distance:"),
        "max_distance_tv": total_variation(train_features, test_features, "max_distance:"),
        "length_tv": total_variation(train_features, test_features, "length:"),
        "cross_window_document_tv": total_variation(
            train_features, test_features, "cross_window:"
        ),
    }
    scalar_error = 0.0
    for key, train_value in train_scalars.items():
        scalar_error += abs(test_scalars.get(key, 0.0) - train_value) / max(train_value, 1.0)
    scalar_error /= max(len(train_scalars), 1)
    components["document_density_error"] = scalar_error

    weights = {
        "entity_type_tv": 3.0,
        "relation_type_tv": 4.0,
        "relation_distance_tv": 2.0,
        "max_distance_tv": 0.8,
        "length_tv": 1.2,
        "cross_window_document_tv": 1.5,
        "document_density_error": 1.2,
    }
    objective = sum(weights[key] * value for key, value in components.items())
    components["weighted_objective"] = objective
    return objective, components


def optimize_test_half(
    records: list[dict[str, Any]],
    train_features: Counter[str],
    train_scalars: dict[str, float],
    test_size: int,
    seed: int,
    restarts: int,
) -> tuple[set[int], float, dict[str, float]]:
    rng = random.Random(seed)
    all_indices = set(range(len(records)))
    best_selection: set[int] | None = None
    best_score = math.inf
    best_components: dict[str, float] = {}

    for _ in range(restarts):
        selected = set(rng.sample(sorted(all_indices), test_size))
        score, components = selection_objective(
            train_features, train_scalars, records, selected
        )
        improved = True
        while improved:
            improved = False
            candidates: list[tuple[float, int, int, dict[str, float]]] = []
            for outgoing in sorted(selected):
                for incoming in sorted(all_indices - selected):
                    proposal = (selected - {outgoing}) | {incoming}
                    proposal_score, proposal_components = selection_objective(
                        train_features, train_scalars, records, proposal
                    )
                    candidates.append(
                        (proposal_score, outgoing, incoming, proposal_components)
                    )
            proposal_score, outgoing, incoming, proposal_components = min(
                candidates, key=lambda item: (item[0], item[1], item[2])
            )
            if proposal_score < score - 1e-12:
                selected.remove(outgoing)
                selected.add(incoming)
                score = proposal_score
                components = proposal_components
                improved = True

        if score < best_score - 1e-12:
            best_selection = set(selected)
            best_score = score
            best_components = components

    if best_selection is None:
        raise RuntimeError("No holdout assignment was generated")
    return best_selection, best_score, best_components


def aggregate(documents: list[dict[str, Any]]) -> dict[str, Any]:
    features: Counter[str] = Counter()
    scalar_rows = []
    lengths = []
    for document in documents:
        doc_features, scalars = document_features(document)
        features.update(doc_features)
        scalar_rows.append({"scalars": scalars})
        lengths.append(len(document.get("tokens", [])))
    return {
        "documents": len(documents),
        "tokens": int(sum(lengths)),
        "min_tokens": min(lengths, default=0),
        "median_tokens": median(lengths) if lengths else 0,
        "max_tokens": max(lengths, default=0),
        "entities": int(sum(len(row.get("entities", [])) for row in documents)),
        "relations": int(sum(len(row.get("relations", [])) for row in documents)),
        "cross_window_relations": int(
            sum(row["scalars"]["cross_window_relations"] for row in scalar_rows)
        ),
        "long_distance_relations_ge128": int(
            sum(row["scalars"]["long_distance_relations"] for row in scalar_rows)
        ),
        "entity_type_counts": dict(
            sorted((key.removeprefix("entity:"), value) for key, value in features.items() if key.startswith("entity:"))
        ),
        "relation_type_counts": dict(
            sorted((key.removeprefix("relation:"), value) for key, value in features.items() if key.startswith("relation:"))
        ),
        "relation_distance_buckets": {
            bucket: features[f"distance:{bucket}"] for bucket in DISTANCE_BUCKETS
        },
        "length_buckets": {
            bucket: features[f"length:{bucket}"] for bucket in LENGTH_BUCKETS
        },
    }


def dataset_fingerprint(output_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("train.json", "dev.json", "test.json", "types.json"):
        path = output_dir / name
        digest.update(name.encode("utf-8"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        digest.update(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("datasets/ind_distance_gt28_rebalanced_cross_window_v3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/ind_distance_gt28_rebalanced_cross_window_train_matched"),
    )
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--restarts", type=int, default=64)
    args = parser.parse_args()

    train = json.loads((args.input_dir / "train.json").read_text(encoding="utf-8"))
    holdout: list[dict[str, Any]] = []
    for split in ("dev", "test"):
        rows = json.loads((args.input_dir / f"{split}.json").read_text(encoding="utf-8"))
        for index, document in enumerate(rows):
            validate_document(document, f"{split}[{index}]")
            features, scalars = document_features(document)
            holdout.append(
                {
                    "document": document,
                    "source_split": split,
                    "source_index": index,
                    "hash": document_hash(document),
                    "features": features,
                    "scalars": scalars,
                }
            )

    train_hashes = {document_hash(document) for document in train}
    holdout_hashes = [record["hash"] for record in holdout]
    if len(set(holdout_hashes)) != len(holdout_hashes):
        raise ValueError("Duplicate documents exist in pooled dev/test")
    if train_hashes & set(holdout_hashes):
        raise ValueError("Train and pooled holdout overlap")
    if len(holdout) % 2:
        raise ValueError("Pooled holdout must contain an even number of documents")

    train_features: Counter[str] = Counter()
    train_scalar_rows = []
    for document in train:
        validate_document(document, "train")
        features, scalars = document_features(document)
        train_features.update(features)
        train_scalar_rows.append({"scalars": scalars})
    train_scalars = mean_scalars(train_scalar_rows)
    test_indices, score, components = optimize_test_half(
        holdout,
        train_features,
        train_scalars,
        len(holdout) // 2,
        args.seed,
        args.restarts,
    )
    dev_indices = set(range(len(holdout))) - test_indices

    assignments = {"dev": sorted(dev_indices), "test": sorted(test_indices)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.input_dir / "train.json", args.output_dir / "train.json")
    shutil.copy2(args.input_dir / "types.json", args.output_dir / "types.json")
    for split in ("dev", "test"):
        documents = [holdout[index]["document"] for index in assignments[split]]
        (args.output_dir / f"{split}.json").write_text(
            json.dumps(documents, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest_path = args.output_dir / "document_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for document_id, document in enumerate(train):
            handle.write(
                json.dumps(
                    {
                        "split": "train",
                        "document_id": document_id,
                        "document_hash": document_hash(document),
                        "source_holdout": None,
                        "source_documents": document.get("source_documents", []),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        for split in ("dev", "test"):
            for document_id, index in enumerate(assignments[split]):
                record = holdout[index]
                handle.write(
                    json.dumps(
                        {
                            "split": split,
                            "document_id": document_id,
                            "document_hash": record["hash"],
                            "source_holdout": {
                                "split": record["source_split"],
                                "index": record["source_index"],
                            },
                            "source_documents": record["document"].get("source_documents", []),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    card = {
        "source_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "policy": {
            "train": "copied byte-for-byte from source train.json",
            "holdout": "pool source dev/test and repartition into exactly equal halves",
            "selection_inputs": "gold annotation statistics only; no predictions or model metrics",
            "test_target": "minimize distribution distance from frozen train",
            "development_protocol": "select models and thresholds on dev; evaluate frozen test only for final confirmation",
        },
        "optimization": {
            "restarts": args.restarts,
            "objective": score,
            "components": components,
        },
        "splits": {
            "train": aggregate(train),
            "dev": aggregate([holdout[index]["document"] for index in assignments["dev"]]),
            "test": aggregate([holdout[index]["document"] for index in assignments["test"]]),
        },
        "assignment": {
            split: [
                {
                    "source_split": holdout[index]["source_split"],
                    "source_index": holdout[index]["source_index"],
                    "document_hash": holdout[index]["hash"],
                }
                for index in assignments[split]
            ]
            for split in ("dev", "test")
        },
    }
    (args.output_dir / "dataset_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    card["dataset_fingerprint_sha256"] = dataset_fingerprint(args.output_dir)
    (args.output_dir / "dataset_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(card, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
