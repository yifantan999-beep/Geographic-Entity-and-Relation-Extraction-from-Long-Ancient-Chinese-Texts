"""Repartition distance-filtered corpora with document-level multi-label stratification.

The input datasets already contain only documents whose maximum gold relation
distance exceeds a threshold.  This script merges their existing train/dev/test
files, removes exact duplicate documents, and assigns the unique documents to
new 80/10/10 train/dev/test splits.  Assignment preserves entity-type counts,
relation-type counts, relation-distance buckets, and coarse token-length
bands as closely as possible.  A document is never split across partitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


SPLITS = ("train", "dev", "test")
DISTANCE_BUCKETS = ("0-31", "32-63", "64-127", ">=128")
LENGTH_BUCKETS = ("0-63", "64-127", "128-255", "256-511", ">=512")
FAMILY_WEIGHTS = {
    "entity": 1.0,
    "relation": 1.45,
    "distance": 1.8,
    "max_distance": 2.4,
    "length": 0.45,
}


def canonical_payload(row: dict[str, Any]) -> bytes:
    payload = {
        "tokens": row.get("tokens", []),
        "entities": row.get("entities", []),
        "relations": row.get("relations", []),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def document_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_payload(row)).hexdigest()


def entity_center(entity: dict[str, Any]) -> int:
    return (int(entity["start"]) + int(entity["end"])) // 2


def relation_distances(row: dict[str, Any]) -> list[int]:
    entities = row.get("entities", [])
    distances: list[int] = []
    for relation in row.get("relations", []):
        head = int(relation["head"])
        tail = int(relation["tail"])
        if not (0 <= head < len(entities) and 0 <= tail < len(entities)):
            raise ValueError(f"Invalid relation endpoint: {relation}")
        distances.append(abs(entity_center(entities[head]) - entity_center(entities[tail])))
    return distances


def distance_bucket(distance: int) -> str:
    if distance <= 31:
        return "0-31"
    if distance <= 63:
        return "32-63"
    if distance <= 127:
        return "64-127"
    return ">=128"


def length_bucket(length: int) -> str:
    if length <= 63:
        return "0-63"
    if length <= 127:
        return "64-127"
    if length <= 255:
        return "128-255"
    if length <= 511:
        return "256-511"
    return ">=512"


def feature_family(label: str) -> str:
    return label.split(":", 1)[0]


def make_record(row: dict[str, Any], origins: list[dict[str, Any]], digest: str) -> dict[str, Any]:
    distances = relation_distances(row)
    max_distance = max(distances) if distances else None
    labels: set[str] = set()
    values: Counter[str] = Counter()

    for entity in row.get("entities", []):
        label = f"entity:{entity['type']}"
        labels.add(label)
        values[label] += 1
    for relation, distance in zip(row.get("relations", []), distances):
        relation_label = f"relation:{relation['type']}"
        labels.add(relation_label)
        values[relation_label] += 1
        distance_label = f"distance:{distance_bucket(distance)}"
        labels.add(distance_label)
        values[distance_label] += 1

    max_bucket = distance_bucket(max_distance) if max_distance is not None else "none"
    max_label = f"max_distance:{max_bucket}"
    labels.add(max_label)
    values[max_label] += 1

    token_length = len(row.get("tokens", []))
    length_label = f"length:{length_bucket(token_length)}"
    labels.add(length_label)
    values[length_label] += 1

    return {
        "row": row,
        "origins": origins,
        "hash": digest,
        "distances": distances,
        "max_distance": max_distance,
        "token_length": token_length,
        "labels": labels,
        "values": values,
    }


def load_unique_records(input_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    raw_counts: dict[str, int] = {}
    for source_split in SPLITS:
        path = input_dir / f"{source_split}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = json.loads(path.read_text(encoding="utf-8"))
        raw_counts[source_split] = len(rows)
        for source_index, row in enumerate(rows):
            digest = document_hash(row)
            if digest not in grouped:
                grouped[digest] = {"row": row, "origins": []}
            grouped[digest]["origins"].append(
                {"split": source_split, "index": source_index}
            )

    records = [
        make_record(item["row"], item["origins"], digest)
        for digest, item in sorted(grouped.items())
    ]
    stats = {
        "raw_documents_by_split": raw_counts,
        "raw_documents": sum(raw_counts.values()),
        "unique_documents": len(records),
        "duplicate_rows_removed": sum(len(record["origins"]) - 1 for record in records),
        "duplicate_groups": sum(len(record["origins"]) > 1 for record in records),
    }
    return records, stats


def target_counts(total: int, train_ratio: float, dev_ratio: float) -> dict[str, int]:
    train = int(round(total * train_ratio))
    dev = int(round(total * dev_ratio))
    train = max(1, min(total - 2, train))
    dev = max(1, min(total - train - 1, dev))
    return {"train": train, "dev": dev, "test": total - train - dev}


def global_feature_totals(records: list[dict[str, Any]]) -> Counter[str]:
    totals: Counter[str] = Counter()
    for record in records:
        totals.update(record["values"])
    return totals


def assignment_objective(
    assignments: dict[str, list[int]],
    records: list[dict[str, Any]],
    targets: dict[str, int],
    feature_totals: Counter[str],
) -> float:
    feature_counts = feature_counts_for_assignment(assignments, records)
    return objective_from_feature_counts(
        feature_counts,
        {split: len(assignments[split]) for split in SPLITS},
        records,
        targets,
        feature_totals,
    )


def feature_counts_for_assignment(
    assignments: dict[str, list[int]], records: list[dict[str, Any]]
) -> dict[str, Counter[str]]:
    feature_counts = {split: Counter() for split in SPLITS}
    for split in SPLITS:
        for index in assignments[split]:
            feature_counts[split].update(records[index]["values"])
    return feature_counts


def objective_from_feature_counts(
    feature_counts: dict[str, Counter[str]],
    counts: dict[str, int],
    records: list[dict[str, Any]],
    targets: dict[str, int],
    feature_totals: Counter[str],
) -> float:
    total = len(records)
    objective = 0.0
    for split in SPLITS:
        target = targets[split]
        objective += 5.0 * ((counts[split] - target) / max(target, 1)) ** 2

    for label, total_value in feature_totals.items():
        family = feature_family(label)
        weight = FAMILY_WEIGHTS.get(family, 1.0)
        for split in SPLITS:
            target_value = total_value * targets[split] / max(total, 1)
            actual = feature_counts[split][label]
            objective += weight * ((actual - target_value) / max(target_value, 1.0)) ** 2
    return objective


def improve_by_swaps(
    assignments: dict[str, list[int]],
    records: list[dict[str, Any]],
    targets: dict[str, int],
    feature_totals: Counter[str],
    seed: int,
    attempts: int,
) -> float:
    """Improve a fixed-size assignment by accepting beneficial document swaps."""
    rng = random.Random(seed)
    feature_counts = feature_counts_for_assignment(assignments, records)
    counts = {split: len(assignments[split]) for split in SPLITS}
    current = objective_from_feature_counts(
        feature_counts, counts, records, targets, feature_totals
    )
    split_pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))
    for _ in range(max(0, attempts)):
        left, right = split_pairs[rng.randrange(len(split_pairs))]
        if not assignments[left] or not assignments[right]:
            continue
        left_pos = rng.randrange(len(assignments[left]))
        right_pos = rng.randrange(len(assignments[right]))
        left_index = assignments[left][left_pos]
        right_index = assignments[right][right_pos]
        affected = set(records[left_index]["values"]) | set(records[right_index]["values"])
        delta = 0.0
        for label in affected:
            total_value = feature_totals[label]
            weight = FAMILY_WEIGHTS.get(feature_family(label), 1.0)
            for split, before, after in (
                (
                    left,
                    feature_counts[left][label],
                    feature_counts[left][label]
                    - records[left_index]["values"].get(label, 0)
                    + records[right_index]["values"].get(label, 0),
                ),
                (
                    right,
                    feature_counts[right][label],
                    feature_counts[right][label]
                    - records[right_index]["values"].get(label, 0)
                    + records[left_index]["values"].get(label, 0),
                ),
            ):
                target_value = total_value * targets[split] / max(len(records), 1)
                delta += weight * (
                    ((after - target_value) / max(target_value, 1.0)) ** 2
                    - ((before - target_value) / max(target_value, 1.0)) ** 2
                )
        if delta < -1e-12:
            assignments[left][left_pos], assignments[right][right_pos] = (
                right_index,
                left_index,
            )
            for label in affected:
                left_value = records[left_index]["values"].get(label, 0)
                right_value = records[right_index]["values"].get(label, 0)
                feature_counts[left][label] += right_value - left_value
                feature_counts[right][label] += left_value - right_value
            current += delta
    return current


def greedy_assignment(
    records: list[dict[str, Any]],
    targets: dict[str, int],
    feature_totals: Counter[str],
    seed: int,
) -> dict[str, list[int]]:
    rng = random.Random(seed)
    label_doc_frequency = Counter()
    for record in records:
        label_doc_frequency.update(record["labels"])

    tie_break = {index: rng.random() for index in range(len(records))}
    order = sorted(
        range(len(records)),
        key=lambda index: (
            -sum(
                FAMILY_WEIGHTS.get(feature_family(label), 1.0)
                / max(label_doc_frequency[label], 1)
                for label in records[index]["labels"]
            ),
            -len(records[index]["labels"]),
            -records[index]["token_length"],
            tie_break[index],
        ),
    )

    assignments = {split: [] for split in SPLITS}
    feature_counts = {split: Counter() for split in SPLITS}
    total = len(records)

    for index in order:
        record = records[index]
        candidates = [
            split for split in SPLITS if len(assignments[split]) < targets[split]
        ]
        if not candidates:
            raise RuntimeError("No split has remaining capacity")

        def candidate_cost(split: str) -> tuple[float, float, str]:
            count_after = len(assignments[split]) + 1
            size_cost = ((count_after - targets[split]) / max(targets[split], 1)) ** 2
            label_cost = 0.0
            for label, value in record["values"].items():
                target_value = feature_totals[label] * targets[split] / max(total, 1)
                actual_after = feature_counts[split][label] + value
                weight = FAMILY_WEIGHTS.get(feature_family(label), 1.0)
                label_cost += weight * (
                    (actual_after - target_value) / max(target_value, 1.0)
                ) ** 2
            # A small count-deficit term prevents rare-label ordering from
            # filling the validation/test buckets too early.
            count_deficit = (targets[split] - len(assignments[split])) / max(targets[split], 1)
            return label_cost + 0.35 * size_cost - 0.15 * count_deficit, rng.random(), split

        destination = min(candidates, key=candidate_cost)
        assignments[destination].append(index)
        feature_counts[destination].update(record["values"])

    return assignments


def ensure_rare_label_coverage(
    assignments: dict[str, list[int]], records: list[dict[str, Any]], feature_totals: Counter[str]
) -> None:
    """Put labels with at least three documents in every split when feasible."""
    label_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        for label in record["labels"]:
            label_indices[label].append(index)

    location: dict[int, str] = {
        index: split for split in SPLITS for index in assignments[split]
    }
    for label, indices in sorted(label_indices.items(), key=lambda item: (len(item[1]), item[0])):
        if feature_family(label) not in {"entity", "relation"} or len(indices) < 3:
            continue
        for destination in SPLITS:
            if any(index in assignments[destination] for index in indices):
                continue
            donors = [
                index
                for index in indices
                if location[index] != destination
                and sum(label in records[item]["labels"] for item in assignments[location[index]]) > 1
            ]
            if not donors:
                continue
            best_swap: tuple[float, str, int, int] | None = None
            for donor in donors:
                donor_split = location[donor]
                for recipient_pos, recipient in enumerate(assignments[destination]):
                    # Match coarse difficulty first, then sentence length, so
                    # the coverage repair does not alter the target profile.
                    distance_penalty = int(
                        distance_bucket(records[donor]["max_distance"] or 0)
                        != distance_bucket(records[recipient]["max_distance"] or 0)
                    )
                    length_penalty = int(
                        length_bucket(records[donor]["token_length"])
                        != length_bucket(records[recipient]["token_length"])
                    )
                    score = distance_penalty * 10.0 + length_penalty + abs(
                        records[donor]["token_length"] - records[recipient]["token_length"]
                    ) / 512.0
                    candidate = (score, donor_split, donor, recipient_pos)
                    if best_swap is None or candidate < best_swap:
                        best_swap = candidate
            if best_swap is None:
                continue
            _, donor_split, donor, recipient_pos = best_swap
            recipient = assignments[destination][recipient_pos]
            donor_pos = assignments[donor_split].index(donor)
            assignments[donor_split][donor_pos], assignments[destination][recipient_pos] = (
                recipient,
                donor,
            )
            location[donor] = destination
            location[recipient] = donor_split


def repartition(
    records: list[dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
    attempts: int,
    swap_attempts: int,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    targets = target_counts(len(records), train_ratio, dev_ratio)
    totals = global_feature_totals(records)
    best: dict[str, list[int]] | None = None
    best_score = math.inf
    for attempt in range(max(1, attempts)):
        candidate = greedy_assignment(records, targets, totals, seed + attempt * 1009)
        ensure_rare_label_coverage(candidate, records, totals)
        score = improve_by_swaps(
            candidate,
            records,
            targets,
            totals,
            seed + attempt * 1009 + 17,
            swap_attempts,
        )
        # The local swap pass can move a singleton document back out of a
        # small validation/test bucket.  Run the deterministic coverage repair
        # once more after swaps so every feasible rare entity/relation label is
        # represented in all three splits.
        ensure_rare_label_coverage(candidate, records, totals)
        score = assignment_objective(candidate, records, targets, totals)
        if score < best_score:
            best = candidate
            best_score = score
    assert best is not None
    return best, {
        "target_documents": targets,
        "objective": best_score,
        "attempts": max(1, attempts),
        "seed": seed,
    }


def aggregate_stats(records: list[dict[str, Any]], indices: list[int]) -> dict[str, Any]:
    rows = [records[index] for index in indices]
    entity_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    relation_distance_counts: Counter[str] = Counter()
    max_distance_counts: Counter[str] = Counter()
    length_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    all_distances: list[int] = []
    for record in rows:
        row = record["row"]
        entity_counts.update(str(entity["type"]) for entity in row.get("entities", []))
        relation_counts.update(str(relation["type"]) for relation in row.get("relations", []))
        relation_distance_counts.update(distance_bucket(distance) for distance in record["distances"])
        max_bucket = (
            distance_bucket(record["max_distance"])
            if record["max_distance"] is not None
            else "none"
        )
        max_distance_counts[max_bucket] += 1
        length_counts[length_bucket(record["token_length"])] += 1
        source_counts.update(origin["split"] for origin in record["origins"])
        all_distances.extend(record["distances"])
    sorted_distances = sorted(all_distances)
    return {
        "documents": len(rows),
        "token_count": sum(record["token_length"] for record in rows),
        "entities": sum(len(record["row"].get("entities", [])) for record in rows),
        "relations": sum(len(record["row"].get("relations", [])) for record in rows),
        "token_length": {
            "min": min((record["token_length"] for record in rows), default=0),
            "median": median(record["token_length"] for record in rows) if rows else 0,
            "max": max((record["token_length"] for record in rows), default=0),
            "buckets": dict((bucket, length_counts[bucket]) for bucket in LENGTH_BUCKETS),
        },
        "max_relation_distance": {
            "min": min((record["max_distance"] for record in rows if record["max_distance"] is not None), default=None),
            "median": median(record["max_distance"] for record in rows if record["max_distance"] is not None)
            if any(record["max_distance"] is not None for record in rows)
            else None,
            "max": max((record["max_distance"] for record in rows if record["max_distance"] is not None), default=None),
            "buckets": dict((bucket, max_distance_counts[bucket]) for bucket in DISTANCE_BUCKETS),
        },
        "relation_distance_buckets": dict(
            (bucket, relation_distance_counts[bucket]) for bucket in DISTANCE_BUCKETS
        ),
        "relation_distance_stats": {
            "min": min(sorted_distances) if sorted_distances else None,
            "median": median(sorted_distances) if sorted_distances else None,
            "max": max(sorted_distances) if sorted_distances else None,
        },
        "entity_type_counts": dict(sorted(entity_counts.items())),
        "relation_type_counts": dict(sorted(relation_counts.items())),
        "source_origin_counts": dict(sorted(source_counts.items())),
    }


def distribution_error(split_stats: dict[str, Any], global_stats: dict[str, Any], ratio: float) -> dict[str, float]:
    def error(key: str, nested: str | None = None) -> float:
        left = split_stats[key] if nested is None else split_stats[key][nested]
        right = global_stats[key] if nested is None else global_stats[key][nested]
        keys = set(left) | set(right)
        total = max(sum(right.values()), 1)
        return sum(abs(left.get(item, 0) / max(sum(left.values()), 1) - right.get(item, 0) / total) for item in keys) / 2

    return {
        "entity_type_total_variation": error("entity_type_counts"),
        "relation_type_total_variation": error("relation_type_counts"),
        "relation_distance_total_variation": error("relation_distance_buckets"),
        "max_distance_total_variation": error("max_relation_distance", "buckets"),
        "length_total_variation": error("token_length", "buckets"),
        "target_ratio": ratio,
    }


def write_dataset(
    output_dir: Path,
    config_root: Path,
    threshold: int,
    records: list[dict[str, Any]],
    assignments: dict[str, list[int]],
    repartition_info: dict[str, Any],
    dedup_info: dict[str, Any],
    input_dir: Path,
    dataset_suffix: str,
    config_prefix: str,
    dataset_name_prefix: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    types_path = input_dir / "types.json"
    if types_path.exists():
        shutil.copy2(types_path, output_dir / "types.json")

    split_stats: dict[str, Any] = {}
    global_indices = list(range(len(records)))
    global_stats = aggregate_stats(records, global_indices)
    for split in SPLITS:
        indices = assignments[split]
        rows = [records[index]["row"] for index in indices]
        (output_dir / f"{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        stats = aggregate_stats(records, indices)
        stats["distribution_error"] = distribution_error(
            stats,
            global_stats,
            len(indices) / max(len(records), 1),
        )
        split_stats[split] = stats

    manifest_path = output_dir / "document_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for split in SPLITS:
            for index in assignments[split]:
                record = records[index]
                handle.write(
                    json.dumps(
                        {
                            "output_split": split,
                            "document_hash": record["hash"],
                            "source_origins": record["origins"],
                            "token_count": record["token_length"],
                            "entity_types": sorted(
                                {str(entity["type"]) for entity in record["row"].get("entities", [])}
                            ),
                            "relation_types": sorted(
                                {str(relation["type"]) for relation in record["row"].get("relations", [])}
                            ),
                            "max_relation_distance": record["max_distance"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    card = {
        "source_dir": str(input_dir),
        "output_dir": str(output_dir),
        "selection": {
            "inherited_rule": "input dataset contains documents whose maximum gold relation center distance exceeds the threshold",
            "threshold": threshold,
            "distance_definition": "abs((head.start + head.end)//2 - (tail.start + tail.end)//2)",
        },
        "repartition": {
            "rule": "merge input train/dev/test, exact-deduplicate, then deterministic multi-label stratification",
            "ratios": {"train": 0.8, "dev": 0.1, "test": 0.1},
            **repartition_info,
        },
        "deduplication": dedup_info,
        "global": global_stats,
        "splits": split_stats,
        "provenance_manifest": str(manifest_path),
    }
    (output_dir / "dataset_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    config_root.mkdir(parents=True, exist_ok=True)
    config_path = config_root / f"{config_prefix}{threshold}{dataset_suffix}.json"
    config = {
        "base": "wenyanwen/wenyanwen_long_context_512_type_aux",
        "dataset": {
            "name": f"{dataset_name_prefix}{threshold}{dataset_suffix}",
            "data_dir": output_dir.name,
            "reinitialize_dataset": False,
        },
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "threshold": threshold,
        "dataset_dir": str(output_dir),
        "config": str(config_path),
        "dataset_card": str(output_dir / "dataset_card.json"),
        "splits": split_stats,
        "deduplication": dedup_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("datasets"))
    parser.add_argument("--input-prefix", default="ind_distance_gt")
    parser.add_argument("--output-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-suffix", default="_rebalanced")
    parser.add_argument("--config-root", type=Path, default=Path("cfg/wenyanwen"))
    parser.add_argument(
        "--config-prefix",
        default="wenyanwen_long_context_512_type_aux_ind_distance_gt",
    )
    parser.add_argument(
        "--dataset-name-prefix",
        default="wenyanwen_long_context_type_aux_ind_distance_gt",
    )
    parser.add_argument("--thresholds", type=int, nargs="+", default=[28, 36])
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--swap-attempts", type=int, default=10000)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("artifacts/analysis/ind_distance_rebalanced_manifest.json"),
    )
    args = parser.parse_args()
    if not 0 < args.train_ratio < 1 or not 0 < args.dev_ratio < 1:
        raise ValueError("train/dev ratios must be between 0 and 1")
    if args.train_ratio + args.dev_ratio >= 1:
        raise ValueError("train_ratio + dev_ratio must be less than 1")

    manifest: dict[str, Any] = {
        "seed": args.seed,
        "attempts": args.attempts,
        "ratios": {
            "train": args.train_ratio,
            "dev": args.dev_ratio,
            "test": 1 - args.train_ratio - args.dev_ratio,
        },
        "datasets": {},
    }
    # Keep previously generated threshold entries when adding one more
    # dataset.  This makes the manifest safe for incremental runs such as
    # generating gt32 after gt28 and gt36 already exist.
    if args.manifest_output.exists():
        try:
            previous = json.loads(args.manifest_output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        if isinstance(previous, dict) and isinstance(previous.get("datasets"), dict):
            manifest["datasets"].update(previous["datasets"])
    for threshold in sorted(set(args.thresholds)):
        input_dir = args.input_root / f"{args.input_prefix}{threshold}"
        output_dir = args.output_root / f"{args.input_prefix}{threshold}{args.output_suffix}"
        records, dedup_info = load_unique_records(input_dir)
        if len(records) < 30:
            raise ValueError(f"{input_dir} leaves only {len(records)} unique documents")
        assignments, repartition_info = repartition(
            records,
            args.train_ratio,
            args.dev_ratio,
            args.seed + threshold,
            args.attempts,
            args.swap_attempts,
        )
        result = write_dataset(
            output_dir=output_dir,
            config_root=args.config_root,
            threshold=threshold,
            records=records,
            assignments=assignments,
            repartition_info=repartition_info,
            dedup_info=dedup_info,
            input_dir=input_dir,
            dataset_suffix=args.output_suffix,
            config_prefix=args.config_prefix,
            dataset_name_prefix=args.dataset_name_prefix,
        )
        manifest["datasets"][str(threshold)] = result
        counts = ", ".join(f"{split}={len(assignments[split])}" for split in SPLITS)
        print(f">{threshold}: {counts}; unique={len(records)}; objective={repartition_info['objective']:.6f}")

    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.manifest_output}")


if __name__ == "__main__":
    main()
