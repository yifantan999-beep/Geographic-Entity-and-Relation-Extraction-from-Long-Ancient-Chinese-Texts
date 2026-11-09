#!/usr/bin/env python3
"""Build the isolated long-context corpus used by every later experiment.

The source is the converted ``datasets/ind`` corpus.  The source is read only
by this preparation step; training, inference and evaluation use the generated
``datasets/long_context`` directory exclusively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def percentile(values: list[int], fraction: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(values[lower])
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def choose_threshold(lengths: list[int], step: int = 32) -> tuple[int, float]:
    """Use the upper tail of the original distribution as the long threshold.

    P90 makes "long" an empirical property of this corpus rather than an
    arbitrary model limit.  Rounding upward to a 32-token boundary keeps the
    split easy to reproduce and leaves enough documents for held-out tests.
    """
    p90 = percentile(lengths, 0.90)
    return max(step, int(math.ceil(p90 / step) * step)), p90


def _group_key(item: dict[str, Any]) -> str:
    payload = json.dumps(
        {"tokens": item["tokens"], "entities": item.get("entities", []), "relations": item.get("relations", [])},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assign_groups(groups: list[list[dict[str, Any]]], seed: int) -> dict[str, list[dict[str, Any]]]:
    total = sum(len(group) for group in groups)
    target = {"train": round(total * 0.80), "dev": round(total * 0.10)}
    target["test"] = total - target["train"] - target["dev"]
    buckets = {name: [] for name in target}
    rng = random.Random(seed)
    keyed_groups = []
    for group in groups:
        key = _group_key(group[0])
        keyed_groups.append((key, group))
    rng.shuffle(keyed_groups)
    keyed_groups.sort(key=lambda pair: (len(pair[1][0]["tokens"]), pair[0]), reverse=True)

    counts = Counter()
    for _, group in keyed_groups:
        choices = [name for name in buckets if counts[name] + len(group) <= target[name]]
        if not choices:
            choices = list(buckets)
        destination = min(
            choices,
            key=lambda name: (counts[name] / max(target[name], 1), name),
        )
        buckets[destination].extend(group)
        counts[destination] += len(group)
    for name in buckets:
        buckets[name].sort(key=lambda item: (len(item["tokens"]), _group_key(item)))
    return buckets


def build(source_dir: Path, output_dir: Path, threshold: int | None, seed: int) -> dict[str, Any]:
    source_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "dev", "test"):
        path = source_dir / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        source_rows[split] = json.loads(path.read_text(encoding="utf-8"))
    all_rows = [row for rows in source_rows.values() for row in rows]
    lengths = [len(row["tokens"]) for row in all_rows]
    selected_threshold, p90 = choose_threshold(lengths)
    threshold = selected_threshold if threshold is None else threshold
    selected = [row for row in all_rows if len(row["tokens"]) >= threshold]
    if len(selected) < 30:
        raise ValueError(f"threshold={threshold} leaves only {len(selected)} documents")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[_group_key(row)].append(row)
    # Stratify by coarse length band before deterministic assignment.
    bands: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        bands[min((len(row["tokens"]) - threshold) // 64, 3)].append(row)
    groups_by_band: list[list[dict[str, Any]]] = []
    for band in sorted(bands):
        band_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in bands[band]:
            band_groups[_group_key(row)].append(row)
        groups_by_band.extend(band_groups.values())
    buckets = _assign_groups(groups_by_band, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in buckets.items():
        (output_dir / f"{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    types_path = source_dir / "types.json"
    if types_path.exists():
        (output_dir / "types.json").write_text(types_path.read_text(encoding="utf-8"), encoding="utf-8")

    source_counts = {split: sum(len(row["tokens"]) >= threshold for row in rows) for split, rows in source_rows.items()}
    report: dict[str, Any] = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "selection": {
            "rule": "documents at or above P90 of the original word-length distribution, rounded up to a multiple of 32",
            "p90_words": p90,
            "threshold_words": threshold,
            "source_documents": len(all_rows),
            "selected_documents": len(selected),
            "source_long_documents_by_split": source_counts,
        },
        "splits": {},
        "entity_type_counts": {},
        "relation_type_counts": {},
    }
    entity_counts = Counter()
    relation_counts = Counter()
    for split, rows in buckets.items():
        lengths = [len(row["tokens"]) for row in rows]
        report["splits"][split] = {
            "documents": len(rows),
            "min_words": min(lengths),
            "median_words": sorted(lengths)[len(lengths) // 2],
            "max_words": max(lengths),
            "entities": sum(len(row.get("entities", [])) for row in rows),
            "relations": sum(len(row.get("relations", [])) for row in rows),
        }
        entity_counts.update(entity["type"] for row in rows for entity in row.get("entities", []))
        relation_counts.update(relation["type"] for row in rows for relation in row.get("relations", []))
    report["entity_type_counts"] = dict(sorted(entity_counts.items()))
    report["relation_type_counts"] = dict(sorted(relation_counts.items()))
    (output_dir / "dataset_card.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, default=Path("datasets/ind"))
    parser.add_argument("--output_dir", type=Path, default=Path("datasets/long_context"))
    parser.add_argument("--threshold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    report = build(args.source_dir, args.output_dir, args.threshold, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
