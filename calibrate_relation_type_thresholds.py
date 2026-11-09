#!/usr/bin/env python3
"""Calibrate per-relation decision thresholds on eval/dev predictions only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iter.misc.document_metrics import evaluate_documents


def _load_documents(path: Path, split: str) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("prediction", {}).get("relations") is None:
                raise ValueError(f"Missing prediction relations in {path}")
            records.append(record)
    if not records:
        raise ValueError(f"No prediction records found in {path}")
    return records


def _score(records: list[dict], thresholds: dict[str, float], long_threshold: int) -> dict:
    documents = []
    for record in records:
        document = dict(record)
        prediction = dict(record["prediction"])
        prediction["relations"] = [
            relation
            for relation in prediction.get("relations", [])
            if float(relation.get("final_probability", relation.get("max_probability", 0.0)))
            >= thresholds.get(relation["type"], thresholds["__default__"])
        ]
        document["prediction"] = prediction
        documents.append(document)
    return evaluate_documents(documents, long_threshold=long_threshold)


def _prune_candidates(records: list[dict], probability_floor: float) -> tuple[list[dict], int, int]:
    """Drop candidates that cannot pass any threshold in this calibration."""
    pruned = []
    before = after = 0
    for record in records:
        document = dict(record)
        prediction = dict(record["prediction"])
        relations = prediction.get("relations", [])
        before += len(relations)
        prediction["relations"] = [
            relation
            for relation in relations
            if float(relation.get("final_probability", relation.get("max_probability", 0.0)))
            >= probability_floor
        ]
        after += len(prediction["relations"])
        document["prediction"] = prediction
        pruned.append(document)
    return pruned, before, after


def _metric_key(metrics: dict) -> tuple[float, float, float, float]:
    overall = metrics["overall"]
    long_slice = metrics["length_buckets"]["256-511"]
    return (
        overall["re_strict_end_to_end"]["f1"],
        long_slice["re_strict_end_to_end"]["f1"],
        overall["re_boundary_end_to_end"]["f1"],
        overall["re_strict_end_to_end"]["precision"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-split", choices=("eval", "dev"), default="eval")
    parser.add_argument("--global-threshold", type=float, default=0.20)
    parser.add_argument("--min-support", type=int, default=8)
    parser.add_argument(
        "--threshold-grid",
        type=float,
        nargs="+",
        default=[0.10, 0.14, 0.18, 0.20, 0.22, 0.26, 0.30, 0.34, 0.38, 0.40, 0.44, 0.50, 0.60, 0.70],
    )
    parser.add_argument("--long-threshold", type=int, default=128)
    args = parser.parse_args()
    if not 0.0 <= args.global_threshold <= 1.0:
        parser.error("--global-threshold must be between 0 and 1")
    if args.min_support < 1:
        parser.error("--min-support must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.threshold_grid):
        parser.error("All threshold-grid values must be between 0 and 1")

    records = _load_documents(args.predictions, args.source_split)
    probability_floor = min([args.global_threshold, *args.threshold_grid])
    records, candidates_before, candidates_after = _prune_candidates(records, probability_floor)
    print(
        f"Candidate pruning at {probability_floor:.4f}: "
        f"{candidates_before} -> {candidates_after}",
        flush=True,
    )

    metrics_path = args.predictions.with_name(f"{args.source_split}_metrics.json")
    candidate_run = None
    if metrics_path.exists():
        candidate_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        candidate_run = candidate_metrics.get("run")
        if candidate_run and candidate_run.get("split") != args.source_split:
            raise SystemExit(
                f"Candidate metrics split mismatch: {candidate_run.get('split')!r}"
            )
    support: dict[str, int] = {}
    relation_types = set()
    for record in records:
        for relation in record.get("gold", {}).get("relations", []):
            relation_types.add(relation["type"])
            support[relation["type"]] = support.get(relation["type"], 0) + 1
        for relation in record.get("prediction", {}).get("relations", []):
            relation_types.add(relation["type"])

    thresholds = {"__default__": args.global_threshold}
    for relation_type in sorted(relation_types):
        thresholds[relation_type] = args.global_threshold

    baseline = _score(records, thresholds, args.long_threshold)
    history = []
    eligible = sorted(
        relation_type
        for relation_type in relation_types
        if support.get(relation_type, 0) >= args.min_support
    )
    # Coordinate descent keeps the calibration reproducible and avoids an
    # exponential joint search over all relation types.
    for round_id in range(2):
        changed = False
        for relation_type in eligible:
            current = thresholds[relation_type]
            best_threshold = current
            best_metrics = baseline
            for candidate in args.threshold_grid:
                trial = dict(thresholds)
                trial[relation_type] = candidate
                metrics = _score(records, trial, args.long_threshold)
                if _metric_key(metrics) > _metric_key(best_metrics):
                    best_threshold = candidate
                    best_metrics = metrics
            thresholds[relation_type] = best_threshold
            if best_threshold != current:
                changed = True
            baseline = best_metrics
            history.append({
                "round": round_id + 1,
                "relation_type": relation_type,
                "support": support.get(relation_type, 0),
                "threshold": best_threshold,
                "metrics": {
                    "re_strict": best_metrics["overall"]["re_strict_end_to_end"],
                    "re_strict_256_511": best_metrics["length_buckets"]["256-511"]["re_strict_end_to_end"],
                },
            })
        if not changed:
            break

    result = {
        "source_split": args.source_split,
        "fusion_mode": "learned",
        "candidate_source": str(args.predictions.resolve()),
        "candidate_run": candidate_run,
        "candidate_probability_floor": probability_floor,
        "candidate_count_before_pruning": candidates_before,
        "candidate_count_after_pruning": candidates_after,
        "global_threshold": args.global_threshold,
        "min_support": args.min_support,
        "threshold_grid": args.threshold_grid,
        "thresholds": {key: value for key, value in thresholds.items() if key != "__default__"},
        "support": support,
        "eligible_types": eligible,
        "best_at_search_boundary": any(
            thresholds.get(name, args.global_threshold) in {min(args.threshold_grid), max(args.threshold_grid)}
            for name in eligible
        ),
        "metrics": {
            "re_strict": baseline["overall"]["re_strict_end_to_end"],
            "re_boundary": baseline["overall"]["re_boundary_end_to_end"],
            "re_on_matched_entities": baseline["overall"]["re_on_matched_entities"],
            "re_strict_256_511": baseline["length_buckets"]["256-511"]["re_strict_end_to_end"],
        },
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"Per-relation calibration written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
