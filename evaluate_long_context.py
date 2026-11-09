#!/usr/bin/env python3
"""Recompute overall and long-context metrics from inference JSONL output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from iter.misc.document_metrics import evaluate_documents


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--long_threshold", type=int, default=128)
    parser.add_argument("--length_buckets", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--distance_buckets", type=int, nargs="+", default=[32, 64, 128])
    args = parser.parse_args()

    documents = []
    with args.predictions.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))
    metrics = evaluate_documents(
        documents,
        long_threshold=args.long_threshold,
        length_buckets=tuple(sorted(set(args.length_buckets))),
        relation_distance_buckets=tuple(sorted(set(args.distance_buckets))),
    )
    output = args.output or args.predictions.with_name(
        args.predictions.stem.replace("_predictions", "_metrics_recomputed") + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    csv_output = output.with_suffix(".csv")
    with csv_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slice", "task", "precision", "recall", "f1", "tp", "fp", "fn", "documents"])
        slices = {
            "overall": metrics["overall"],
            "long_documents": metrics["long_documents"],
            **metrics["length_buckets"],
        }
        for slice_name, values in slices.items():
            for task in (
                "ner_strict",
                "ner_boundary",
                "re_strict_end_to_end",
                "re_boundary_end_to_end",
                "re_on_matched_entities",
            ):
                score = values[task]
                writer.writerow([
                    slice_name, task, score["precision"], score["recall"], score["f1"],
                    score["tp"], score["fp"], score["fn"], values["documents"],
                ])
    per_type_output = output.with_name(output.stem + "_per_type.csv")
    with per_type_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "type", "precision", "recall", "f1", "tp", "fp", "fn", "support"])
        for task, values in (("ner", metrics["ner_per_type"]), ("re", metrics["re_per_type"])):
            for type_name, score in values.items():
                writer.writerow([
                    task, type_name, score["precision"], score["recall"], score["f1"],
                    score["tp"], score["fp"], score["fn"], score["support"],
                ])
    print(json.dumps(metrics["overall"], ensure_ascii=False, indent=2))
    print(f"Metrics written to: {output.resolve()}")


if __name__ == "__main__":
    main()
