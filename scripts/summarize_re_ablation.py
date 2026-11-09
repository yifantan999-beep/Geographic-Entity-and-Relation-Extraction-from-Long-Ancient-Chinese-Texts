#!/usr/bin/env python3
"""Summarize fixed-test RE ablation metrics into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


VARIANTS = (
    ("full", "Full v8 fixed"),
    ("no_learned_fusion", "-Learned Fusion"),
    ("no_distance_gate", "-Distance Gate"),
    ("no_long_distance_head", "-Long-Distance Head"),
    ("no_distance_experts", "-Distance Experts"),
    ("no_pair_type", "-Pair-Type Compatibility"),
    ("no_direction", "-Direction Embedding"),
    ("no_window_position", "-Window Position Encoding"),
    ("no_long_distance", "-All Long-Distance Modules"),
)

BUCKETS = ("0-31", "32-63", "64-127", ">=128")


def score(payload: dict, section: str, field: str):
    return payload.get("overall", {}).get(section, {}).get(field)


def f1(payload: dict, section: str):
    value = score(payload, section, "f1")
    return float(value) if value is not None else None


def load_row(root: Path, tag: str, label: str, expected_threshold: float | None = None) -> dict:
    path = root / tag / "test_metrics.json"
    row = {
        "variant": label,
        "tag": tag,
        "status": "ok" if path.is_file() else "missing",
        "metrics": str(path.resolve()),
    }
    if not path.is_file():
        return row
    payload = json.loads(path.read_text(encoding="utf-8"))
    run = payload.get("run", {})
    actual_checkpoint = run.get("document_re_checkpoint")
    actual_threshold = run.get("relation_threshold")
    if not actual_checkpoint:
        row["status"] = "invalid_no_document_re"
    elif expected_threshold is not None:
        try:
            if abs(float(actual_threshold) - expected_threshold) > 1e-8:
                row["status"] = "invalid_threshold"
        except (TypeError, ValueError):
            row["status"] = "invalid_threshold"
    strict = payload.get("overall", {}).get("re_strict_end_to_end", {})
    row.update(
        {
            "split": run.get("split"),
            "checkpoint": run.get("document_re_checkpoint"),
            "config": run.get("config"),
            "relation_threshold": run.get("relation_threshold"),
            "ner_span_threshold": run.get("ner_span_threshold"),
            "ner_type_threshold": run.get("ner_type_threshold"),
            "ner_overlap_strategy": run.get("ner_overlap_strategy"),
            "strict_precision": strict.get("precision"),
            "strict_recall": strict.get("recall"),
            "strict_f1": strict.get("f1"),
            "boundary_f1": f1(payload, "re_boundary_end_to_end"),
            "matched_f1": f1(payload, "re_on_matched_entities"),
            "re_macro_f1": (payload.get("macro_average", {}).get("re") or {}).get("f1"),
            "tp": strict.get("tp"),
            "fp": strict.get("fp"),
            "fn": strict.get("fn"),
        }
    )
    for bucket in BUCKETS:
        value = payload.get("relation_distance_buckets", {}).get(bucket, {}).get("f1")
        row[f"distance_{bucket}_f1"] = value
    return row


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--expected-threshold", type=float, default=None)
    args = parser.parse_args()

    rows = [load_row(args.root, tag, label, args.expected_threshold) for tag, label in VARIANTS]
    fields = [
        "variant", "tag", "status", "split", "strict_precision", "strict_recall",
        "strict_f1", "boundary_f1", "matched_f1", "re_macro_f1", "tp", "fp", "fn",
        "distance_0-31_f1", "distance_32-63_f1", "distance_64-127_f1",
        "distance_>=128_f1", "relation_threshold", "ner_span_threshold",
        "ner_type_threshold", "ner_overlap_strategy", "checkpoint", "config", "metrics",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    columns = [
        ("Variant", "variant"), ("Status", "status"), ("RE-P", "strict_precision"),
        ("RE-R", "strict_recall"), ("Strict RE-F1", "strict_f1"),
        ("Boundary RE-F1", "boundary_f1"), ("Matched RE-F1", "matched_f1"),
        ("RE Macro-F1", "re_macro_f1"), ("0-31", "distance_0-31_f1"),
        ("32-63", "distance_32-63_f1"), ("64-127", "distance_64-127_f1"),
        (">=128", "distance_>=128_f1"), ("TP", "tp"), ("FP", "fp"), ("FN", "fn"),
    ]
    lines = [
        "# RE Ablation Results", "",
        "All rows use the same Stage-1 predicted entities, fixed NER decoding, and a pre-registered RE threshold. Test is not calibrated.",
        "", "| " + " | ".join(title for title, _ in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for _, key in columns) + " |")
    lines.extend(["", "Metrics files:", ""])
    lines.extend(f"- `{row['variant']}`: `{row['metrics']}`" for row in rows)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"CSV written to: {args.output_csv.resolve()}")
    print(f"Markdown written to: {args.output_md.resolve()}")


if __name__ == "__main__":
    main()
