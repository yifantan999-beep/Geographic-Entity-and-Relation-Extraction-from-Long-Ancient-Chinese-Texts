#!/usr/bin/env python3
"""Create a compact NER ablation table from fixed-split metrics files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


VARIANTS = (
    ("full", "Full v8 fixed"),
    ("no_type_aux", "-TypeAux"),
    ("no_boundary_aux", "-BoundaryAux"),
    ("no_span_biaffine", "-Span-Biaffine"),
    ("no_nms", "-NMS"),
)


def _f1(payload: dict, section: str) -> float | None:
    value = payload.get("overall", {}).get(section, {}).get("f1")
    return float(value) if value is not None else None


def _metric(payload: dict, section: str, field: str) -> float | int | None:
    value = payload.get("overall", {}).get(section, {}).get(field)
    if value is None:
        return None
    return value


def _macro_ner_f1(payload: dict) -> float | None:
    macro = payload.get("macro_average") or payload.get("macro") or {}
    value = (macro.get("ner") or {}).get("f1")
    return float(value) if value is not None else None


def load_row(root: Path, tag: str, label: str) -> dict:
    metrics_path = root / tag / "test_metrics.json"
    row = {
        "variant": label,
        "tag": tag,
        "metrics": str(metrics_path.resolve()),
        "status": "ok" if metrics_path.is_file() else "missing",
    }
    if not metrics_path.is_file():
        return row

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    run = payload.get("run", {})
    strict = "ner_strict"
    boundary = "ner_boundary"
    row.update(
        {
            "split": run.get("split", "test"),
            "checkpoint": run.get("checkpoint"),
            "config": run.get("config"),
            "span_threshold": run.get("ner_span_threshold"),
            "type_threshold": run.get("ner_type_threshold"),
            "overlap_strategy": run.get("ner_overlap_strategy"),
            "ner_precision": _metric(payload, strict, "precision"),
            "ner_recall": _metric(payload, strict, "recall"),
            "ner_strict_f1": _f1(payload, strict),
            "ner_boundary_f1": _f1(payload, boundary),
            "ner_macro_f1": _macro_ner_f1(payload),
            "predicted_entities": _metric(payload, strict, "predicted"),
            "ner_fp": _metric(payload, strict, "fp"),
            "ner_fn": _metric(payload, strict, "fn"),
        }
    )
    return row


def format_value(value: object) -> str:
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
    args = parser.parse_args()

    rows = [load_row(args.root, tag, label) for tag, label in VARIANTS]
    fields = [
        "variant",
        "tag",
        "status",
        "split",
        "ner_precision",
        "ner_recall",
        "ner_strict_f1",
        "ner_boundary_f1",
        "ner_macro_f1",
        "predicted_entities",
        "ner_fp",
        "ner_fn",
        "span_threshold",
        "type_threshold",
        "overlap_strategy",
        "checkpoint",
        "config",
        "metrics",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    columns = [
        ("Variant", "variant"),
        ("Status", "status"),
        ("NER-P", "ner_precision"),
        ("NER-R", "ner_recall"),
        ("NER strict F1", "ner_strict_f1"),
        ("NER boundary F1", "ner_boundary_f1"),
        ("Entity macro-F1", "ner_macro_f1"),
        ("Candidates", "predicted_entities"),
        ("FP", "ner_fp"),
        ("FN", "ner_fn"),
    ]
    lines = [
        "# NER Ablation Results",
        "",
        "All rows use the same fixed split and decoding thresholds; test is not calibrated.",
        "",
        "| " + " | ".join(title for title, _ in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(format_value(row.get(key)) for _, key in columns)
            + " |"
        )
    lines.extend(
        [
            "",
            "Metrics files:",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- `{row['variant']}`: `{row['metrics']}`")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"CSV written to: {args.output_csv.resolve()}")
    print(f"Markdown written to: {args.output_md.resolve()}")


if __name__ == "__main__":
    main()
