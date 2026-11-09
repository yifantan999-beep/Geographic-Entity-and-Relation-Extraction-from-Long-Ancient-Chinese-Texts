#!/usr/bin/env python3
"""Validate aligned NER ablations and emit LaTeX rows for paper Table 3."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


VARIANTS = (
    ("full", "Full LongGeoIE"),
    ("no_type_aux", r"$-$TypeAux"),
    ("no_span_biaffine", r"$-$Span-Biaffine"),
    ("no_nms", r"$-$NMS"),
)
METRICS = ("ner_strict", "ner_boundary")
MATCH_FIELDS = ("precision", "recall", "f1", "tp", "fp", "fn", "support", "predicted")


def load_metrics(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Required metrics file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_full_matches_primary(full: dict, primary: dict) -> None:
    full_run = full.get("run", {})
    primary_run = primary.get("run", {})
    for name, run in (("Full ablation", full_run), ("Table 2 primary", primary_run)):
        if run.get("split") != "test":
            raise SystemExit(f"{name} metrics are not from the test split.")
        if float(run.get("ner_span_threshold", -1)) != 0.80:
            raise SystemExit(f"{name} does not use span threshold 0.80.")
        if float(run.get("ner_type_threshold", -1)) != 0.50:
            raise SystemExit(f"{name} does not use type threshold 0.50.")
    if full_run.get("ner_overlap_strategy") != "nms":
        raise SystemExit("Full LongGeoIE must use NMS, matching Table 2.")

    for metric in METRICS:
        full_values = full.get("overall", {}).get(metric, {})
        primary_values = primary.get("overall", {}).get(metric, {})
        for field in MATCH_FIELDS:
            if full_values.get(field) != primary_values.get(field):
                raise SystemExit(
                    f"Full checkpoint does not reproduce Table 2 for {metric}.{field}: "
                    f"{full_values.get(field)!r} != {primary_values.get(field)!r}."
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--primary-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    loaded = {
        tag: load_metrics(args.root / tag / "test_metrics.json")
        for tag, _ in VARIANTS
    }
    primary = load_metrics(args.primary_metrics)
    validate_full_matches_primary(loaded["full"], primary)

    primary_support = primary["overall"]["ner_strict"]["support"]
    rows = []
    for tag, label in VARIANTS:
        payload = loaded[tag]
        run = payload.get("run", {})
        strict = payload.get("overall", {}).get("ner_strict", {})
        boundary = payload.get("overall", {}).get("ner_boundary", {})
        expected_overlap = "none" if tag == "no_nms" else "nms"
        if run.get("ner_overlap_strategy") != expected_overlap:
            raise SystemExit(
                f"{label} uses overlap strategy {run.get('ner_overlap_strategy')!r}; "
                f"expected {expected_overlap!r}."
            )
        if strict.get("support") != primary_support:
            raise SystemExit(f"{label} was evaluated against a different entity support.")
        if run.get("source_documents") != primary_run_documents(primary):
            raise SystemExit(f"{label} was evaluated on a different number of test contexts.")
        rows.append(
            {
                "variant": label,
                "tag": tag,
                "ner_strict_precision": strict.get("precision"),
                "ner_strict_recall": strict.get("recall"),
                "ner_strict_f1": strict.get("f1"),
                "ner_boundary_f1": boundary.get("f1"),
                "tp": strict.get("tp"),
                "fp": strict.get("fp"),
                "fn": strict.get("fn"),
                "support": strict.get("support"),
                "predicted": strict.get("predicted"),
                "span_threshold": run.get("ner_span_threshold"),
                "type_threshold": run.get("ner_type_threshold"),
                "overlap_strategy": run.get("ner_overlap_strategy"),
                "checkpoint": run.get("checkpoint"),
                "config": run.get("config"),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "table3_ner_ablation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    strict_values = [float(row["ner_strict_f1"]) for row in rows]
    boundary_values = [float(row["ner_boundary_f1"]) for row in rows]
    tex_lines = []
    for index, row in enumerate(rows):
        strict = f"{100 * strict_values[index]:.2f}"
        boundary = f"{100 * boundary_values[index]:.2f}"
        if strict_values[index] == max(strict_values):
            strict = rf"\textbf{{{strict}}}"
        if boundary_values[index] == max(boundary_values):
            boundary = rf"\textbf{{{boundary}}}"
        tex_lines.append(f"{row['variant']} & {strict} & {boundary} " + r"\\")

    tex_path = args.output_dir / "table3_ner_ablation_rows.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    manifest = {
        "primary_metrics": str(args.primary_metrics.resolve()),
        "ablation_metrics_root": str(args.root.resolve()),
        "full_matches_table2": True,
        "independent_test_set": False,
        "evaluation_note": (
            "The diagnostic dataset reuses the same contexts for development and test; "
            "report these results as aligned ablation comparisons, not as an independent "
            "held-out test estimate."
        ),
        "table_rows": rows,
    }
    (args.output_dir / "table3_ner_ablation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(tex_path.read_text(encoding="utf-8"), end="")
    print(f"CSV: {csv_path.resolve()}")


def primary_run_documents(payload: dict) -> int | None:
    value = payload.get("run", {}).get("source_documents")
    if value is not None:
        return int(value)
    return payload.get("overall", {}).get("ner_strict", {}).get("documents")


if __name__ == "__main__":
    main()
