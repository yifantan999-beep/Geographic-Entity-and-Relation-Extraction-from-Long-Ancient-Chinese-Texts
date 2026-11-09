#!/usr/bin/env python3
"""Build the three minimum LongGeoIE experiment figures and their source CSVs.

The script accepts one or more METHOD=predictions.jsonl inputs.  The first
method is treated as the reference system for support bars and fusion-gate
analysis.  Predictions produced with ``infer_long_context.py
--export_fusion_trace`` provide the preferred gold-relation gate statistics.
Legacy predictions remain supported through an explicitly labelled fallback
that uses recalled gold relations only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


DISTANCE_BUCKETS = (
    ("0-31", 0, 32),
    ("32-63", 32, 64),
    ("64-95", 64, 96),
    ("96-127", 96, 128),
    ("128-191", 128, 192),
    ("192-255", 192, 256),
    (">=256", 256, None),
)
METHOD_COLORS = {
    "LongGeoIE": "#0072B2",
    "w/o Long-distance modules": "#D55E00",
    "SpIB": "#D55E00",
    "LDNet": "#009E73",
    "ATG": "#CC79A7",
}
METHOD_MARKERS = {
    "LongGeoIE": "o",
    "w/o Long-distance modules": "s",
    "SpIB": "s",
    "LDNet": "^",
    "ATG": "D",
}


def parse_method_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected METHOD=predictions.jsonl")
    method, raw_path = value.split("=", 1)
    method, path = method.strip(), Path(raw_path.strip())
    if not method:
        raise argparse.ArgumentTypeError("method name cannot be empty")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"predictions file does not exist: {path}")
    return method, path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def entity_key(entity: dict) -> tuple[int, int, str]:
    return int(entity["start"]), int(entity["end"]), str(entity["type"])


def relation_key(relation: dict, entities: list[dict]) -> tuple:
    head = entity_key(entities[int(relation["head"])])
    tail = entity_key(entities[int(relation["tail"])])
    if relation.get("symmetric") and tail < head:
        head, tail = tail, head
    return head, str(relation["type"]), tail


def relation_distance(relation: dict, entities: list[dict]) -> int:
    head = entities[int(relation["head"])]
    tail = entities[int(relation["tail"])]
    head_center = (int(head["start"]) + int(head["end"])) // 2
    tail_center = (int(tail["start"]) + int(tail["end"])) // 2
    return abs(head_center - tail_center)


def distance_bucket(distance: int) -> str:
    for label, lower, upper in DISTANCE_BUCKETS:
        if distance >= lower and (upper is None or distance < upper):
            return label
    raise AssertionError(distance)


def safe_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def strict_relation_sets(document: dict) -> tuple[dict[tuple, int], dict[tuple, int]]:
    gold = document["gold"]
    pred = document["prediction"]
    gold_rows = {
        relation_key(relation, gold["entities"]): relation_distance(relation, gold["entities"])
        for relation in gold.get("relations", [])
    }
    pred_rows = {
        relation_key(relation, pred["entities"]): relation_distance(relation, pred["entities"])
        for relation in pred.get("relations", [])
    }
    return gold_rows, pred_rows


def build_distance_rows(method_documents: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for method, documents in method_documents.items():
        counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label, _, _ in DISTANCE_BUCKETS}
        for document in documents:
            gold, pred = strict_relation_sets(document)
            gold_keys, pred_keys = set(gold), set(pred)
            for key in gold_keys & pred_keys:
                counts[distance_bucket(gold[key])]["tp"] += 1
            for key in gold_keys - pred_keys:
                counts[distance_bucket(gold[key])]["fn"] += 1
            for key in pred_keys - gold_keys:
                counts[distance_bucket(pred[key])]["fp"] += 1
        for label, _, _ in DISTANCE_BUCKETS:
            values = counts[label]
            precision, recall, f1 = safe_prf(values["tp"], values["fp"], values["fn"])
            rows.append(
                {
                    "method": method,
                    "bucket": label,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    **values,
                    "support": values["tp"] + values["fn"],
                    "predicted": values["tp"] + values["fp"],
                }
            )
    return rows


def exact_recalled_gate_rows(method: str, documents: list[dict]) -> list[dict]:
    """Compatibility fallback for predictions written before fusion_trace existed."""
    rows = []
    for document in documents:
        gold = document["gold"]
        pred = document["prediction"]
        gold_keys = {
            relation_key(relation, gold["entities"]): relation
            for relation in gold.get("relations", [])
        }
        for relation in pred.get("relations", []):
            key = relation_key(relation, pred["entities"])
            if key not in gold_keys or relation.get("fusion_gate") is None:
                continue
            distance = relation_distance(gold_keys[key], gold["entities"])
            local_weight = float(relation["fusion_gate"])
            rows.append(
                {
                    "method": method,
                    "context_id": document["document_id"],
                    "gold_relation_id": "",
                    "relation_type": relation["type"],
                    "distance": distance,
                    "bucket": distance_bucket(distance),
                    "local_gate_weight": local_weight,
                    "global_gate_weight": 1.0 - local_weight,
                    "endpoint_match": True,
                    "local_available": relation.get("local_probability") is not None,
                    "is_recalled": True,
                    "same_window": gold_keys[key].get("window_observable", True),
                    "scope": "recalled_gold_relations_fallback",
                }
            )
    return rows


def build_gate_trace(method: str, documents: list[dict]) -> list[dict]:
    trace_rows = []
    has_trace = any("fusion_trace" in document.get("prediction", {}) for document in documents)
    if not has_trace:
        return exact_recalled_gate_rows(method, documents)
    for document in documents:
        for item in document["prediction"].get("fusion_trace", []):
            distance = int(item["distance"])
            trace_rows.append(
                {
                    "method": method,
                    "context_id": document["document_id"],
                    "gold_relation_id": item.get("gold_relation_id", ""),
                    "relation_type": item["relation_type"],
                    "distance": distance,
                    "bucket": distance_bucket(distance),
                    "local_gate_weight": item.get("local_gate_weight"),
                    "global_gate_weight": item.get("global_gate_weight"),
                    "endpoint_match": bool(item.get("endpoint_match")),
                    "local_available": bool(item.get("local_available")),
                    "is_recalled": bool(item.get("is_recalled")),
                    "same_window": bool(item.get("same_window")),
                    "scope": "all_gold_relations_with_exact_predicted_endpoints",
                }
            )
    return trace_rows


def clustered_bootstrap_ci(
    rows: list[dict], field: str, rng: np.random.Generator, samples: int
) -> tuple[float, float]:
    by_document: dict[object, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            by_document[row["context_id"]].append(float(value))
    document_ids = list(by_document)
    if not document_ids:
        return float("nan"), float("nan")
    draws = []
    for _ in range(samples):
        sampled_ids = rng.choice(document_ids, size=len(document_ids), replace=True)
        values = [value for doc_id in sampled_ids for value in by_document[doc_id]]
        draws.append(float(np.mean(values)))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def build_gate_summary(
    trace_rows: list[dict], seed: int, bootstrap_samples: int
) -> list[dict]:
    rng = np.random.default_rng(seed)
    summary = []
    for label, _, _ in DISTANCE_BUCKETS:
        bucket_rows = [
            row for row in trace_rows
            if row["bucket"] == label and row.get("global_gate_weight") is not None
        ]
        values = [float(row["global_gate_weight"]) for row in bucket_rows]
        lower, upper = clustered_bootstrap_ci(
            bucket_rows, "global_gate_weight", rng, bootstrap_samples
        )
        summary.append(
            {
                "bucket": label,
                "mean_global_gate_weight": float(np.mean(values)) if values else float("nan"),
                "ci95_lower": lower,
                "ci95_upper": upper,
                "n_relations": len(values),
                "n_contexts": len({row["context_id"] for row in bucket_rows}),
                "trace_scope": bucket_rows[0]["scope"] if bucket_rows else "unavailable",
            }
        )
    return summary


def build_gate_outcome_summary(
    trace_rows: list[dict], seed: int, bootstrap_samples: int
) -> list[dict]:
    summary = []
    for outcome, recalled in (("Recalled", True), ("Missed", False)):
        subset = [row for row in trace_rows if row.get("is_recalled") is recalled]
        for row in build_gate_summary(subset, seed, bootstrap_samples):
            summary.append({"outcome": outcome, **row})
    return summary


def balanced_length_groups(documents: list[dict]) -> dict[object, str]:
    ordered = sorted(documents, key=lambda item: (int(item["word_count"]), str(item["document_id"])))
    assignments = {}
    for label, indices in zip(("Short", "Medium", "Long"), np.array_split(np.arange(len(ordered)), 3)):
        for index in indices.tolist():
            assignments[ordered[index]["document_id"]] = label
    return assignments


def build_context_rows(
    method_documents: dict[str, list[dict]], assignments: dict[object, str]
) -> list[dict]:
    rows = []
    for method, documents in method_documents.items():
        grouped: dict[str, list[dict]] = defaultdict(list)
        for document in documents:
            grouped[assignments[document["document_id"]]].append(document)
        for group in ("Short", "Medium", "Long"):
            tp = fp = fn = support = 0
            lengths = []
            for document in grouped[group]:
                gold, pred = strict_relation_sets(document)
                gold_keys, pred_keys = set(gold), set(pred)
                tp += len(gold_keys & pred_keys)
                fp += len(pred_keys - gold_keys)
                fn += len(gold_keys - pred_keys)
                support += len(gold_keys)
                lengths.append(int(document["word_count"]))
            precision, recall, f1 = safe_prf(tp, fp, fn)
            rows.append(
                {
                    "method": method,
                    "length_group": group,
                    "min_context_length": min(lengths) if lengths else 0,
                    "max_context_length": max(lengths) if lengths else 0,
                    "n_contexts": len(lengths),
                    "n_gold_relations": support,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    return rows


def build_window_rows(method_documents: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for method, documents in method_documents.items():
        totals = {
            "Same-window": {"tp": 0, "fp": 0, "support": 0},
            "Cross-window": {"tp": 0, "fp": 0, "support": 0},
        }
        for document in documents:
            gold = document["gold"]
            pred = document["prediction"]
            windows = document.get("windows", [])
            pred_categories = {}
            for relation in pred.get("relations", []):
                head = pred["entities"][int(relation["head"])]
                tail = pred["entities"][int(relation["tail"])]
                is_observable = any(
                    int(window["word_start"]) <= int(head["start"])
                    and int(head["end"]) <= int(window["word_end"])
                    and int(window["word_start"]) <= int(tail["start"])
                    and int(tail["end"]) <= int(window["word_end"])
                    for window in windows
                )
                pred_categories[relation_key(relation, pred["entities"])] = (
                    "Same-window" if is_observable else "Cross-window"
                )
            seen = set()
            gold_keys = set()
            for relation in gold.get("relations", []):
                key = relation_key(relation, gold["entities"])
                if key in seen:
                    continue
                seen.add(key)
                gold_keys.add(key)
                category = "Same-window" if relation.get("window_observable", True) else "Cross-window"
                totals[category]["support"] += 1
                totals[category]["tp"] += int(key in pred_categories)
            for key, category in pred_categories.items():
                if key not in gold_keys:
                    totals[category]["fp"] += 1
        for category in ("Same-window", "Cross-window"):
            values = totals[category]
            precision, recall, f1 = safe_prf(
                values["tp"], values["fp"], values["support"] - values["tp"]
            )
            recall = values["tp"] / values["support"] if values["support"] else float("nan")
            rows.append(
                {
                    "method": method,
                    "window_category": category,
                    "tp": values["tp"],
                    "fp": values["fp"],
                    "fn": values["support"] - values["tp"],
                    "support": values["support"],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    return rows


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.7,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_distance(rows: list[dict], reference_method: str, output: Path) -> None:
    labels = [item[0] for item in DISTANCE_BUCKETS]
    x = np.arange(len(labels))
    support = [
        next(row["support"] for row in rows if row["method"] == reference_method and row["bucket"] == label)
        for label in labels
    ]
    fig, axis = plt.subplots(figsize=(7.1, 3.25), constrained_layout=True)
    support_axis = axis.twinx()
    support_axis.set_zorder(1)
    axis.set_zorder(2)
    axis.patch.set_visible(False)
    support_axis.bar(x, support, width=0.62, color="#D9DEE7", edgecolor="#AEB7C4", linewidth=0.6, label="Gold relations")
    support_axis.set_ylabel("Gold relations")
    support_axis.set_ylim(0, max(support) * 1.35 if support else 1)
    support_axis.grid(False)
    for method_index, method in enumerate(dict.fromkeys(row["method"] for row in rows)):
        method_rows = {row["bucket"]: row for row in rows if row["method"] == method}
        values = [100 * method_rows[label]["f1"] for label in labels]
        color = METHOD_COLORS.get(method, mpl.colormaps["tab10"](method_index))
        axis.plot(x, values, marker=METHOD_MARKERS.get(method, "o"), color=color, label=method, zorder=4)
        if method == reference_method:
            for index, value in enumerate(values):
                label = f"{value:.1f}"
                if support[index] < 30:
                    label += f"\nn={support[index]}"
                axis.annotate(label, (x[index], value), xytext=(0, 6), textcoords="offset points", ha="center", color=color, fontsize=7.5)
    axis.set_xticks(x, labels)
    axis.set_xlabel("Entity distance (tokens)")
    axis.set_ylabel("Strict RE-F1 (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    handles, legends = axis.get_legend_handles_labels()
    bar_handles, bar_legends = support_axis.get_legend_handles_labels()
    axis.legend(handles + bar_handles, legends + bar_legends, loc="upper left", ncol=min(4, len(handles) + 1), frameon=False)
    save_figure(fig, output)


def plot_gate(
    summary: list[dict],
    outcome_summary: list[dict],
    distance_rows: list[dict],
    reference_method: str,
    comparison_method: str | None,
    output: Path,
) -> None:
    labels = [item[0] for item in DISTANCE_BUCKETS]
    x = np.arange(len(labels))
    has_comparison = comparison_method is not None
    if has_comparison:
        fig, (axis, gain_axis) = plt.subplots(
            1, 2, figsize=(7.1, 3.05), constrained_layout=True
        )
    else:
        fig, axis = plt.subplots(figsize=(7.1, 3.05), constrained_layout=True)
        gain_axis = None

    outcome_styles = {
        "Recalled": ("#0072B2", "o", "-"),
        "Missed": ("#D55E00", "s", "--"),
    }
    plotted_outcomes = []
    for outcome in ("Recalled", "Missed"):
        rows_by_bucket = {
            row["bucket"]: row for row in outcome_summary if row["outcome"] == outcome
        }
        means = np.array(
            [rows_by_bucket[label]["mean_global_gate_weight"] for label in labels],
            dtype=float,
        )
        if not np.isfinite(means).any():
            continue
        lower = np.array(
            [rows_by_bucket[label]["ci95_lower"] for label in labels], dtype=float
        )
        upper = np.array(
            [rows_by_bucket[label]["ci95_upper"] for label in labels], dtype=float
        )
        errors = np.vstack((means - lower, upper - means))
        color, marker, linestyle = outcome_styles[outcome]
        axis.errorbar(
            x,
            means,
            yerr=errors,
            color=color,
            marker=marker,
            linestyle=linestyle,
            capsize=2.5,
            linewidth=1.6,
            label=f"{outcome} gold relations",
        )
        plotted_outcomes.append(outcome)

    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_xlabel("Entity distance (tokens)")
    axis.set_ylabel("Mean global gate weight")
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    axis.legend(frameon=False, loc="upper left")
    if has_comparison:
        axis.text(-0.18, 1.03, "(a)", transform=axis.transAxes, fontweight="bold")

        reference = {
            row["bucket"]: row
            for row in distance_rows
            if row["method"] == reference_method
        }
        comparison = {
            row["bucket"]: row
            for row in distance_rows
            if row["method"] == comparison_method
        }
        gains = np.array(
            [100 * (reference[label]["f1"] - comparison[label]["f1"]) for label in labels]
        )
        colors = ["#009E73" if gain >= 0 else "#D55E00" for gain in gains]
        bars = gain_axis.bar(x, gains, width=0.68, color=colors)
        gain_axis.axhline(0, color="#4B5563", linewidth=0.8)
        for bar, gain in zip(bars, gains):
            offset = 0.8 if gain >= 0 else -0.8
            gain_axis.text(
                bar.get_x() + bar.get_width() / 2,
                gain + offset,
                f"{gain:+.1f}",
                ha="center",
                va="bottom" if gain >= 0 else "top",
                fontsize=7,
            )
        gain_axis.set_xticks(x, labels, rotation=25, ha="right")
        gain_axis.set_xlabel("Entity distance (tokens)")
        gain_axis.set_ylabel("Strict RE-F1 gain (points)")
        gain_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        gain_axis.text(-0.18, 1.03, "(b)", transform=gain_axis.transAxes, fontweight="bold")
    save_figure(fig, output)


def plot_robustness(context_rows: list[dict], window_rows: list[dict], output: Path) -> None:
    methods = list(dict.fromkeys(row["method"] for row in context_rows))
    groups = ("Short", "Medium", "Long")
    fig, (length_axis, window_axis) = plt.subplots(1, 2, figsize=(7.1, 3.15), constrained_layout=True)
    x = np.arange(len(groups))
    for method_index, method in enumerate(methods):
        values = [100 * next(row["f1"] for row in context_rows if row["method"] == method and row["length_group"] == group) for group in groups]
        color = METHOD_COLORS.get(method, mpl.colormaps["tab10"](method_index))
        length_axis.plot(x, values, marker=METHOD_MARKERS.get(method, "o"), color=color, label=method)
        if method_index == 0:
            for position, value in zip(x, values):
                length_axis.annotate(
                    f"{value:.1f}",
                    (position, value),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    color=color,
                    fontsize=7.2,
                )
    reference = methods[0]
    reference_rows = {row["length_group"]: row for row in context_rows if row["method"] == reference}
    tick_labels = [
        f"{group}\n{reference_rows[group]['min_context_length']}-{reference_rows[group]['max_context_length']}\n"
        f"{reference_rows[group]['n_contexts']} ctx / {reference_rows[group]['n_gold_relations']} rel"
        for group in groups
    ]
    length_axis.set_xticks(x, tick_labels)
    length_axis.set_ylabel("Strict RE-F1 (%)")
    length_axis.set_xlabel("Context-length quantile")
    length_axis.set_ylim(0, 100)
    length_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    length_axis.text(-0.16, 1.03, "(a)", transform=length_axis.transAxes, fontweight="bold")
    length_axis.legend(frameon=False, loc="lower left")

    categories = ("Same-window", "Cross-window")
    group_width = 0.78
    width = group_width / max(len(methods), 1)
    for method_index, method in enumerate(methods):
        values = [100 * next(row["f1"] for row in window_rows if row["method"] == method and row["window_category"] == category) for category in categories]
        positions = np.arange(len(categories)) - group_width / 2 + width / 2 + method_index * width
        color = METHOD_COLORS.get(method, mpl.colormaps["tab10"](method_index))
        bars = window_axis.bar(positions, values, width=width * 0.9, color=color, label=method)
        for bar, value in zip(bars, values):
            window_axis.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}", ha="center", va="bottom", fontsize=7.2)
    reference_window = {row["window_category"]: row for row in window_rows if row["method"] == reference}
    window_axis.set_xticks(
        np.arange(len(categories)),
        [f"{category}\nn={reference_window[category]['support']}" for category in categories],
    )
    window_axis.set_ylabel("Strict RE-F1 (%)")
    window_axis.set_xlabel("Endpoint visibility")
    window_axis.set_ylim(0, 100)
    window_axis.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    window_axis.text(-0.16, 1.03, "(b)", transform=window_axis.transAxes, fontweight="bold")
    save_figure(fig, output)


def write_manifest(
    path: Path,
    prediction_inputs: list[tuple[str, Path]],
    gate_summary: list[dict],
) -> None:
    scope = next((row["trace_scope"] for row in gate_summary if row["trace_scope"] != "unavailable"), "unavailable")
    payload = {
        "prediction_inputs": {method: str(source.resolve()) for method, source in prediction_inputs},
        "distance_definition": "absolute difference between global entity-center token offsets",
        "distance_buckets": [item[0] for item in DISTANCE_BUCKETS],
        "gate_definition": "global_gate_weight = 1 - model fusion_gate; model fusion_gate is the local-branch weight",
        "gate_trace_scope": scope,
        "length_grouping": "three balanced rank-based context-length groups; metrics are micro-aggregated",
        "cross_window_definition": "gold relation window_observable=false (no actual encoder window jointly covers both endpoints)",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        action="append",
        type=parse_method_path,
        required=True,
        help="METHOD=path/to/test_predictions.jsonl; repeat for baselines",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")

    method_documents = {method: load_jsonl(path) for method, path in args.predictions}
    reference_method = args.predictions[0][0]
    reference_documents = method_documents[reference_method]
    reference_ids = {document["document_id"] for document in reference_documents}
    for method, documents in method_documents.items():
        if {document["document_id"] for document in documents} != reference_ids:
            raise ValueError(f"{method} predictions do not cover the same document ids")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    distance_rows = build_distance_rows(method_documents)
    gate_trace = build_gate_trace(reference_method, reference_documents)
    gate_summary = build_gate_summary(gate_trace, args.seed, args.bootstrap_samples)
    gate_outcome_summary = build_gate_outcome_summary(
        gate_trace, args.seed, args.bootstrap_samples
    )
    assignments = balanced_length_groups(reference_documents)
    context_rows = build_context_rows(method_documents, assignments)
    window_rows = build_window_rows(method_documents)

    write_csv(args.output_dir / "distance_eval.csv", distance_rows)
    write_csv(args.output_dir / "gate_trace.csv", gate_trace)
    write_csv(args.output_dir / "gate_distance_summary.csv", gate_summary)
    write_csv(args.output_dir / "gate_outcome_summary.csv", gate_outcome_summary)
    write_csv(args.output_dir / "context_robustness.csv", context_rows)
    write_csv(args.output_dir / "window_relation_eval.csv", window_rows)
    write_manifest(args.output_dir / "figure_data_manifest.json", args.predictions, gate_summary)

    configure_matplotlib()
    plot_distance(distance_rows, reference_method, args.figure_dir / "fig_distance_performance")
    comparison_method = args.predictions[1][0] if len(args.predictions) > 1 else None
    plot_gate(
        gate_summary,
        gate_outcome_summary,
        distance_rows,
        reference_method,
        comparison_method,
        args.figure_dir / "fig_global_gate_behavior",
    )
    plot_robustness(context_rows, window_rows, args.figure_dir / "fig_long_context_robustness")
    print(f"Data written to: {args.output_dir.resolve()}")
    print(f"Figures written to: {args.figure_dir.resolve()}")
    print(f"Gate trace scope: {gate_summary[0]['trace_scope'] if gate_summary else 'unavailable'}")


if __name__ == "__main__":
    main()
