#!/usr/bin/env python3
"""Experiment 4: quantify NER -> RE error propagation with entity oracles.

The oracle is diagnostic only.  It keeps the relation candidates emitted by the
model, but remaps an emitted relation to gold entities whenever both endpoint
boundaries are present.  It therefore does not invent missing relation
candidates and does not use test data to tune any threshold.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

_METRICS_PATH = Path(__file__).resolve().parents[1] / "iter" / "misc" / "document_metrics.py"
_SPEC = importlib.util.spec_from_file_location("long_context_document_metrics", _METRICS_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MODULE)
evaluate_documents, entity_key, relation_key = _MODULE.evaluate_documents, _MODULE.entity_key, _MODULE.relation_key


TASKS = (
    "ner_strict",
    "ner_boundary",
    "re_strict_end_to_end",
    "re_boundary_end_to_end",
    "re_on_matched_entities",
)


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _gold_entity_oracle(document: dict) -> dict:
    """Replace predicted entities by gold entities and remap emitted relations."""
    gold = document["gold"]
    pred = document["prediction"]
    gold_by_boundary = {(e["start"], e["end"]): i for i, e in enumerate(gold["entities"])}
    remapped = []
    for relation in pred.get("relations", []):
        try:
            head = pred["entities"][relation["head"]]
            tail = pred["entities"][relation["tail"]]
        except (IndexError, KeyError, TypeError):
            continue
        head_id = gold_by_boundary.get((head["start"], head["end"]))
        tail_id = gold_by_boundary.get((tail["start"], tail["end"]))
        if head_id is None or tail_id is None:
            continue
        item = dict(relation)
        item["head"] = head_id
        item["tail"] = tail_id
        remapped.append(item)
    oracle = dict(document)
    oracle["prediction"] = {"entities": list(gold["entities"]), "relations": remapped}
    return oracle


def _boundary_oracle(document: dict) -> dict:
    """Use gold boundaries while retaining predicted type when available.

    This isolates endpoint-boundary propagation.  A boundary that was not
    emitted by NER is still present in this diagnostic oracle, but its type is
    copied from gold because no predicted type exists for that boundary.
    """
    gold = document["gold"]
    pred = document["prediction"]
    gold_by_boundary = {(e["start"], e["end"]): e for e in gold["entities"]}
    pred_by_boundary = {(e["start"], e["end"]): e for e in pred["entities"]}
    entities = []
    for gold_entity in gold["entities"]:
        boundary = (gold_entity["start"], gold_entity["end"])
        entity = dict(gold_entity)
        if boundary in pred_by_boundary:
            entity["type"] = pred_by_boundary[boundary]["type"]
        entities.append(entity)
    gold_index = {(e["start"], e["end"]): i for i, e in enumerate(gold["entities"])}
    remapped = []
    for relation in pred.get("relations", []):
        head = pred["entities"][relation["head"]]
        tail = pred["entities"][relation["tail"]]
        head_id = gold_index.get((head["start"], head["end"]))
        tail_id = gold_index.get((tail["start"], tail["end"]))
        if head_id is None or tail_id is None:
            continue
        item = dict(relation)
        item["head"], item["tail"] = head_id, tail_id
        remapped.append(item)
    oracle = dict(document)
    oracle["prediction"] = {"entities": entities, "relations": remapped}
    return oracle


def _error_breakdown(documents: list[dict]) -> tuple[Counter, list[dict]]:
    counts: Counter = Counter()
    rows: list[dict] = []
    for document in documents:
        gold = document["gold"]
        pred = document["prediction"]
        pred_bounds = {(e["start"], e["end"]): e for e in pred["entities"]}
        pred_relations = {relation_key(r, pred["entities"], True) for r in pred.get("relations", [])}
        pred_relations_boundary = {relation_key(r, pred["entities"], False) for r in pred.get("relations", [])}
        seen_gold_keys = set()
        for index, relation in enumerate(gold["relations"]):
            gold_key = relation_key(relation, gold["entities"], True)
            if gold_key in seen_gold_keys:
                continue
            seen_gold_keys.add(gold_key)
            head = gold["entities"][relation["head"]]
            tail = gold["entities"][relation["tail"]]
            hb = (head["start"], head["end"])
            tb = (tail["start"], tail["end"])
            h_ok, t_ok = hb in pred_bounds, tb in pred_bounds
            if not h_ok and not t_ok:
                category = "both_endpoint_boundary_missed"
            elif not h_ok:
                category = "head_boundary_missed"
            elif not t_ok:
                category = "tail_boundary_missed"
            else:
                hp = pred_bounds[hb]["type"] == head["type"]
                tp = pred_bounds[tb]["type"] == tail["type"]
                if not hp or not tp:
                    category = "endpoint_type_error"
                else:
                    key = relation_key(relation, gold["entities"], True)
                    category = "relation_type_or_link_error" if key not in pred_relations else "correct"
            counts[category] += 1
            rows.append({
                "document_id": document["document_id"],
                "relation_index": index,
                "relation_type": relation["type"],
                "head_text": head.get("text", ""),
                "tail_text": tail.get("text", ""),
                "distance": abs((head["start"] + head["end"]) // 2 - (tail["start"] + tail["end"]) // 2),
                "window_observable": relation.get("window_observable", True),
                "error_category": category,
            })
    return counts, rows


def _write_metrics_csv(path: Path, metrics: dict) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "task", "precision", "recall", "f1", "tp", "fp", "fn", "support"])
        for variant, value in metrics.items():
            for task in TASKS:
                score = value["overall"][task]
                writer.writerow([variant, task, score["precision"], score["recall"], score["f1"], score["tp"], score["fp"], score["fn"], score["support"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    documents = _load(args.predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    length_edges = (128, 256, 512, 768, 1024)
    distance_edges = (32, 64, 96, 128, 192, 256)
    variants = {
        "baseline_predicted_entities": documents,
        "boundary_type_oracle": [_boundary_oracle(d) for d in documents],
        "gold_entity_oracle": [_gold_entity_oracle(d) for d in documents],
    }
    metrics = {
        name: evaluate_documents(
            docs,
            long_threshold=256,
            length_buckets=length_edges,
            relation_distance_buckets=distance_edges,
        )
        for name, docs in variants.items()
    }
    counts, rows = _error_breakdown(documents)
    total = sum(counts.values())
    payload = {
        "experiment": "oracle_and_error_propagation",
        "predictions": str(args.predictions),
        "documents": len(documents),
        "oracle_definition": "Keep emitted relation candidates; remap endpoints to gold entities when exact boundaries exist.",
        "variants": metrics,
        "error_breakdown": {k: {"count": v, "ratio": v / total if total else 0.0} for k, v in sorted(counts.items())},
        "ner_to_re_gap": {
            "strict_re_f1_gap_gold_entity_oracle_minus_baseline": metrics["gold_entity_oracle"]["overall"]["re_strict_end_to_end"]["f1"] - metrics["baseline_predicted_entities"]["overall"]["re_strict_end_to_end"]["f1"],
            "strict_re_f1_baseline": metrics["baseline_predicted_entities"]["overall"]["re_strict_end_to_end"]["f1"],
            "strict_re_f1_gold_entity_oracle": metrics["gold_entity_oracle"]["overall"]["re_strict_end_to_end"]["f1"],
        },
    }
    (args.output_dir / "oracle_error_propagation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_metrics_csv(args.output_dir / "oracle_metrics.csv", metrics)
    with (args.output_dir / "relation_error_breakdown.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["error_category"])
        writer.writeheader()
        writer.writerows(rows)
    md = ["# 实验四：Oracle 与 NER→RE 误差传播", "", f"预测文件：`{args.predictions}`", "", "## 变体指标", "", "|变体|NER strict F1|NER boundary F1|RE strict F1|RE matched F1|", "|---|---:|---:|---:|---:|"]
    for name, value in metrics.items():
        o = value["overall"]
        md.append(f"|{name}|{o['ner_strict']['f1']:.4f}|{o['ner_boundary']['f1']:.4f}|{o['re_strict_end_to_end']['f1']:.4f}|{o['re_on_matched_entities']['f1']:.4f}|")
    md += ["", "## Gold relation 漏检归因", "", "|类别|数量|比例|", "|---|---:|---:|"]
    for key, value in sorted(payload["error_breakdown"].items()):
        md.append(f"|{key}|{value['count']}|{value['ratio']:.4f}|")
    (args.output_dir / "oracle_error_propagation_zh.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(payload["ner_to_re_gap"], ensure_ascii=False, indent=2))
    print(f"Wrote: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
