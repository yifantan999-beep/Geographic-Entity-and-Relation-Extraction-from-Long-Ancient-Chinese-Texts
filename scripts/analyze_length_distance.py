#!/usr/bin/env python3
"""Experiment 5: measure degradation by document length and relation distance."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

_METRICS_PATH = Path(__file__).resolve().parents[1] / "iter" / "misc" / "document_metrics.py"
_SPEC = importlib.util.spec_from_file_location("long_context_document_metrics", _METRICS_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MODULE)
evaluate_documents, relation_key, safe_prf = _MODULE.evaluate_documents, _MODULE.relation_key, _MODULE.safe_prf


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _relation_distance(relation: dict, entities: list[dict]) -> int:
    h, t = entities[relation["head"]], entities[relation["tail"]]
    return abs((h["start"] + h["end"]) // 2 - (t["start"] + t["end"]) // 2)


def _length_label(n: int) -> str:
    for lo, hi in ((0, 128), (128, 256), (256, 512), (512, 768), (768, 1024)):
        if lo <= n < hi:
            return f"{lo}-{hi - 1}"
    return ">=1024"


def _window_sets(document: dict) -> dict[int, set[int]]:
    output: dict[int, set[int]] = defaultdict(set)
    for window in document.get("windows", []):
        for entity_id in window.get("source_entity_ids", []):
            output[int(entity_id)].add(int(window.get("window_id", 0)))
    return output


def _cross_bucket(document: dict, relation: dict, entities: list[dict]) -> str:
    sets = _window_sets(document)
    gold_entities = document["gold"]["entities"]
    gold_by_boundary = {(e["start"], e["end"]): i for i, e in enumerate(gold_entities)}
    head, tail = entities[relation["head"]], entities[relation["tail"]]
    h_id = gold_by_boundary.get((head["start"], head["end"]))
    t_id = gold_by_boundary.get((tail["start"], tail["end"]))
    if h_id is None or t_id is None:
        return "unmatched_prediction"
    h_windows = sets.get(int(h_id), set())
    t_windows = sets.get(int(t_id), set())
    if not h_windows or not t_windows:
        return "uncovered"
    gap = min(abs(a - b) for a in h_windows for b in t_windows)
    if gap == 0:
        return "same_window"
    if gap == 1:
        return "adjacent_windows"
    return "two_or_more_windows"


def _cross_window_metrics(documents: list[dict]) -> dict:
    """Report gold-support recall and candidate precision for observability buckets."""
    output = {}
    buckets = ("same_window", "adjacent_windows", "two_or_more_windows", "uncovered", "unmatched_prediction")
    for bucket in buckets:
        tp = fn = pred = 0
        documents_with_support = set()
        for document in documents:
            gold = document["gold"]
            pred_rel = document["prediction"].get("relations", [])
            gold_keys = set()
            for relation in gold["relations"]:
                if _cross_bucket(document, relation, gold["entities"]) == bucket:
                    gold_keys.add(relation_key(relation, gold["entities"], True))
                    documents_with_support.add(document["document_id"])
            pred_keys = {
                relation_key(relation, document["prediction"]["entities"], True)
                for relation in pred_rel
                if _cross_bucket(document, relation, document["prediction"]["entities"]) == bucket
            }
            hit = len(gold_keys & pred_keys)
            tp += hit
            fn += len(gold_keys - pred_keys)
            pred += len(pred_keys)
        score = safe_prf(tp, pred - tp, fn)
        output[bucket] = {**score, "documents": len(documents_with_support), "note": "cross-window endpoint bucket; unmatched predictions are reported separately"}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    documents = _load(args.predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_documents(
        documents,
        long_threshold=256,
        length_buckets=(128, 256, 512, 768, 1024),
        relation_distance_buckets=(32, 64, 96, 128, 192, 256),
    )
    length_rows = []
    for label, values in metrics["length_buckets"].items():
        row = {"bucket": label, "documents": values["documents"]}
        for task in ("ner_strict", "ner_boundary", "re_strict_end_to_end", "re_boundary_end_to_end", "re_on_matched_entities"):
            row[f"{task}_f1"] = values[task]["f1"]
            row[f"{task}_precision"] = values[task]["precision"]
            row[f"{task}_recall"] = values[task]["recall"]
            row[f"{task}_support"] = values[task]["support"]
        length_rows.append(row)
    distance_rows = []
    for label, values in metrics["relation_distance_buckets"].items():
        distance_rows.append({"bucket": label, "precision": values["precision"], "recall": values["recall"], "f1": values["f1"], "tp": values["tp"], "fp": values["fp"], "fn": values["fn"], "support": values["support"]})
    cross = _cross_window_metrics(documents)
    payload = {
        "experiment": "length_and_distance_analysis",
        "predictions": str(args.predictions),
        "documents": len(documents),
        "overall": metrics["overall"],
        "length_buckets": metrics["length_buckets"],
        "relation_distance_buckets": metrics["relation_distance_buckets"],
        "cross_window_buckets": cross,
        "window_relation_coverage": metrics["window_relation_coverage"],
        "macro_average": metrics["macro_average"],
    }
    (args.output_dir / "length_distance_analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, rows in (("length_buckets.csv", length_rows), ("distance_buckets.csv", distance_rows)):
        with (args.output_dir / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["bucket"])
            writer.writeheader()
            writer.writerows(rows)
    with (args.output_dir / "cross_window_buckets.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bucket", "precision", "recall", "f1", "tp", "fp", "fn", "support", "predicted", "documents", "note"], extrasaction="ignore")
        writer.writeheader()
        for bucket, row in cross.items():
            writer.writerow({"bucket": bucket, **row})
    md = ["# 实验五：长度、关系距离与跨窗口退化分析", "", f"预测文件：`{args.predictions}`", "", "## 文档长度分桶", "", "|长度|文档数|NER strict F1|NER boundary F1|RE strict F1|RE matched F1|", "|---|---:|---:|---:|---:|---:|"]
    for row in length_rows:
        md.append(f"|{row['bucket']}|{row['documents']}|{row['ner_strict_f1']:.4f}|{row['ner_boundary_f1']:.4f}|{row['re_strict_end_to_end_f1']:.4f}|{row['re_on_matched_entities_f1']:.4f}|")
    md += ["", "## 关系距离分桶", "", "|距离|P|R|F1|support|", "|---|---:|---:|---:|---:|"]
    for row in distance_rows:
        md.append(f"|{row['bucket']}|{row['precision']:.4f}|{row['recall']:.4f}|{row['f1']:.4f}|{row['support']}|")
    md += ["", "## 跨窗口分桶", "", "|类别|P|R|F1|support|", "|---|---:|---:|---:|---:|"]
    for bucket, row in cross.items():
        md.append(f"|{bucket}|{row['precision']:.4f}|{row['recall']:.4f}|{row['f1']:.4f}|{row['support']}|")
    (args.output_dir / "length_distance_analysis_zh.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"length_buckets": metrics["length_buckets"], "relation_distance_buckets": metrics["relation_distance_buckets"], "cross_window_buckets": cross}, ensure_ascii=False, indent=2))
    print(f"Wrote: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
