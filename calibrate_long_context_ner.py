#!/usr/bin/env python3
"""Calibrate long-context NER decoding on the eval/dev split only.

The inference command can dump raw span/type evidence once.  This utility
then evaluates a threshold/overlap grid without loading the model again and
refuses evidence generated from the test split.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

from infer_long_context import _merge_document
from iter.misc.document_metrics import entity_key, safe_prf


def _restore_evidence(value: dict) -> dict:
    """Restore tuple keys that ``infer_long_context`` serializes as lists."""
    evidence = copy.deepcopy(value)
    for entity in evidence.get("entities", []):
        if isinstance(entity.get("key"), list):
            entity["key"] = tuple(entity["key"])
    for key in ("relations", "relation_scores"):
        for relation in evidence.get(key, []):
            for field in ("head_key", "tail_key"):
                if isinstance(relation.get(field), list):
                    relation[field] = tuple(relation[field])
    return evidence


def _prepare_documents(payload: dict) -> list[tuple[int, dict, list[dict], dict]]:
    """Normalize JSON evidence once before evaluating the threshold grid."""
    prepared = []
    for document_id, raw in enumerate(payload["raw_documents"]):
        evidence = _restore_evidence(payload["evidence"].get(str(document_id), {}))
        records = payload["window_records"].get(str(document_id), [])
        prepared.append((document_id, raw, records, evidence))
    return prepared


def _build_documents(
    prepared_documents: list[tuple[int, dict, list[dict], dict]],
    span_threshold: float,
    type_threshold: float,
    strategy: str,
    long_threshold: int,
    joint_rerank_weight: float = 0.0,
    type_margin_weight: float = 0.0,
    boundary_rerank_weight: float = 0.0,
    type_rerank_weight: float = 0.0,
    boundary_shift_penalty: float = 0.0,
    type_confusion_penalty: float = 0.0,
):
    documents = []
    for document_id, raw, records, base_evidence in prepared_documents:
        # The dump contains all positive-probability candidates.  Apply the
        # requested thresholds here, before the document-level merger.
        filtered_entities = []
        for entity in base_evidence.get("entities", []):
            span_confidence = entity.get("span_confidence")
            type_confidence = entity.get("type_confidence")
            if span_confidence is not None and span_confidence < span_threshold:
                continue
            if type_confidence is not None and type_confidence < type_threshold:
                continue
            filtered_entities.append(entity)
        # The merger is read-only with respect to relation evidence.  A
        # shallow copy is sufficient and avoids copying the full evidence
        # graph for every grid point.
        evidence = dict(base_evidence)
        evidence["entities"] = filtered_entities
        merged = _merge_document(
            raw,
            records,
            evidence,
            set(),
            ner_overlap_strategy=strategy,
            ner_joint_rerank_weight=joint_rerank_weight,
            ner_type_margin_weight=type_margin_weight,
            ner_boundary_rerank_weight=boundary_rerank_weight,
            ner_type_rerank_weight=type_rerank_weight,
            ner_boundary_shift_penalty=boundary_shift_penalty,
            ner_type_confusion_penalty=type_confusion_penalty,
            include_relations=False,
        )
        document = {
            "document_id": document_id,
            "word_count": len(raw["tokens"]),
            "is_long": len(raw["tokens"]) >= long_threshold,
            **merged,
        }
        documents.append(document)
    return documents


def _ner_overall(documents: list[dict]) -> dict:
    """Compute only the two NER metrics needed for threshold selection."""
    strict_tp = strict_fp = strict_fn = 0
    boundary_tp = boundary_fp = boundary_fn = 0
    for document in documents:
        gold_entities = set(entity_key(entity) for entity in document["gold"]["entities"])
        pred_entities = set(entity_key(entity) for entity in document["prediction"]["entities"])
        gold_bounds = set(entity_key(entity, False) for entity in document["gold"]["entities"])
        pred_bounds = set(entity_key(entity, False) for entity in document["prediction"]["entities"])
        strict_tp += len(gold_entities & pred_entities)
        strict_fp += len(pred_entities - gold_entities)
        strict_fn += len(gold_entities - pred_entities)
        boundary_tp += len(gold_bounds & pred_bounds)
        boundary_fp += len(pred_bounds - gold_bounds)
        boundary_fn += len(gold_bounds - pred_bounds)
    return {
        "ner_strict": safe_prf(strict_tp, strict_fp, strict_fn),
        "ner_boundary": safe_prf(boundary_tp, boundary_fp, boundary_fn),
    }


def _candidate_diagnostics(
    prepared_documents,
    documents,
    span_threshold: float,
    type_threshold: float,
) -> dict:
    """Separate candidate absence, threshold filtering, and NMS loss.

    This uses gold spans only on eval/dev calibration data and is never called
    during test inference.  It tells us whether reranking can help at all:
    a gold span absent from raw candidates requires model/candidate changes,
    not decoder changes.
    """
    totals = {"gold": 0, "raw_boundary": 0, "raw_strict": 0,
              "threshold_boundary": 0, "threshold_strict": 0,
              "pred_boundary": 0, "pred_strict": 0}
    for (_, raw, _, evidence), document in zip(prepared_documents, documents):
        gold_entities = {(e["start"], e["end"], e["type"]) for e in raw.get("entities", [])}
        gold_bounds = {(e["start"], e["end"]) for e in raw.get("entities", [])}
        raw_entities = evidence.get("entities", [])
        threshold_entities = [
            e for e in raw_entities
            if (e.get("span_confidence") is None or e.get("span_confidence") >= span_threshold)
            and (e.get("type_confidence") is None or e.get("type_confidence") >= type_threshold)
        ]
        raw_boundaries = {(e["start"], e["end"]) for e in raw_entities}
        raw_strict = {(e["start"], e["end"], e["type"]) for e in raw_entities}
        threshold_boundaries = {(e["start"], e["end"]) for e in threshold_entities}
        threshold_strict = {(e["start"], e["end"], e["type"]) for e in threshold_entities}
        pred = document["prediction"]["entities"]
        pred_boundaries = {(e["start"], e["end"]) for e in pred}
        pred_strict = {(e["start"], e["end"], e["type"]) for e in pred}
        totals["gold"] += len(gold_entities)
        totals["raw_boundary"] += len(gold_bounds & raw_boundaries)
        totals["raw_strict"] += len(gold_entities & raw_strict)
        totals["threshold_boundary"] += len(gold_bounds & threshold_boundaries)
        totals["threshold_strict"] += len(gold_entities & threshold_strict)
        totals["pred_boundary"] += len(gold_bounds & pred_boundaries)
        totals["pred_strict"] += len(gold_entities & pred_strict)
    gold = max(1, totals["gold"])
    totals["raw_boundary_recall"] = totals["raw_boundary"] / gold
    totals["raw_strict_recall"] = totals["raw_strict"] / gold
    totals["threshold_boundary_recall"] = totals["threshold_boundary"] / gold
    totals["threshold_strict_recall"] = totals["threshold_strict"] / gold
    totals["pred_boundary_recall"] = totals["pred_boundary"] / gold
    totals["pred_strict_recall"] = totals["pred_strict"] / gold
    totals["lost_to_threshold_boundary"] = totals["raw_boundary"] - totals["threshold_boundary"]
    totals["lost_to_threshold_strict"] = totals["raw_strict"] - totals["threshold_strict"]
    totals["lost_to_decoder_boundary"] = totals["threshold_boundary"] - totals["pred_boundary"]
    totals["lost_to_decoder_strict"] = totals["threshold_strict"] - totals["pred_strict"]
    return totals


def _error_breakdown(documents: list[dict]) -> dict:
    """Count the two strict-NER error modes after decoding."""
    boundary_type_errors = 0
    type_correct_boundary_errors = 0
    exact = 0
    for document in documents:
        gold = document["gold"]["entities"]
        pred = document["prediction"]["entities"]
        gold_by_boundary = {(e["start"], e["end"]): e for e in gold}
        pred_by_boundary = {(e["start"], e["end"]): e for e in pred}
        boundary_type_errors += sum(
            1 for boundary in gold_by_boundary.keys() & pred_by_boundary.keys()
            if gold_by_boundary[boundary]["type"] != pred_by_boundary[boundary]["type"]
        )
        gold_typed_bounds = {(e["type"], e["start"], e["end"]) for e in gold}
        gold_type_spans = defaultdict(set)
        for e in gold:
            gold_type_spans[e["type"]].add((e["start"], e["end"]))
        for e in pred:
            if (e["type"], e["start"], e["end"]) in gold_typed_bounds:
                exact += 1
            elif e["type"] in gold_type_spans and gold_type_spans[e["type"]]:
                type_correct_boundary_errors += 1
    return {
        "boundary_correct_type_errors": boundary_type_errors,
        "type_correct_boundary_errors": type_correct_boundary_errors,
        "strict_exact_entities": exact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True, help="Evidence dump generated from --split eval")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/calibration/long_context_ner_dev_calibration.json"),
    )
    parser.add_argument(
        "--span_thresholds",
        type=float,
        nargs="+",
        default=[0.40, 0.50, 0.60],
        help="Span threshold grid (use the previous 5-point grid explicitly for a full sweep)",
    )
    parser.add_argument(
        "--type_thresholds",
        type=float,
        nargs="+",
        default=[0.00, 0.10],
        help="Type threshold grid (use more values explicitly for a full sweep)",
    )
    parser.add_argument(
        "--overlap_strategies",
        nargs="+",
        choices=("none", "nms", "weighted_interval"),
        default=["none", "nms", "weighted_interval"],
    )
    parser.add_argument("--long_threshold", type=int, default=128)
    parser.add_argument(
        "--joint_rerank_weights", type=float, nargs="+", default=[0.0],
        help="Eval-only blend weights for span/type joint reranking",
    )
    parser.add_argument(
        "--type_margin_weights", type=float, nargs="+", default=[0.0],
        help="Eval-only weights for top1-top2 type margin reranking",
    )
    parser.add_argument("--boundary_rerank_weights", type=float, nargs="+", default=[0.0])
    parser.add_argument("--type_rerank_weights", type=float, nargs="+", default=[0.0])
    parser.add_argument("--boundary_shift_penalties", type=float, nargs="+", default=[0.0])
    parser.add_argument("--type_confusion_penalties", type=float, nargs="+", default=[0.0])
    parser.add_argument("--quiet", action="store_true", help="Suppress per-grid progress output")
    args = parser.parse_args()

    evidence_size = args.evidence.stat().st_size
    if not args.quiet:
        print(
            f"Reading evidence: {args.evidence} "
            f"({evidence_size / (1024 ** 3):.2f} GiB)",
            flush=True,
        )
        if evidence_size >= 512 * 1024 * 1024:
            print(
                "WARNING: this is a large/legacy evidence dump. "
                "NER calibration only needs entities; regenerate it with the "
                "current infer_long_context.py to omit relation scores.",
                flush=True,
            )
    payload = json.loads(args.evidence.read_text(encoding="utf-8"))
    if payload.get("split") not in {"eval", "dev"}:
        raise SystemExit(
            f"Refusing calibration evidence from split={payload.get('split')!r}; use the eval/dev split, never test."
        )
    for value in [*args.span_thresholds, *args.type_thresholds]:
        if not 0.0 <= value <= 1.0:
            raise SystemExit("All thresholds must be between 0 and 1")

    prepared_documents = _prepare_documents(payload)
    span_thresholds = sorted(set(args.span_thresholds))
    type_thresholds = sorted(set(args.type_thresholds))
    combinations = [
        (strategy, span_threshold, type_threshold, joint_weight, margin_weight)
        for strategy in args.overlap_strategies
        for span_threshold in span_thresholds
        for type_threshold in type_thresholds
        for joint_weight in sorted(set(args.joint_rerank_weights))
        for margin_weight in sorted(set(args.type_margin_weights))
        for boundary_weight in sorted(set(args.boundary_rerank_weights))
        for type_weight in sorted(set(args.type_rerank_weights))
        for shift_penalty in sorted(set(args.boundary_shift_penalties))
        for confusion_penalty in sorted(set(args.type_confusion_penalties))
    ]
    if not args.quiet:
        entity_count = sum(len(item[3].get("entities", [])) for item in prepared_documents)
        print(
            f"Loaded {len(prepared_documents)} documents and {entity_count} NER candidates; "
            f"evaluating {len(combinations)} combinations.",
            flush=True,
        )

    rows = []
    for combination_index, (strategy, span_threshold, type_threshold, joint_weight, margin_weight, boundary_weight, type_weight, shift_penalty, confusion_penalty) in enumerate(combinations, start=1):
        started = time.perf_counter()
        if not args.quiet:
            print(
                f"[{combination_index}/{len(combinations)}] "
                f"strategy={strategy} span={span_threshold:.2f} type={type_threshold:.2f} "
                f"joint={joint_weight:.2f} margin={margin_weight:.2f}",
                flush=True,
            )
        documents = _build_documents(
            prepared_documents,
            span_threshold,
            type_threshold,
            strategy,
            args.long_threshold,
            joint_weight,
            margin_weight,
            boundary_weight,
            type_weight,
            shift_penalty,
            confusion_penalty,
        )
        overall = _ner_overall(documents)
        rows.append({
            "overlap_strategy": strategy,
            "span_threshold": span_threshold,
            "type_threshold": type_threshold,
            "joint_rerank_weight": joint_weight,
            "type_margin_weight": margin_weight,
            "boundary_rerank_weight": boundary_weight,
            "type_rerank_weight": type_weight,
            "boundary_shift_penalty": shift_penalty,
            "type_confusion_penalty": confusion_penalty,
            "ner_strict": overall["ner_strict"],
            "ner_boundary": overall["ner_boundary"],
            "documents": len(documents),
        })
        if not args.quiet:
            elapsed = time.perf_counter() - started
            print(f"    completed in {elapsed:.2f}s", flush=True)

    def metric_key(row):
        return (
            row["ner_strict"]["f1"],
            row["ner_boundary"]["f1"],
            row["ner_strict"]["precision"],
        )

    rows.sort(key=metric_key, reverse=True)
    best = rows[0] if rows else None
    plateau = []
    if best is not None:
        # Threshold grids often contain a broad exact plateau. Selecting the
        # smallest threshold as a tie-breaker incorrectly marks that stable
        # plateau as an unchecked boundary. Prefer an interior, conservative
        # point; retain boundary warnings when no equally good interior point
        # exists.
        top_key = metric_key(best)
        plateau = [row for row in rows if metric_key(row) == top_key]
        span_edges = {min(span_thresholds), max(span_thresholds)}
        type_edges = {min(type_thresholds), max(type_thresholds)}
        joint_edges = {min(args.joint_rerank_weights), max(args.joint_rerank_weights)}
        margin_edges = {min(args.type_margin_weights), max(args.type_margin_weights)}
        boundary_edges = {min(args.boundary_rerank_weights), max(args.boundary_rerank_weights)}
        type_edges2 = {min(args.type_rerank_weights), max(args.type_rerank_weights)}
        shift_edges = {min(args.boundary_shift_penalties), max(args.boundary_shift_penalties)}
        confusion_edges = {min(args.type_confusion_penalties), max(args.type_confusion_penalties)}
        interior = [
            row for row in plateau
            if row["span_threshold"] not in span_edges
            and row["type_threshold"] not in type_edges
            and row["joint_rerank_weight"] not in joint_edges
            and row["type_margin_weight"] not in margin_edges
            and row["boundary_rerank_weight"] not in boundary_edges
            and row["type_rerank_weight"] not in type_edges2
            and row["boundary_shift_penalty"] not in shift_edges
            and row["type_confusion_penalty"] not in confusion_edges
        ]
        if interior:
            strategy_rank = {
                name: len(args.overlap_strategies) - index
                for index, name in enumerate(args.overlap_strategies)
            }
            best = max(
                interior,
                key=lambda row: (
                    strategy_rank.get(row["overlap_strategy"], 0),
                    row["span_threshold"],
                    row["type_threshold"],
                ),
            )
    boundary = {
        "span_threshold": bool(
            best
            and best["span_threshold"] in {min(span_thresholds), max(span_thresholds)}
        ),
        "type_threshold": bool(
            best
            and best["type_threshold"] in {min(type_thresholds), max(type_thresholds)}
        ),
        "joint_rerank_weight": bool(
            best
            and best["joint_rerank_weight"] in {min(args.joint_rerank_weights), max(args.joint_rerank_weights)}
        ),
        "type_margin_weight": bool(
            best
            and best["type_margin_weight"] in {min(args.type_margin_weights), max(args.type_margin_weights)}
        ),
        "boundary_rerank_weight": bool(
            best
            and best["boundary_rerank_weight"] in {min(args.boundary_rerank_weights), max(args.boundary_rerank_weights)}
        ),
        "type_rerank_weight": bool(
            best
            and best["type_rerank_weight"] in {min(args.type_rerank_weights), max(args.type_rerank_weights)}
        ),
        "boundary_shift_penalty": bool(
            best
            and best["boundary_shift_penalty"] in {min(args.boundary_shift_penalties), max(args.boundary_shift_penalties)}
        ),
        "type_confusion_penalty": bool(
            best
            and best["type_confusion_penalty"] in {min(args.type_confusion_penalties), max(args.type_confusion_penalties)}
        ),
    }
    result = {
        "source_split": payload["split"],
        "evidence": str(args.evidence.resolve()),
        "selection_metric": "overall.ner_strict.f1, then ner_boundary.f1, then precision",
        "selection_plateau_size": len(plateau),
        "best_at_search_boundary": any(boundary.values()),
        "boundary_parameters": boundary,
        "search_space": {
            "span_thresholds": span_thresholds,
            "type_thresholds": type_thresholds,
            "overlap_strategies": args.overlap_strategies,
            "joint_rerank_weights": sorted(set(args.joint_rerank_weights)),
            "type_margin_weights": sorted(set(args.type_margin_weights)),
            "boundary_rerank_weights": sorted(set(args.boundary_rerank_weights)),
            "type_rerank_weights": sorted(set(args.type_rerank_weights)),
            "boundary_shift_penalties": sorted(set(args.boundary_shift_penalties)),
            "type_confusion_penalties": sorted(set(args.type_confusion_penalties)),
        },
        "best": best,
        "grid": rows,
    }
    if best is not None:
        best_documents = _build_documents(
            prepared_documents,
            best["span_threshold"],
            best["type_threshold"],
            best["overlap_strategy"],
            args.long_threshold,
            best["joint_rerank_weight"],
            best["type_margin_weight"],
            best["boundary_rerank_weight"],
            best["type_rerank_weight"],
            best["boundary_shift_penalty"],
            best["type_confusion_penalty"],
        )
        result["candidate_diagnostics"] = _candidate_diagnostics(
            prepared_documents,
            best_documents,
            best["span_threshold"],
            best["type_threshold"],
        )
        result["error_breakdown"] = _error_breakdown(best_documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"source_split": result["source_split"], "best": result["best"]}, ensure_ascii=False, indent=2))
    if result["best_at_search_boundary"]:
        names = ", ".join(name for name, value in boundary.items() if value)
        print(f"WARNING: best NER setting is on the search boundary: {names}")
    print(f"Calibration written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
