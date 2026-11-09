"""Build an explicitly biased easy/hard diagnostic split from model outputs.

This utility MUST NOT be used to construct an unbiased test set.  It consumes
predictions on the candidate holdout and selects the half that maximizes model
metrics.  The output is useful only for easy-vs-hard diagnostic analysis and
is marked as invalid for primary reporting in its dataset card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any


def entity_key(entity: dict[str, Any], with_type: bool = True) -> tuple:
    key = (int(entity["start"]), int(entity["end"]))
    return key + (str(entity["type"]),) if with_type else key


def relation_key(
    relation: dict[str, Any], entities: list[dict[str, Any]], with_type: bool = True
) -> tuple:
    head = entity_key(entities[int(relation["head"])], with_type)
    tail = entity_key(entities[int(relation["tail"])], with_type)
    if bool(relation.get("symmetric", False)) and tail < head:
        head, tail = tail, head
    return head, str(relation["type"]), tail


def set_counts(gold: set[tuple], predicted: set[tuple]) -> tuple[int, int, int]:
    return len(gold & predicted), len(predicted - gold), len(gold - predicted)


def prediction_counts(row: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    gold, predicted = row["gold"], row["prediction"]
    gold_entities = {entity_key(entity) for entity in gold["entities"]}
    predicted_entities = {entity_key(entity) for entity in predicted["entities"]}
    gold_bounds = {entity_key(entity, False) for entity in gold["entities"]}
    predicted_bounds = {
        entity_key(entity, False) for entity in predicted["entities"]
    }
    gold_relations = {
        relation_key(relation, gold["entities"]) for relation in gold["relations"]
    }
    predicted_relations = {
        relation_key(relation, predicted["entities"])
        for relation in predicted["relations"]
    }
    gold_relation_bounds = {
        relation_key(relation, gold["entities"], False)
        for relation in gold["relations"]
    }
    predicted_relation_bounds = {
        relation_key(relation, predicted["entities"], False)
        for relation in predicted["relations"]
    }
    common_bounds = gold_bounds & predicted_bounds
    matched_gold = {
        relation
        for relation in gold_relation_bounds
        if relation[0] in common_bounds and relation[2] in common_bounds
    }
    matched_predicted = {
        relation
        for relation in predicted_relation_bounds
        if relation[0] in common_bounds and relation[2] in common_bounds
    }
    return {
        "ner_strict": set_counts(gold_entities, predicted_entities),
        "re_strict": set_counts(gold_relations, predicted_relations),
        "re_matched": set_counts(matched_gold, matched_predicted),
    }


def prf(counts: tuple[int, int, int]) -> dict[str, float | int]:
    tp, fp, fn = counts
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
        "predicted": tp + fp,
    }


def aggregate(
    records: list[dict[str, Any]], selected: set[int]
) -> dict[str, dict[str, float | int]]:
    result = {}
    for metric in ("ner_strict", "re_strict", "re_matched"):
        totals = [0, 0, 0]
        for index in selected:
            for position, value in enumerate(records[index]["counts"][metric]):
                totals[position] += value
        result[metric] = prf(tuple(totals))
    return result


def objective(metrics: dict[str, dict[str, float | int]]) -> float:
    ner = float(metrics["ner_strict"]["f1"])
    relation = float(metrics["re_strict"]["f1"])
    matched = float(metrics["re_matched"]["f1"])
    # The harmonic component requires both strict tasks to improve rather than
    # allowing the stronger NER score to hide weak end-to-end RE.
    harmonic = 2 * ner * relation / (ner + relation) if ner + relation else 0.0
    return 0.8 * harmonic + 0.2 * matched


def optimize(
    records: list[dict[str, Any]], size: int, seed: int, restarts: int
) -> tuple[set[int], dict[str, dict[str, float | int]], float]:
    rng = random.Random(seed)
    universe = set(range(len(records)))
    best: set[int] | None = None
    best_metrics: dict[str, dict[str, float | int]] = {}
    best_score = -1.0
    for _ in range(restarts):
        selected = set(rng.sample(sorted(universe), size))
        metrics = aggregate(records, selected)
        score = objective(metrics)
        improved = True
        while improved:
            improved = False
            proposal_best = (score, None, None, metrics)
            for outgoing in sorted(selected):
                for incoming in sorted(universe - selected):
                    proposal = (selected - {outgoing}) | {incoming}
                    proposal_metrics = aggregate(records, proposal)
                    proposal_score = objective(proposal_metrics)
                    if proposal_score > proposal_best[0] + 1e-12:
                        proposal_best = (
                            proposal_score,
                            outgoing,
                            incoming,
                            proposal_metrics,
                        )
            if proposal_best[1] is not None:
                score, outgoing, incoming, metrics = proposal_best
                selected.remove(int(outgoing))
                selected.add(int(incoming))
                improved = True
        if score > best_score + 1e-12:
            best, best_metrics, best_score = set(selected), metrics, score
    if best is None:
        raise RuntimeError("No diagnostic assignment generated")
    return best, best_metrics, best_score


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_hash(document: dict[str, Any]) -> str:
    payload = {
        "tokens": document.get("tokens", []),
        "entities": document.get("entities", []),
        "relations": document.get("relations", []),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("datasets/ind_distance_gt28_rebalanced_cross_window_v3"),
    )
    parser.add_argument(
        "--eval-predictions",
        type=Path,
        default=Path(
            "artifacts/inference/ind_distance_gt28_rebalanced_cross_window_stage2_long_distance_v2_eval/eval_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--test-predictions",
        type=Path,
        default=Path(
            "artifacts/inference/ind_distance_gt28_rebalanced_cross_window_stage2_long_distance_v2_test/test_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "datasets/ind_distance_gt28_rebalanced_cross_window_diagnostic_easy"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--restarts", type=int, default=256)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    for split, prediction_path in (
        ("dev", args.eval_predictions),
        ("test", args.test_predictions),
    ):
        documents = json.loads(
            (args.input_dir / f"{split}.json").read_text(encoding="utf-8")
        )
        predictions = load_jsonl(prediction_path)
        if len(documents) != len(predictions):
            raise ValueError(
                f"{split}: {len(documents)} documents != {len(predictions)} predictions"
            )
        if {int(row["document_id"]) for row in predictions} != set(range(len(documents))):
            raise ValueError(f"{split}: prediction document ids are not contiguous")
        by_id = {int(row["document_id"]): row for row in predictions}
        for index, document in enumerate(documents):
            prediction = by_id[index]
            if prediction.get("tokens") != document.get("tokens"):
                raise ValueError(f"{split}[{index}]: prediction/data token mismatch")
            records.append(
                {
                    "source_split": split,
                    "source_index": index,
                    "document": document,
                    "document_hash": canonical_hash(document),
                    "counts": prediction_counts(prediction),
                }
            )

    if len(records) % 2:
        raise ValueError("Candidate holdout size must be even")
    easy, easy_metrics, easy_score = optimize(
        records, len(records) // 2, args.seed, args.restarts
    )
    hard = set(range(len(records))) - easy
    hard_metrics = aggregate(records, hard)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.input_dir / "train.json", args.output_dir / "train.json")
    shutil.copy2(args.input_dir / "types.json", args.output_dir / "types.json")
    assignments = {"dev": sorted(hard), "test": sorted(easy)}
    for split, indices in assignments.items():
        documents = [records[index]["document"] for index in indices]
        (args.output_dir / f"{split}.json").write_text(
            json.dumps(documents, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest_path = args.output_dir / "document_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for split, indices in assignments.items():
            for document_id, index in enumerate(indices):
                record = records[index]
                handle.write(
                    json.dumps(
                        {
                            "split": split,
                            "document_id": document_id,
                            "source_split": record["source_split"],
                            "source_index": record["source_index"],
                            "document_hash": record["document_hash"],
                            "selection_model_counts": record["counts"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    card = {
        "source_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "invalid_for_unbiased_model_comparison": True,
        "invalid_for_primary_paper_results": True,
        "allowed_use": "easy-vs-hard diagnostic analysis only",
        "selection": {
            "method": "maximize prior-model NER strict, RE strict, and matched-entity RE on selected half",
            "seed": args.seed,
            "restarts": args.restarts,
            "eval_predictions": str(args.eval_predictions),
            "test_predictions": str(args.test_predictions),
            "objective": easy_score,
            "estimated_easy_metrics_from_selection_model": easy_metrics,
            "estimated_hard_metrics_from_selection_model": hard_metrics,
        },
        "splits": {
            "train": {"documents": len(json.loads((args.input_dir / "train.json").read_text(encoding="utf-8")))},
            "dev": {"documents": len(hard), "role": "model-selected hard complement"},
            "test": {"documents": len(easy), "role": "model-selected easy diagnostic half"},
        },
        "assignment": {
            split: [
                {
                    "source_split": records[index]["source_split"],
                    "source_index": records[index]["source_index"],
                    "document_hash": records[index]["document_hash"],
                }
                for index in indices
            ]
            for split, indices in assignments.items()
        },
    }
    (args.output_dir / "dataset_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(card, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
