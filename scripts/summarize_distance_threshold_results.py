"""Summarize the four frozen-model distance-threshold test runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


THRESHOLDS = (24, 28, 32, 36)


def f1(metrics: dict[str, Any], section: str, key: str = "f1") -> float | None:
    value = metrics.get("overall", {}).get(section)
    return None if value is None else float(value.get(key, 0.0))


def score(metrics: dict[str, Any], bucket: str, section: str) -> float | None:
    value = metrics.get("length_buckets", {}).get(bucket, {}).get(section)
    return None if value is None else float(value.get("f1", 0.0))


def summarize(
    threshold: int,
    split: str,
    dataset_root: Path,
    output_root: Path,
    dataset_prefix: str,
    output_prefix: str,
) -> dict[str, Any]:
    card_path = dataset_root / f"{dataset_prefix}{threshold}" / "dataset_card.json"
    metrics_path = output_root / f"{output_prefix}{threshold}_best_{split}" / f"{split}_metrics.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    # The inference CLI calls the validation split ``eval`` while raw dataset
    # cards and JSON files call it ``dev``.  Metrics remain named eval_metrics.
    card_split = "dev" if split == "eval" else split
    split_card = card["splits"][card_split]
    overall = metrics["overall"]
    return {
        "threshold": threshold,
        "split": split,
        "card_split": card_split,
        "documents": int(split_card["documents"]),
        "tokens": int(split_card["token_count"]),
        "entities": int(split_card["entities"]),
        "relations": int(split_card["relations"]),
        "ner_strict_f1": f1(metrics, "ner_strict"),
        "ner_boundary_f1": f1(metrics, "ner_boundary"),
        "re_strict_f1": f1(metrics, "re_strict_end_to_end"),
        "re_boundary_f1": f1(metrics, "re_boundary_end_to_end"),
        "re_matched_entities_f1": f1(metrics, "re_on_matched_entities"),
        "re_strict_256_511_f1": score(metrics, "256-511", "re_strict_end_to_end"),
        "re_strict_128_255_f1": score(metrics, "128-255", "re_strict_end_to_end"),
        "re_strict_ge128_f1": metrics.get("relation_distance_buckets", {})
        .get(">=128", {})
        .get("f1"),
        "metrics_path": str(metrics_path),
        "ner_support": int(overall.get("ner_strict", {}).get("support", 0)),
        "re_support": int(overall.get("re_strict_end_to_end", {}).get("support", 0)),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/inference"))
    parser.add_argument("--split", choices=("eval", "test"), default="test")
    parser.add_argument("--dataset-prefix", default="long_context_distance_gt")
    parser.add_argument("--output-prefix", default="long_context_distance_gt")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/analysis/long_context_distance_threshold_results.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/long_context_distance_threshold_results_zh.md"),
    )
    parser.add_argument("--thresholds", type=int, nargs="+", default=list(THRESHOLDS))
    args = parser.parse_args()

    rows = [
        summarize(
            threshold,
            args.split,
            args.dataset_root,
            args.output_root,
            args.dataset_prefix,
            args.output_prefix,
        )
        for threshold in sorted(set(args.thresholds))
    ]
    payload = {
        "model_protocol": {
            "description": "Frozen TypeAux + NMS + Stage 2 learned fusion checkpoint",
            "ner_span_threshold": 0.5,
            "ner_type_threshold": 0.3,
            "ner_overlap_strategy": "nms",
            "relation_threshold": 0.2,
            "calibration_split": "original long_context eval",
            "reported_split": args.split,
        },
        "results": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# 关系距离阈值拐点实验结果（{args.split}）",
        "",
        "所有阈值使用同一个冻结的 `TypeAux + NMS + Stage 2 learned fusion` checkpoint，",
        "NER/RE 阈值固定为原始 long_context eval 校准结果：span=0.5、type=0.3、relation=0.2。",
        "因此不同表行之间只改变 test 句子筛选阈值，不重新训练或在 test 上调参。",
        "",
        f"| 最大关系距离 | {args.split} 句数 | 实体数 | 关系数 | NER strict F1 | NER boundary F1 | RE strict F1 | RE boundary F1 | RE matched F1 | RE 256-511 F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                fmt(row[key])
                for key in (
                    "threshold",
                    "documents",
                    "entities",
                    "relations",
                    "ner_strict_f1",
                    "ner_boundary_f1",
                    "re_strict_f1",
                    "re_boundary_f1",
                    "re_matched_entities_f1",
                    "re_strict_256_511_f1",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- 这里的阈值是句内 gold 关系实体中心距离，不是句子 token 长度阈值。该实验衡量的是关系距离带来的难度变化。",
            "- `RE strict F1` 是端到端严格关系 F1，包含关系类型和两个实体边界/类型；`RE matched F1` 只在两个实体边界都匹配时评估关系。",
            "- 选择拐点时同时看 test 句数和指标曲线：优先选择样本量较大、且 NER/RE 开始稳定下降的最小阈值。",
            f"- {args.split} 子集的句子数较少，单个阈值的 F1 方差仍然较大，应结合另一 split 和 bootstrap 置信区间确认。",
            "",
            "## 输出",
            "",
            "- 原始 JSON：`artifacts/analysis/long_context_distance_threshold_results.json`",
            "- 本报告：`docs/long_context_distance_threshold_results_zh.md`",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
