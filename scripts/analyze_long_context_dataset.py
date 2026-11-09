"""Analyze the long-context corpus at the raw sentence/document level.

The raw JSON objects are treated as one sentence/document each.  Relation
distance follows the project evaluator convention: the absolute difference
between the integer token-center positions of the two endpoint entities.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_THRESHOLDS = (28, 32, 36, 64, 80, 96, 128)
SPLITS = ("train", "dev", "test")


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p90": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
        "p90": _percentile(values, 90),
    }


def _rate(count: int, denominator: int) -> float:
    return count / denominator if denominator else 0.0


def _entity_center(entity: dict[str, Any]) -> int:
    # Raw spans use [start, end); this matches iter/misc/document_metrics.py.
    return (int(entity["start"]) + int(entity["end"])) // 2


def _distance(head: dict[str, Any], tail: dict[str, Any]) -> int:
    return abs(_entity_center(head) - _entity_center(tail))


def _category_rows(
    names: list[str],
    mention_counts: Counter[str],
    sentence_ids: dict[str, set[int]],
    sentence_count: int,
) -> list[dict[str, Any]]:
    rows = []
    for name in names:
        mentions = int(mention_counts.get(name, 0))
        sentences = len(sentence_ids.get(name, set()))
        rows.append(
            {
                "type": name,
                "mention_count": mentions,
                "mention_share": _rate(mentions, sum(mention_counts.values())),
                "sentence_count": sentences,
                "sentence_rate": _rate(sentences, sentence_count),
                "mean_mentions_per_present_sentence": _rate(mentions, sentences),
            }
        )
    return rows


def _analyze_threshold_subset(
    documents: list[dict[str, Any]],
    selected_ids: set[int],
    entity_names: list[str],
    relation_names: list[str],
) -> dict[str, Any]:
    """Summarize labels inside sentences selected by a max-distance threshold."""
    entity_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    entity_sentence_ids: dict[str, set[int]] = defaultdict(set)
    relation_sentence_ids: dict[str, set[int]] = defaultdict(set)
    token_lengths: list[int] = []
    entity_lengths: list[int] = []
    relation_lengths: list[int] = []

    for sentence_id in sorted(selected_ids):
        document = documents[sentence_id]
        tokens = document.get("tokens", [])
        entities = document.get("entities", [])
        relations = document.get("relations", [])
        token_lengths.append(len(tokens))
        entity_lengths.append(len(entities))
        relation_lengths.append(len(relations))

        valid = [_validate_entity(entity, len(tokens)) for entity in entities]
        for entity_index, entity in enumerate(entities):
            if not valid[entity_index]:
                continue
            entity_type = str(entity["type"])
            entity_counts[entity_type] += 1
            entity_sentence_ids[entity_type].add(sentence_id)
        for relation in relations:
            head_index = int(relation["head"])
            tail_index = int(relation["tail"])
            if (
                not (0 <= head_index < len(entities))
                or not (0 <= tail_index < len(entities))
                or not valid[head_index]
                or not valid[tail_index]
            ):
                continue
            relation_type = str(relation["type"])
            relation_counts[relation_type] += 1
            relation_sentence_ids[relation_type].add(sentence_id)

    sentence_count = len(selected_ids)
    return {
        "sentence_count": sentence_count,
        "sentence_rate": _rate(sentence_count, len(documents)),
        "token_total": sum(token_lengths),
        "token_count": _stats(token_lengths),
        "entity_count": sum(entity_counts.values()),
        "relation_count": sum(relation_counts.values()),
        "entity_count_per_sentence": _stats(entity_lengths),
        "relation_count_per_sentence": _stats(relation_lengths),
        "entity_type_distribution": _category_rows(
            entity_names, entity_counts, entity_sentence_ids, sentence_count
        ),
        "relation_type_distribution": _category_rows(
            relation_names, relation_counts, relation_sentence_ids, sentence_count
        ),
    }


def _validate_entity(entity: dict[str, Any], token_count: int) -> bool:
    start = int(entity["start"])
    end = int(entity["end"])
    return 0 <= start < end <= token_count


def _analyze_split(
    split: str,
    documents: list[dict[str, Any]],
    entity_names: list[str],
    relation_names: list[str],
    thresholds: tuple[int, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entity_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    entity_sentence_ids: dict[str, set[int]] = defaultdict(set)
    relation_sentence_ids: dict[str, set[int]] = defaultdict(set)
    token_lengths: list[int] = []
    entity_lengths: list[int] = []
    relation_lengths: list[int] = []
    relation_distances: list[int] = []
    max_distances: list[int] = []
    sentence_rows: list[dict[str, Any]] = []
    invalid_entities = 0
    invalid_relations = 0

    for sentence_id, document in enumerate(documents):
        tokens = document.get("tokens", [])
        entities = document.get("entities", [])
        relations = document.get("relations", [])
        token_count = len(tokens)
        token_lengths.append(token_count)
        entity_lengths.append(len(entities))
        relation_lengths.append(len(relations))

        valid_entities: list[dict[str, Any]] = []
        for entity in entities:
            if not _validate_entity(entity, token_count):
                invalid_entities += 1
                continue
            valid_entities.append(entity)
            entity_type = str(entity["type"])
            entity_counts[entity_type] += 1
            entity_sentence_ids[entity_type].add(sentence_id)

        sentence_distances: list[int] = []
        longest: dict[str, Any] | None = None
        for relation in relations:
            head_index = int(relation["head"])
            tail_index = int(relation["tail"])
            if not (0 <= head_index < len(valid_entities)) or not (0 <= tail_index < len(valid_entities)):
                invalid_relations += 1
                continue
            relation_type = str(relation["type"])
            relation_counts[relation_type] += 1
            relation_sentence_ids[relation_type].add(sentence_id)
            distance = _distance(valid_entities[head_index], valid_entities[tail_index])
            sentence_distances.append(distance)
            relation_distances.append(distance)
            if longest is None or distance > int(longest["distance"]):
                longest = {
                    "distance": distance,
                    "relation_type": relation_type,
                    "head_entity_index": head_index,
                    "tail_entity_index": tail_index,
                    "head_entity_type": str(valid_entities[head_index]["type"]),
                    "tail_entity_type": str(valid_entities[tail_index]["type"]),
                }

        max_distance = max(sentence_distances) if sentence_distances else None
        if max_distance is not None:
            max_distances.append(max_distance)
        row = {
            "split": split,
            "sentence_id": sentence_id,
            "token_count": token_count,
            "entity_count": len(valid_entities),
            "relation_count": len(sentence_distances),
            "max_relation_distance": max_distance,
        }
        if longest is not None:
            row.update(
                {
                    "max_relation_type": longest["relation_type"],
                    "max_relation_head_entity_index": longest["head_entity_index"],
                    "max_relation_tail_entity_index": longest["tail_entity_index"],
                    "max_relation_head_entity_type": longest["head_entity_type"],
                    "max_relation_tail_entity_type": longest["tail_entity_type"],
                }
            )
        sentence_rows.append(row)

    threshold_counts = {
        f">{threshold}": sum(distance > threshold for distance in max_distances)
        for threshold in thresholds
    }
    threshold_rates = {
        key: _rate(count, len(documents)) for key, count in threshold_counts.items()
    }
    bucket_counts = {
        "no_relation": sum(row["max_relation_distance"] is None for row in sentence_rows),
        "0-64": sum(row["max_relation_distance"] is not None and row["max_relation_distance"] <= 64 for row in sentence_rows),
        "65-80": sum(row["max_relation_distance"] is not None and 64 < row["max_relation_distance"] <= 80 for row in sentence_rows),
        "81-96": sum(row["max_relation_distance"] is not None and 80 < row["max_relation_distance"] <= 96 for row in sentence_rows),
        "97-128": sum(row["max_relation_distance"] is not None and 96 < row["max_relation_distance"] <= 128 for row in sentence_rows),
        ">128": sum(row["max_relation_distance"] is not None and row["max_relation_distance"] > 128 for row in sentence_rows),
    }
    threshold_subsets = {}
    for threshold in thresholds:
        selected_ids = {
            int(row["sentence_id"])
            for row in sentence_rows
            if row["max_relation_distance"] is not None
            and int(row["max_relation_distance"]) > threshold
        }
        threshold_subsets[f">{threshold}"] = _analyze_threshold_subset(
            documents,
            selected_ids,
            entity_names,
            relation_names,
        )

    summary = {
        "split": split,
        "sentence_count": len(documents),
        "token_count": _stats(token_lengths),
        "entity_count": sum(entity_counts.values()),
        "relation_count": sum(relation_counts.values()),
        "entity_count_per_sentence": _stats(entity_lengths),
        "relation_count_per_sentence": _stats(relation_lengths),
        "sentences_with_entities": sum(bool(document.get("entities")) for document in documents),
        "sentences_with_relations": sum(bool(document.get("relations")) for document in documents),
        "sentences_without_relations": sum(not bool(document.get("relations")) for document in documents),
        "entity_type_distribution": _category_rows(
            entity_names, entity_counts, entity_sentence_ids, len(documents)
        ),
        "relation_type_distribution": _category_rows(
            relation_names, relation_counts, relation_sentence_ids, len(documents)
        ),
        "relation_distance": {
            "definition": "abs((head.start + head.end)//2 - (tail.start + tail.end)//2)",
            "sentences_with_relations": len(max_distances),
            "max_distance_per_sentence": _stats(max_distances),
            "threshold_sentence_counts": threshold_counts,
            "threshold_sentence_rates": threshold_rates,
            "max_distance_buckets": bucket_counts,
            "all_relation_pair_distances": _stats(relation_distances),
            "threshold_subsets": threshold_subsets,
        },
        "validation": {
            "invalid_entities": invalid_entities,
            "invalid_relations": invalid_relations,
        },
    }
    return summary, sentence_rows


def _fmt_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return f"{value:,}"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# 长上下文数据集统计（按句子/文档）",
        "",
        "## 统计口径",
        "",
        "- 每个 JSON 对象视为一条原始长句/document；不是滑动窗口，也不是单个 BERT 子词窗口。",
        "- `tokens` 数量为数据文件中字符级 token 数量。当前数据集筛选标准为 `tokens >= 128`（构建阶段字段名为 words）。",
        "- 关系距离采用项目评估器口径：`abs((head.start + head.end)//2 - (tail.start + tail.end)//2)`，即两个实体 token 中心位置的绝对距离。",
        "- 类型统计同时给出实体/关系标注次数和包含该类型的句子数，避免把 mention 分布误解为句子覆盖率。",
        "",
        "## Split 概览",
        "",
        "| Split | 句子数 | 总 token | 平均 token | 中位 token | 最短 | 最长 | 实体数 | 关系数 | 含关系句数 | 无关系句数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        item = result["splits"][split]
        token_stats = item["token_count"]
        lines.append(
            f"| {split} | {item['sentence_count']:,} | {sum(result['sentence_rows'][split][i]['token_count'] for i in range(item['sentence_count'])):,} | "
            f"{_fmt_number(token_stats['mean'])} | {_fmt_number(token_stats['median'])} | {_fmt_number(token_stats['min'])} | {_fmt_number(token_stats['max'])} | "
            f"{item['entity_count']:,} | {item['relation_count']:,} | {item['sentences_with_relations']:,} | {item['sentences_without_relations']:,} |"
        )

    thresholds = result["thresholds"]
    threshold_columns = " | ".join(f">{threshold}" for threshold in thresholds)
    threshold_alignment = "|".join("---:" for _ in thresholds)
    lines.extend(
        [
            "",
            "## 每句最大关系距离",
            "",
            "这里先在每条句子内部找到所有关系实体对的距离，再取该句最大值。无关系句的最大距离记为 `N/A`，不会计入任何 `>阈值`。",
            "",
            f"| Split | 有关系句数 | 无关系句数 | 最大距离均值 | 中位数 | P90 | 最大值 | {threshold_columns} |",
            f"|---|---:|---:|---:|---:|---:|---:|{threshold_alignment}|",
        ]
    )
    for split in SPLITS:
        distance = result["splits"][split]["relation_distance"]
        stats = distance["max_distance_per_sentence"]
        counts = distance["threshold_sentence_counts"]
        lines.append(
            f"| {split} | {distance['sentences_with_relations']:,} | {result['splits'][split]['sentences_without_relations']:,} | "
            f"{_fmt_number(stats['mean'])} | {_fmt_number(stats['median'])} | {_fmt_number(stats['p90'])} | {_fmt_number(stats['max'])} | "
            + " | ".join(f"{counts[f'>{threshold}']:,}" for threshold in thresholds)
            + " |"
        )
    lines.extend(
        [
            "",
            "### 最大距离区间（按句子）",
            "",
            "| Split | 无关系 | 0-64 | 65-80 | 81-96 | 97-128 | >128 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in SPLITS:
        buckets = result["splits"][split]["relation_distance"]["max_distance_buckets"]
        lines.append(
            f"| {split} | {buckets['no_relation']:,} | {buckets['0-64']:,} | {buckets['65-80']:,} | "
            f"{buckets['81-96']:,} | {buckets['97-128']:,} | {buckets['>128']:,} |"
        )

    # Compare category distributions inside the increasingly long-distance
    # sentence subsets.  Cells show mention count / sentence coverage count.
    for split in SPLITS:
        subsets = result["splits"][split]["relation_distance"]["threshold_subsets"]
        lines.extend(
            [
                "",
                f"## {split}：最大关系距离阈值子集",
                "",
                "每个阈值只保留句内最大关系距离大于该阈值的句子；下表中的实体/关系分布均在该子集内重新统计。",
                "",
                "| 阈值 | 句子数 | 占 split | token 总数 | 实体数 | 关系数 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for threshold in thresholds:
            subset = subsets[f">{threshold}"]
            lines.append(
                f"| >{threshold} | {subset['sentence_count']:,} | {_fmt_pct(subset['sentence_rate'])} | "
                f"{subset['token_total']:,} | {subset['entity_count']:,} | {subset['relation_count']:,} |"
            )

        lines.extend(
            [
                "",
                "### 实体类型（mention 数 / 覆盖句数）",
                "",
                "| 实体类型 | " + " | ".join(f">{threshold}" for threshold in thresholds) + " |",
                "|---|" + "|".join("---:" for _ in thresholds) + "|",
            ]
        )
        entity_rows = {
            row["type"]: row
            for row in subsets[f">{thresholds[0]}"]["entity_type_distribution"]
        }
        for entity_type in entity_rows:
            cells = []
            for threshold in thresholds:
                row = next(
                    item
                    for item in subsets[f">{threshold}"]["entity_type_distribution"]
                    if item["type"] == entity_type
                )
                cells.append(f"{row['mention_count']:,} / {row['sentence_count']:,}")
            lines.append(f"| {entity_type} | " + " | ".join(cells) + " |")

        lines.extend(
            [
                "",
                "### 关系类别（mention 数 / 覆盖句数）",
                "",
                "| 关系类别 | " + " | ".join(f">{threshold}" for threshold in thresholds) + " |",
                "|---|" + "|".join("---:" for _ in thresholds) + "|",
            ]
        )
        relation_rows = {
            row["type"]: row
            for row in subsets[f">{thresholds[0]}"]["relation_type_distribution"]
        }
        for relation_type in relation_rows:
            cells = []
            for threshold in thresholds:
                row = next(
                    item
                    for item in subsets[f">{threshold}"]["relation_type_distribution"]
                    if item["type"] == relation_type
                )
                cells.append(f"{row['mention_count']:,} / {row['sentence_count']:,}")
            lines.append(f"| {relation_type} | " + " | ".join(cells) + " |")

    for split in SPLITS:
        item = result["splits"][split]
        lines.extend(
            [
                "",
                f"## {split} 实体类型分布",
                "",
                "`mention_count` 是实体标注总数；`sentence_count` 是至少包含一个该类型实体的句子数。",
                "",
                "| 实体类型 | mention 数 | mention 占比 | 句子数 | 句子覆盖率 | 出现句平均 mention 数 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in item["entity_type_distribution"]:
            lines.append(
                f"| {row['type']} | {row['mention_count']:,} | {_fmt_pct(row['mention_share'])} | "
                f"{row['sentence_count']:,} | {_fmt_pct(row['sentence_rate'])} | {_fmt_number(row['mean_mentions_per_present_sentence'])} |"
            )
        lines.extend(
            [
                "",
                f"## {split} 关系类别分布",
                "",
                "`mention_count` 是关系标注总数；`sentence_count` 是至少包含一个该关系类别的句子数。",
                "",
                "| 关系类别 | mention 数 | mention 占比 | 句子数 | 句子覆盖率 | 出现句平均 mention 数 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in item["relation_type_distribution"]:
            lines.append(
                f"| {row['type']} | {row['mention_count']:,} | {_fmt_pct(row['mention_share'])} | "
                f"{row['sentence_count']:,} | {_fmt_pct(row['sentence_rate'])} | {_fmt_number(row['mean_mentions_per_present_sentence'])} |"
            )

    lines.extend(
        [
            "",
            "## 复核文件",
            "",
            "- 原始统计 JSON：`artifacts/analysis/long_context_dataset_distribution.json`",
            "- 逐句最大关系距离 CSV：`artifacts/analysis/long_context_sentence_stats.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/long_context"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/analysis/long_context_dataset_distribution.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/long_context_dataset_distribution_zh.md"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/analysis/long_context_sentence_stats.csv"),
    )
    parser.add_argument("--thresholds", type=int, nargs="+", default=list(DEFAULT_THRESHOLDS))
    args = parser.parse_args()
    thresholds = tuple(sorted(set(args.thresholds)))

    types_path = args.dataset_dir / "types.json"
    type_config = json.loads(types_path.read_text(encoding="utf-8")) if types_path.exists() else {}
    entity_names = list(type_config.get("entities", {}).keys())
    relation_names = list(type_config.get("relations", {}).keys())

    result: dict[str, Any] = {
        "dataset_dir": str(args.dataset_dir),
        "distance_definition": "abs((head.start + head.end)//2 - (tail.start + tail.end)//2)",
        "thresholds": list(thresholds),
        "splits": {},
        "sentence_rows": {},
    }
    all_names = {"entities": set(entity_names), "relations": set(relation_names)}
    for split in SPLITS:
        path = args.dataset_dir / f"{split}.json"
        documents = json.loads(path.read_text(encoding="utf-8"))
        # Keep unknown labels visible rather than silently dropping them.
        for document in documents:
            all_names["entities"].update(str(entity["type"]) for entity in document.get("entities", []))
            all_names["relations"].update(str(relation["type"]) for relation in document.get("relations", []))
        result["splits"][split], rows = _analyze_split(
            split,
            documents,
            entity_names,
            relation_names,
            thresholds,
        )
        result["sentence_rows"][split] = rows

    # Re-run only if an unexpected label appeared outside types.json, preserving
    # the configured order for the normal dataset.
    for category, names in all_names.items():
        configured = entity_names if category == "entities" else relation_names
        for name in sorted(names - set(configured)):
            configured.append(name)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(_markdown_report(result), encoding="utf-8")
    fieldnames = [
        "split",
        "sentence_id",
        "token_count",
        "entity_count",
        "relation_count",
        "max_relation_distance",
        "max_relation_type",
        "max_relation_head_entity_index",
        "max_relation_tail_entity_index",
        "max_relation_head_entity_type",
        "max_relation_tail_entity_type",
    ]
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split in SPLITS:
            writer.writerows(result["sentence_rows"][split])

    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_csv}")
    for split in SPLITS:
        item = result["splits"][split]
        distance = item["relation_distance"]
        counts = distance["threshold_sentence_counts"]
        print(
            split,
            f"sentences={item['sentence_count']}",
            f"entities={item['entity_count']}",
            f"relations={item['relation_count']}",
            " ".join(f"{key}={value}" for key, value in counts.items()),
        )


if __name__ == "__main__":
    main()
