"""Build long-context subsets selected by the maximum relation distance.

Each raw JSON object is one document/sentence.  A document is retained when
the maximum center-to-center distance among its gold relations is strictly
greater than the requested threshold.  Entity and relation indices are kept
unchanged because the complete document is copied.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


SPLITS = ("train", "dev", "test")
DEFAULT_THRESHOLDS = (24, 28, 32, 36)


def entity_center(entity: dict[str, Any]) -> int:
    return (int(entity["start"]) + int(entity["end"])) // 2


def relation_distance(entities: list[dict[str, Any]], relation: dict[str, Any]) -> int:
    head = entities[int(relation["head"])]
    tail = entities[int(relation["tail"])]
    return abs(entity_center(head) - entity_center(tail))


def max_relation_distance(document: dict[str, Any]) -> int | None:
    entities = document.get("entities", [])
    relations = document.get("relations", [])
    distances = []
    for relation in relations:
        head = int(relation["head"])
        tail = int(relation["tail"])
        if not (0 <= head < len(entities) and 0 <= tail < len(entities)):
            raise ValueError(f"Invalid relation endpoint: {relation}")
        distances.append(relation_distance(entities, relation))
    return max(distances) if distances else None


def load_documents(source_dir: Path, split: str) -> list[dict[str, Any]]:
    return json.loads((source_dir / f"{split}.json").read_text(encoding="utf-8"))


def build_dataset(
    source_dir: Path,
    output_root: Path,
    threshold: int,
    dataset_prefix: str,
) -> dict[str, Any]:
    dataset_dir = output_root / f"{dataset_prefix}{threshold}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "types.json", dataset_dir / "types.json")

    split_stats: dict[str, Any] = {}
    selected_ids: dict[str, list[int]] = {}
    for split in SPLITS:
        documents = load_documents(source_dir, split)
        distances = [max_relation_distance(document) for document in documents]
        selected = [
            document
            for document, distance in zip(documents, distances)
            if distance is not None and distance > threshold
        ]
        ids = [index for index, distance in enumerate(distances) if distance is not None and distance > threshold]
        (dataset_dir / f"{split}.json").write_text(
            json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        selected_ids[split] = ids
        split_stats[split] = {
            "source_documents": len(documents),
            "documents": len(selected),
            "selected_fraction": len(selected) / len(documents) if documents else 0.0,
            "token_count": sum(len(document.get("tokens", [])) for document in selected),
            "entities": sum(len(document.get("entities", [])) for document in selected),
            "relations": sum(len(document.get("relations", [])) for document in selected),
            "max_relation_distance": max(
                (distance for distance in distances if distance is not None and distance > threshold),
                default=None,
            ),
            "source_document_indices": ids,
        }

    card = {
        "source_dir": str(source_dir),
        "output_dir": str(dataset_dir),
        "selection": {
            "rule": "retain documents whose maximum gold relation center distance is strictly greater than threshold",
            "threshold": threshold,
            "distance_definition": "abs((head.start + head.end)//2 - (tail.start + tail.end)//2)",
        },
        "splits": split_stats,
    }
    (dataset_dir / "dataset_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return card


def write_config(
    config_root: Path,
    threshold: int,
    dataset_dir_name: str,
    config_prefix: str,
    dataset_name_prefix: str,
) -> Path:
    config_path = config_root / f"{config_prefix}{threshold}.json"
    config = {
        "base": "wenyanwen/wenyanwen_long_context_512_type_aux",
        "dataset": {
            "name": f"{dataset_name_prefix}{threshold}",
            "data_dir": dataset_dir_name,
            "reinitialize_dataset": False,
        },
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("datasets/long_context"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets"))
    parser.add_argument("--config-root", type=Path, default=Path("cfg/wenyanwen"))
    parser.add_argument(
        "--dataset-prefix",
        default="long_context_distance_gt",
        help="Output dataset directory prefix; threshold is appended, e.g. ind_distance_gt24",
    )
    parser.add_argument(
        "--config-prefix",
        default="wenyanwen_long_context_512_type_aux_distance_gt",
        help="Output config filename prefix; threshold is appended before .json",
    )
    parser.add_argument(
        "--dataset-name-prefix",
        default="wenyanwen_long_context_type_aux_distance_gt",
        help="Dataset name prefix stored in each generated config",
    )
    parser.add_argument("--thresholds", type=int, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("artifacts/analysis/long_context_distance_threshold_datasets.json"),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.config_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"source_dir": str(args.source_dir), "thresholds": {}}
    for threshold in sorted(set(args.thresholds)):
        card = build_dataset(args.source_dir, args.output_root, threshold, args.dataset_prefix)
        dataset_name = f"{args.dataset_prefix}{threshold}"
        config_path = write_config(
            args.config_root,
            threshold,
            dataset_name,
            args.config_prefix,
            args.dataset_name_prefix,
        )
        manifest["thresholds"][str(threshold)] = {
            "dataset_dir": card["output_dir"],
            "config": str(config_path),
            "splits": card["splits"],
        }
        print(
            f">{threshold}: "
            + ", ".join(
                f"{split}={card['splits'][split]['documents']}"
                for split in SPLITS
            )
        )

    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.manifest_output}")


if __name__ == "__main__":
    main()
