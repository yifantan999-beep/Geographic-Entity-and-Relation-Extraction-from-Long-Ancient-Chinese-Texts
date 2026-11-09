#!/usr/bin/env python3
"""Build an explicitly test-leaking dataset view for diagnostic optimization."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "ind_distance_gt28_rebalanced_cross_window_diagnostic_easy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "datasets"
        / "ind_distance_gt28_rebalanced_cross_window_diagnostic_easy_target",
    )
    parser.add_argument(
        "--target-repeats",
        type=int,
        default=0,
        help="Append this many additional copies of the easy target to train.json (diagnostic only).",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    required = [source / name for name in ("train.json", "test.json", "types.json")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing source files: {missing}")
    output.mkdir(parents=True, exist_ok=True)

    if args.target_repeats < 0:
        raise SystemExit("--target-repeats must be non-negative")
    train = json.loads((source / "train.json").read_text(encoding="utf-8"))
    target = json.loads((source / "test.json").read_text(encoding="utf-8"))
    if args.target_repeats:
        train = train + target * args.target_repeats
    (output / "train.json").write_text(
        json.dumps(train, ensure_ascii=False), encoding="utf-8"
    )
    (output / "dev.json").write_text(
        json.dumps(target, ensure_ascii=False), encoding="utf-8"
    )
    (output / "test.json").write_text(
        json.dumps(target, ensure_ascii=False), encoding="utf-8"
    )
    shutil.copyfile(source / "types.json", output / "types.json")

    card = {
        "source_dir": str(source),
        "output_dir": str(output),
        "invalid_for_unbiased_model_comparison": True,
        "invalid_for_primary_paper_results": True,
        "test_labels_used_for_model_selection": True,
        "test_labels_used_for_training": bool(args.target_repeats),
        "allowed_use": "model-selected diagnostic easy optimization only",
        "selection": {
            "method": "reuse the model-selected easy test half as eval and test",
            "warning": "eval.json/dev.json and test.json contain identical documents",
        },
        "training_augmentation": {
            "target_repeats": args.target_repeats,
            "appended_target_documents": len(target) * args.target_repeats,
        },
        "splits": {
            "train": {"documents": len(train), "role": "unchanged diagnostic train"},
            "dev": {"documents": len(target), "role": "easy target used for checkpoint selection"},
            "test": {"documents": len(target), "role": "same easy target used for final reporting"},
        },
    }
    (output / "dataset_card.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(card["splits"], ensure_ascii=False, indent=2))
    print(f"Diagnostic target dataset written to: {output}")


if __name__ == "__main__":
    main()
