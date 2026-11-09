#!/usr/bin/env python3
"""Train the recommended first-stage long-context NER/RE model.

This is a focused entry point around the existing ITER training loop. It adds
raw-data preflight checks and selects the tokenizer-aware windowing config.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

if "--offline" in sys.argv:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

from iter.datasets import CoNLL
from iter.datasets.training import Hparams
from train import do_train


PROJECT_ROOT = Path(__file__).resolve().parent


def data_preflight(dataset: CoNLL) -> dict:
    """Inspect raw documents before expensive model initialization."""
    report = {"config": dataset.file_path, "data_dir": dataset.data_dir, "splits": {}}
    dataset_card = Path("datasets") / dataset.data_dir / "dataset_card.json"
    if dataset_card.exists():
        report["dataset_card"] = json.loads(dataset_card.read_text(encoding="utf-8"))
    for split in ("train", "eval", "test"):
        raw = dataset.open_split(split, dataset.splits[split])
        lengths = sorted(len(item["tokens"]) for item in raw)
        relation_distances = []
        entity_type_counts = {}
        relation_type_counts = {}
        for item in raw:
            entities = item.get("entities", [])
            for entity in entities:
                entity_type_counts[entity["type"]] = entity_type_counts.get(entity["type"], 0) + 1
            for relation in item.get("relations", []):
                head, tail = entities[relation["head"]], entities[relation["tail"]]
                relation_distances.append(abs(head["start"] - tail["start"]))
                relation_type_counts[relation["type"]] = relation_type_counts.get(relation["type"], 0) + 1
        report["splits"][split] = {
            "documents": len(raw),
            "entities": sum(len(item.get("entities", [])) for item in raw),
            "relations": sum(len(item.get("relations", [])) for item in raw),
            "max_words": max(lengths, default=0),
            "documents_ge_128_words": sum(length >= 128 for length in lengths),
            "documents_ge_256_words": sum(length >= 256 for length in lengths),
            "relations_ge_128_words_apart": sum(distance >= 128 for distance in relation_distances),
            "max_entity_span_words": max(
                (entity["end"] - entity["start"] for item in raw for entity in item.get("entities", [])),
                default=0,
            ),
            "entity_type_counts": dict(sorted(entity_type_counts.items())),
            "relation_type_counts": dict(sorted(relation_type_counts.items())),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer", default="google-bert/bert-base-chinese")
    parser.add_argument("--config", default="wenyanwen/wenyanwen_long_context_512")
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--load_ckpt", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=0, help="0 uses the config value")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size")
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=None,
        help="Override gradient accumulation steps",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bfloat16", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Run data preflight without model initialization")
    parser.add_argument(
        "--skip_final_test",
        action="store_true",
        help="Keep test sealed during model development; write eval history only",
    )
    parser.add_argument(
        "--save_optimizer_state",
        action="store_true",
        help="Also write repro.pt with optimizer state; uses substantially more disk space",
    )
    args = parser.parse_args()

    dataset = CoNLL.from_name(args.config, reinitialize_dataset=args.rebuild_cache)
    hparams = Hparams.from_name(args.config)
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise SystemExit("--batch_size must be at least 1")
        hparams.batch_size = args.batch_size
    if args.gradient_accumulation is not None:
        if args.gradient_accumulation < 1:
            raise SystemExit("--gradient_accumulation must be at least 1")
        hparams.gradient_accumulation = args.gradient_accumulation
    if args.patience is not None:
        if args.patience < 1:
            raise SystemExit("--patience must be at least 1")
        hparams.patience = args.patience
    preflight = data_preflight(dataset)
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    if args.model_path:
        args.model_path.mkdir(parents=True, exist_ok=True)
        with (args.model_path / "data_preflight.json").open("w", encoding="utf-8") as handle:
            json.dump(preflight, handle, ensure_ascii=False, indent=2)
        run_config = {
            "transformer": args.transformer,
            "config": args.config,
            "model_path": str(args.model_path.resolve()),
            "load_ckpt": str(args.load_ckpt.resolve()) if args.load_ckpt else None,
            "epochs": args.epochs,
            "patience": hparams.patience,
            "batch_size": hparams.batch_size,
            "gradient_accumulation": hparams.gradient_accumulation,
            "seed": args.seed,
            "bfloat16": args.bfloat16,
            "offline": args.offline,
            "skip_final_test": args.skip_final_test,
            "python": sys.version,
            "platform": platform.platform(),
        }
        with (args.model_path / "run_config.json").open("w", encoding="utf-8") as handle:
            json.dump(run_config, handle, ensure_ascii=False, indent=2)
    if args.dry_run:
        return

    metrics = do_train(
        transformer=args.transformer,
        datasets=[dataset],
        hparams=hparams,
        model_path=str(args.model_path) if args.model_path else None,
        log_file=None,
        log_append=False,
        seed=args.seed,
        use_tqdm=not args.no_tqdm,
        use_bfloat16=args.bfloat16,
        use_fsdp=False,
        num_epochs=args.epochs,
        verbose=True,
        dont_ckpt=False,
        do_compile=False,
        show_bound_metrics=True,
        skip_final_test=args.skip_final_test,
        load_ckpt=args.load_ckpt,
        save_best_only=not args.save_optimizer_state,
    )
    if metrics is not None:
        print("Final test metrics:")
        print(json.dumps(metrics.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
