#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
assert sys.version_info >= (3, 10), "Python 3.10+ required. Use: python evaluate.py ..."

"""
Evaluate locally trained ITER checkpoints, including their trained task heads.

Usage:
  python evaluate.py --configs wenyanwen_long_context_512 --split test
  python evaluate.py --configs wenyanwen_long_context_512 --split eval
  python evaluate.py --split test

Writes results to artifacts/evaluation/evaluation_results.xlsx / .json.
"""
import os

# ---------------------------------------------------------------------------
# 优化显存分配机制，避免显存碎片化导致的大块内存分配失败
# ---------------------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------------------------------------------------------
# Offline mode must be set before importing transformers/ITER.
# ---------------------------------------------------------------------------
if (
    "--offline" in sys.argv
    or os.environ.get("EVAL_OFFLINE", "").lower() in ("1", "true", "yes")
):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json
import warnings
import contextlib
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `iter` imports work
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from iter import ITER, ITERConfig
from iter.datasets import CoNLL
from iter.datasets.training import Hparams
from iter.misc.metrics import (
    Metrics,
    MetricsDict,
    calculate_f1,
    metrics_ner,
    metrics_ere,
    accumulate_macro_metrics,
    accumulate_micro_metrics,
    remove_entity_tag_from_links,
    remove_entity_tag_from_entities,
    remove_extra_info_from_links,
)
from iter.modeling_iter2 import ITERForRelationExtraction
from iter.generation_utils import decode_actions_and_pairings
from iter.data.data_collator import Batch
from iter.tools.dataloaders import create_primary_dataloader

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_MODELS_BASE_DIR = PROJECT_ROOT / "models"

ALL_CONFIG_NAMES = [
    "wenyanwen/wenyanwen_long_context_512",
]

CONFIG_DISPLAY_NAMES = {
    "wenyanwen/wenyanwen_long_context_512": "wenyanwen_long_context_stage1",
}

DEFAULT_EVAL_SPLIT = "test"
DEFAULT_EVAL_BATCH_SIZE = 2


# ---------------------------------------------------------------------------
# Find the best trained checkpoint
# ---------------------------------------------------------------------------

def find_best_trained_checkpoint(config_name: str, models_base: Path) -> Optional[Path]:
    """
    Search models/{config_short_name}/ for the best trained checkpoint.

    Priority:
      1. Dir with both model.safetensors AND metrics.json → pick highest ner_f1
      2. Dir with model.safetensors only → pick most recently modified
      3. None → return None

    Supports two directory layouts:
      models/{config}/bert-base-chinese/{timestamp}/
      models/{config}/google-bert/bert-base-chinese/{timestamp}/
    """
    short_name = CONFIG_DISPLAY_NAMES.get(config_name, config_name.split("/")[-1])
    config_dir = models_base / short_name

    if not config_dir.exists():
        return None

    # Transformer identifiers can contain "/" (for example
    # google-bert/bert-base-chinese), so checkpoints may be nested more deeply.
    checkpoint_dirs = []
    for config_file in config_dir.rglob("config.json"):
        run_dir = config_file.parent
        has_weights = (
            (run_dir / "model.safetensors").exists()
            or (run_dir / "model.safetensors.index.json").exists()
        )
        if has_weights:
            checkpoint_dirs.append(run_dir)

    if not checkpoint_dirs:
        return None

    # Prefer checkpoint with metrics.json → highest F1
    dirs_with_metrics = [d for d in checkpoint_dirs if (d / "metrics.json").exists()]
    if dirs_with_metrics:
        best_dir = None
        best_f1 = -1.0
        for d in dirs_with_metrics:
            try:
                with open(d / "metrics.json") as f:
                    data = json.load(f)
                metrics = data.get("test_metrics", {})
                f1 = metrics.get("f1", metrics.get("ere_f1", metrics.get("ner_f1", 0.0)))
                if f1 > best_f1:
                    best_f1 = f1
                    best_dir = d
            except Exception:
                pass
        if best_dir:
            return best_dir

    # Fallback: most recent by modification time
    checkpoint_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return checkpoint_dirs[0]


# ---------------------------------------------------------------------------
# F1RE computation (Pipeline Relation Classification)
# ---------------------------------------------------------------------------

def compute_f1re(
    pred_entities_per_sample: list,
    gold_entities_per_sample: list,
    pred_links_per_sample: list,
    gold_links_per_sample: list,
    symmetric_links: set,
) -> tuple:
    """
    Compute F1RE: Pipeline-mode relation classification.
    Given gold entity boundaries, evaluate only relation type correctness.
    """
    all_tp, all_fp, all_fn = 0, 0, 0

    for pred_ents, gold_ents, pred_links, gold_links in zip(
        pred_entities_per_sample,
        gold_entities_per_sample,
        pred_links_per_sample,
        gold_links_per_sample,
    ):
        gold_by_boundary: dict = {}
        for ent in gold_ents:
            boundary = ent[0]
            gold_by_boundary[boundary] = ent

        pred_by_boundary: dict = {}
        for ent in pred_ents:
            boundary = ent[0]
            pred_by_boundary[boundary] = ent

        common_boundaries = set(gold_by_boundary.keys()) & set(pred_by_boundary.keys())

        filtered_pred_links = []
        for link in pred_links:
            head, link_type, tail = link[:3]
            head_b = head[0]
            tail_b = tail[0]
            if head_b in common_boundaries and tail_b in common_boundaries:
                gold_head = gold_by_boundary[head_b]
                gold_tail = gold_by_boundary[tail_b]
                filtered_pred_links.append((gold_head, link_type, gold_tail))

        filtered_gold_links = []
        for link in gold_links:
            head, link_type, tail = link[:3]
            head_b = head[0]
            tail_b = tail[0]
            if head_b in common_boundaries and tail_b in common_boundaries:
                filtered_gold_links.append((head, link_type, tail))

        pred_notag = remove_entity_tag_from_links(filtered_pred_links)
        pred_notag = remove_extra_info_from_links(pred_notag)
        gold_notag = remove_entity_tag_from_links(filtered_gold_links)
        gold_notag = remove_extra_info_from_links(gold_notag)

        if symmetric_links:
            def to_comparable(items):
                result = []
                for h, lt, t in items:
                    if lt in symmetric_links:
                        result.append(frozenset({h, lt, t}))
                    else:
                        result.append((h, lt, t))
                return result

            pred_set = set(to_comparable(pred_notag))
            gold_set = set(to_comparable(gold_notag))
        else:
            pred_set = set(pred_notag)
            gold_set = set(gold_notag)

        all_tp += len(pred_set & gold_set)
        all_fp += len(pred_set - gold_set)
        all_fn += len(gold_set - pred_set)

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


# ---------------------------------------------------------------------------
# Full evaluation with raw prediction collection
# ---------------------------------------------------------------------------

def evaluate_model_with_raw(
    model: ITER,
    dataset: CoNLL,
    hparams: Hparams,
    split: str = "test",
) -> tuple:
    """Run end-to-end evaluation: NER + pipeline RE + strict RE."""
    device = model.device
    primary_dataset = dataset
    model.eval()

    all_pred_entities: list = []
    all_gold_entities: list = []
    all_pred_links: list = []
    all_gold_links: list = []

    ner_metrics_acc = accumulate_macro_metrics([], primary_dataset.num_real_types)
    ner_metrics_no_tag_acc = accumulate_macro_metrics([], 0)
    ere_metrics_acc = accumulate_macro_metrics([], primary_dataset.num_real_links)
    ere_metrics_no_tag_acc = accumulate_macro_metrics([], primary_dataset.num_real_links)

    total_samples = 0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device_type = "cuda" if "cuda" in str(device) else "cpu"
    autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.float16) if device_type == "cuda" else contextlib.nullcontext()

    with torch.no_grad(), autocast_ctx:
        dataloader = create_primary_dataloader(
            [dataset],
            split,
            False,
            hparams,
            world_size=1,
            rank=0,
            seed=0,
            use_fsdp=False,
        )

        for batch in tqdm(dataloader, desc=f"Evaluating {split}", leave=False):
            batch: Batch = batch.to(device)

            generation_output = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                entity_types=primary_dataset.entity_types,
                link_types=primary_dataset.link_types,
            )

            gold_entities, gold_links = decode_actions_and_pairings(
                model,
                batch["input_ids"],
                batch["actions"],
                batch["lr_pair_flag"],
                batch.get("rr_pair_flag", None),
                None,
                entity_types=primary_dataset.entity_types,
                link_types=primary_dataset.link_types,
            )

            all_pred_entities.extend(generation_output.entities)
            all_gold_entities.extend(gold_entities)
            all_pred_links.extend(generation_output.links)
            all_gold_links.extend(gold_links)

            batch_ner_kwargs = {
                "entities": generation_output.entities,
                "gold_entities": gold_entities,
                "average": "macro",
                "batched": True,
                "entity_types": list(
                    range(model.num_types - int(model.is_feature_extra_lr_class))
                ),
                "use_entity_tag": True,
            }
            batch_ner = metrics_ner(**batch_ner_kwargs)
            ner_metrics_acc = accumulate_macro_metrics(
                [ner_metrics_acc, batch_ner], primary_dataset.num_real_types
            )

            batch_ner_no_tag = metrics_ner(**{**batch_ner_kwargs, "use_entity_tag": False})
            ner_metrics_no_tag_acc = accumulate_macro_metrics(
                [ner_metrics_no_tag_acc, batch_ner_no_tag], 0
            )

            if not model.is_feature_ner_only:
                batch_ere_kwargs = {
                    "links": generation_output.links,
                    "gold_links": gold_links,
                    "average": "macro",
                    "batched": True,
                    "link_types": list(
                        range(model.num_links - int(model.is_feature_extra_rr_class))
                    ),
                    "use_entity_tag": True,
                    "symmetric_links": primary_dataset.symmetric_link_types,
                }
                if model.is_feature_mimic_pl_marker_eval:
                    batch_ere_kwargs["symmetric_links"] = set()

                batch_ere = metrics_ere(**batch_ere_kwargs)
                ere_metrics_acc = accumulate_macro_metrics(
                    [ere_metrics_acc, batch_ere], primary_dataset.num_links
                )

                batch_ere_no_tag = metrics_ere(**{**batch_ere_kwargs, "use_entity_tag": False})
                ere_metrics_no_tag_acc = accumulate_macro_metrics(
                    [ere_metrics_no_tag_acc, batch_ere_no_tag], primary_dataset.num_links
                )

            total_samples += batch["input_ids"].size(0)

    def build_metrics(ner_acc, ere_acc) -> Metrics:
        per_class = {k: calculate_f1(v, average="micro") for k, v in ner_acc.items()}
        m_pr, m_rec, m_f1 = calculate_f1(list(ner_acc.values()), average="macro")
        pr, rec, f1 = calculate_f1(
            accumulate_micro_metrics(list(ner_acc.values())), average="micro"
        )
        ere_per_class = {
            k: calculate_f1(v, average="micro") for k, v in ere_acc.items()
        }
        ere_m_pr, ere_m_rec, ere_m_f1 = calculate_f1(
            list(ere_acc.values()), average="macro"
        )
        ere_pr, ere_rec, ere_f1 = calculate_f1(
            accumulate_micro_metrics(list(ere_acc.values())), average="micro"
        )
        return Metrics(
            pr, rec, f1,
            ere_pr, ere_rec, ere_f1,
            m_pr, m_rec, m_f1,
            ere_m_pr, ere_m_rec, ere_m_f1,
            per_class,
            ere_per_class,
            samples_per_second=0.0,
            samples_total_num=total_samples,
            ner_per_class_names=primary_dataset.entity_types,
            ere_per_class_names=primary_dataset.link_types,
        )

    test_metrics = build_metrics(ner_metrics_acc, ere_metrics_acc)
    test_metrics_bounds = build_metrics(ner_metrics_no_tag_acc, ere_metrics_no_tag_acc)

    f1re = compute_f1re(
        all_pred_entities,
        all_gold_entities,
        all_pred_links,
        all_gold_links,
        symmetric_links=primary_dataset.symmetric_link_types,
    )

    return test_metrics, test_metrics_bounds, f1re


# ---------------------------------------------------------------------------
# Excel / CSV / JSON output helpers
# ---------------------------------------------------------------------------

def _write_csv(results: dict, output_path: Path = None):
    import csv
    if output_path is None:
        output_path = (
            PROJECT_ROOT / "artifacts" / "evaluation" / "evaluation_results.csv"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path
    headers = [
        "Config",
        "F1NER_P", "F1NER_R", "F1NER_F1",
        "F1RE_P", "F1RE_R", "F1RE_F1",
        "F1Rel_P", "F1Rel_R", "F1Rel_F1",
        "F1Rel+_P", "F1Rel+_R", "F1Rel+_F1",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for config_name, metrics in results.items():
            row = [config_name]
            for key in [
                "F1NER_Precision", "F1NER_Recall", "F1NER_F1",
                "F1RE_Precision", "F1RE_Recall", "F1RE_F1",
                "F1Rel_Precision", "F1Rel_Recall", "F1Rel_F1",
                "F1Rel+_Precision", "F1Rel+_Recall", "F1Rel+_F1",
            ]:
                row.append(round(metrics[key], 6))
            writer.writerow(row)
    print(f"CSV results saved to: {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LOCALLY TRAINED model checkpoints (not raw HF weights)"
    )
    parser.add_argument(
        "--models_dir",
        type=Path,
        default=DEFAULT_MODELS_BASE_DIR,
        help=f"Base directory containing trained models (default: {DEFAULT_MODELS_BASE_DIR})",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="Comma-separated config names (default: all 4 — wenyanwen_base,fact,multiscale,full)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_EVAL_SPLIT,
        help=f"Dataset split to evaluate on (default: {DEFAULT_EVAL_SPLIT})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
        help=f"Batch size for evaluation (default: {DEFAULT_EVAL_BATCH_SIZE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Excel path (default: artifacts/evaluation/evaluation_results.xlsx)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect cuda/cpu)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Direct path to a trained checkpoint directory (bypasses auto-search). "
        "When used with multiple --configs, the same checkpoint is used for all configs. "
        "Use with a single --configs for proper dataset loading.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only use locally cached Hugging Face files.",
    )
    args = parser.parse_args()

    models_base_dir = args.models_dir
    eval_split = args.split
    eval_batch_size = args.batch_size

    if args.configs:
        config_names = []
        for name in args.configs.split(","):
            name = name.strip()
            if "/" not in name:
                name = f"wenyanwen/{name}"
            config_names.append(name)
    else:
        config_names = ALL_CONFIG_NAMES

    if args.device:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading TRAINED checkpoints from: {models_base_dir}")
    print(f"Evaluation split: {eval_split}")
    print(f"Batch size: {eval_batch_size}")
    print(f"Configs to evaluate: {len(config_names)}")
    for c in config_names:
        print(f"  - {CONFIG_DISPLAY_NAMES.get(c, c)}")
    print()

    results: dict = {}

    for config_name in config_names:
        display_name = CONFIG_DISPLAY_NAMES.get(
            config_name, config_name.split("/")[-1]
        )
        print(f"{'=' * 60}")
        print(f"  Config: {display_name}  ({config_name})")
        print(f"{'=' * 60}")

        # ---- Step 1: Find best trained checkpoint ----
        if args.checkpoint:
            checkpoint_path = args.checkpoint
            has_weights = (
                (checkpoint_path / "model.safetensors").exists()
                or (checkpoint_path / "model.safetensors.index.json").exists()
            )
            if not has_weights:
                print(f"  [ERROR] model weights not found in {checkpoint_path}. SKIPPED\n")
                continue
        else:
            checkpoint_path = find_best_trained_checkpoint(config_name, models_base_dir)
            if checkpoint_path is None:
                print(f"  [ERROR] No trained checkpoint found for {display_name}. SKIPPED\n")
                continue

        print(f"  Trained checkpoint: {checkpoint_path}")

        # ---- Step 2: Load FULL trained model (weights + config from checkpoint) ----
        load_kwargs = {"local_files_only": True} if args.offline else {}
        config = ITERConfig.from_pretrained(checkpoint_path, **load_kwargs)
        model_cls = (
            ITER if config.is_feature_ner_only else ITERForRelationExtraction
        )
        model = model_cls.from_pretrained(checkpoint_path, **load_kwargs)
        model = model.to(device)
        model.eval()

        numel = sum(p.numel() for p in model.parameters())
        print(f"  Params: {numel / 1e6:.1f}M")

        # ---- Step 3: Load dataset ----
        hparams = None
        dataset = None
        load_errors = []

        try:
            hparams = Hparams.from_name(config_name, eval_batch_size=eval_batch_size)
            dataset = CoNLL.from_name(config_name, tokenizer=model.tokenizer)
        except (ValueError, FileNotFoundError, KeyError) as e:
            load_errors.append(f"config_name='{config_name}': {e}")

        if dataset is None:
            short_name = config_name.split("/")[-1] if "/" in config_name else config_name
            if short_name != config_name:
                try:
                    hparams = Hparams.from_name(short_name, eval_batch_size=eval_batch_size)
                    dataset = CoNLL.from_name(short_name, tokenizer=model.tokenizer)
                except (ValueError, FileNotFoundError, KeyError) as e:
                    load_errors.append(f"short_name='{short_name}': {e}")

        if dataset is None and config.dataset:
            ds_name = config.dataset
            try:
                hparams = Hparams.from_name(ds_name, eval_batch_size=eval_batch_size)
                dataset = CoNLL.from_name(ds_name, tokenizer=model.tokenizer)
            except (ValueError, FileNotFoundError, KeyError) as e:
                load_errors.append(f"from model config='{ds_name}': {e}")

        if dataset is None:
            print(f"  [ERROR] Could not load dataset config. Errors:")
            for err in load_errors:
                print(f"    - {err}")
            print(f"  -> SKIPPED\n")
            continue

        if not model.config.entity_types:
            model.config.entity_types = dataset.entity_types
        if not model.config.link_types:
            model.config.link_types = dataset.link_types

        dataset.setup_dataset()
        print(f"  {eval_split} samples: {len(dataset[eval_split])}")

        # ---- Step 4: Evaluate ----
        test_metrics, test_metrics_bounds, (f1re_p, f1re_r, f1re_f1) = (
            evaluate_model_with_raw(model, dataset, hparams, split=eval_split)
        )

        # ---- Step 5: Collect results ----
        results[display_name] = {
            "F1NER_Precision": test_metrics.ner_pr,
            "F1NER_Recall": test_metrics.ner_rec,
            "F1NER_F1": test_metrics.ner_f1,
            "F1RE_Precision": f1re_p,
            "F1RE_Recall": f1re_r,
            "F1RE_F1": f1re_f1,
            "F1Rel_Precision": test_metrics_bounds.ere_pr,
            "F1Rel_Recall": test_metrics_bounds.ere_rec,
            "F1Rel_F1": test_metrics_bounds.ere_f1,
            "F1Rel+_Precision": test_metrics.ere_pr,
            "F1Rel+_Recall": test_metrics.ere_rec,
            "F1Rel+_F1": test_metrics.ere_f1,
        }

        # ---- Print summary ----
        print(f"\n  {'Metric':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-' * 42}")
        print(
            f"  {'F1NER':<12} {test_metrics.ner_pr:>10.4f} "
            f"{test_metrics.ner_rec:>10.4f} {test_metrics.ner_f1:>10.4f}"
        )
        print(
            f"  {'F1RE':<12} {f1re_p:>10.4f} "
            f"{f1re_r:>10.4f} {f1re_f1:>10.4f}"
        )
        print(
            f"  {'F1Rel':<12} {test_metrics_bounds.ere_pr:>10.4f} "
            f"{test_metrics_bounds.ere_rec:>10.4f} {test_metrics_bounds.ere_f1:>10.4f}"
        )
        print(
            f"  {'F1Rel+':<12} {test_metrics.ere_pr:>10.4f} "
            f"{test_metrics.ere_rec:>10.4f} {test_metrics.ere_f1:>10.4f}"
        )
        print()

    # ===================================================================
    # Write results to Excel
    # ===================================================================
    if not results:
        print("No results to write. All configs were skipped.")
        return

    try:
        import openpyxl
    except ImportError:
        print("\n[WARN] openpyxl not installed. Install with: pip install openpyxl")
        print("Writing CSV fallback instead...")
        csv_output = args.output.with_suffix(".csv") if args.output else None
        _write_csv(results, csv_output)
        return

    wb = openpyxl.Workbook()

    # ---- Sheet 1: Summary table ----
    ws = wb.active
    ws.title = "Metrics Summary"

    headers = [
        "Config",
        "F1NER_P", "F1NER_R", "F1NER_F1",
        "F1RE_P", "F1RE_R", "F1RE_F1",
        "F1Rel_P", "F1Rel_R", "F1Rel_F1",
        "F1Rel+_P", "F1Rel+_R", "F1Rel+_F1",
    ]
    for col_idx, h in enumerate(headers, 1):
        ws.cell(row=1, column=col_idx, value=h)

    from openpyxl.styles import Font

    header_font = Font(bold=True)
    for col_idx in range(1, len(headers) + 1):
        ws.cell(row=1, column=col_idx).font = header_font

    for row_idx, (config_name, metrics) in enumerate(results.items(), 2):
        ws.cell(row=row_idx, column=1, value=config_name)
        col = 2
        for key in [
            "F1NER_Precision", "F1NER_Recall", "F1NER_F1",
            "F1RE_Precision", "F1RE_Recall", "F1RE_F1",
            "F1Rel_Precision", "F1Rel_Recall", "F1Rel_F1",
            "F1Rel+_Precision", "F1Rel+_Recall", "F1Rel+_F1",
        ]:
            ws.cell(row=row_idx, column=col, value=round(metrics[key], 6))
            col += 1

    # ---- Sheet 2: Per-metric comparison ----
    ws2 = wb.create_sheet("F1 Comparison")

    metric_names = ["F1NER", "F1RE", "F1Rel", "F1Rel+"]
    ws2.cell(row=1, column=1, value="Config")
    for i, name in enumerate(metric_names):
        ws2.cell(row=1, column=2 + i * 3, value=f"{name}_P")
        ws2.cell(row=1, column=3 + i * 3, value=f"{name}_R")
        ws2.cell(row=1, column=4 + i * 3, value=f"{name}_F1")
    for col_idx in range(1, 2 + len(metric_names) * 3):
        ws2.cell(row=1, column=col_idx).font = header_font

    for row_idx, (config_name, metrics) in enumerate(results.items(), 2):
        ws2.cell(row=row_idx, column=1, value=config_name)
        col = 2
        for name in metric_names:
            ws2.cell(row=row_idx, column=col, value=round(metrics[f"{name}_Precision"], 6))
            ws2.cell(row=row_idx, column=col + 1, value=round(metrics[f"{name}_Recall"], 6))
            ws2.cell(row=row_idx, column=col + 2, value=round(metrics[f"{name}_F1"], 6))
            col += 3

    # ---- Sheet 3: F1-only summary ----
    ws3 = wb.create_sheet("F1 Only")
    f1_headers = ["Config", "F1NER", "F1RE", "F1Rel", "F1Rel+"]
    for col_idx, h in enumerate(f1_headers, 1):
        ws3.cell(row=1, column=col_idx, value=h)
        ws3.cell(row=1, column=col_idx).font = header_font

    for row_idx, (config_name, metrics) in enumerate(results.items(), 2):
        ws3.cell(row=row_idx, column=1, value=config_name)
        for col_idx, name in enumerate(metric_names, 2):
            ws3.cell(row=row_idx, column=col_idx, value=round(metrics[f"{name}_F1"], 6))

    # Auto-fit column widths
    for ws in [ws, ws2, ws3]:
        for col in ws.columns:
            max_length = 0
            col_letter = col[0].column_letter
            for cell in col:
                try:
                    max_length = max(max_length, len(str(cell.value)))
                except Exception:
                    pass
            ws.column_dimensions[col_letter].width = min(max_length + 4, 30)

    if args.output:
        output_path = args.output
    else:
        output_path = (
            PROJECT_ROOT / "artifacts" / "evaluation" / "evaluation_results.xlsx"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 60}")

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results also saved to: {json_path}")


if __name__ == "__main__":
    main()
