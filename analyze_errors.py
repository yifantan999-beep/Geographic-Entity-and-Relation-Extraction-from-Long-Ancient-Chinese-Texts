#!/usr/bin/env python3
"""
Analyze failed predictions for all wenyanwen model configurations on the test set.

For each configuration, loads the best model checkpoint, runs inference on the
test split, and writes every prediction failure to an Excel file with columns:

  - Sample ID / 原文 (original text)
  - 标签 (gold label: expected entities / relations)
  - 模型输出 (model prediction: predicted entities / relations)
  - 差异分析 (difference analysis: FN / FP / type error breakdown)

Output: artifacts/analysis/evaluation_error_analysis.xlsx (one sheet per config)
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Offline mode: set BEFORE any transformers / huggingface imports.
# Use environment variable EVAL_OFFLINE=1 or pass --offline flag.
# ---------------------------------------------------------------------------
if os.environ.get("EVAL_OFFLINE", "").lower() in ("1", "true", "yes"):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

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
    remove_entity_tag_from_entities,
    remove_entity_tag_from_links,
    remove_extra_info_from_links,
)
from iter.modeling_iter2 import ITERForRelationExtraction
from iter.generation_utils import decode_actions_and_pairings
from iter.data.data_collator import Batch
from iter.tools.dataloaders import create_primary_dataloader


# ---------------------------------------------------------------------------
# Defaults
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
_DEFAULT_MAX_ERRORS = 5000  # safety cap (used only as argparse default)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_best_model(config_name: str, models_base: Path) -> Optional[Path]:
    """Find the best completed model checkpoint for a configuration."""
    short_name = CONFIG_DISPLAY_NAMES.get(config_name, config_name.split("/")[-1])
    config_dir = models_base / short_name

    if not config_dir.exists():
        print(f"  [WARN] Config directory not found: {config_dir}")
        return None

    model_dirs = []
    for config_file in config_dir.rglob("config.json"):
        run_dir = config_file.parent
        has_weights = (
            (run_dir / "model.safetensors").exists()
            or (run_dir / "model.safetensors.index.json").exists()
        )
        if has_weights:
            model_dirs.append(run_dir)

    if not model_dirs:
        print(f"  [WARN] No model checkpoints found in {config_dir}")
        return None

    model_dirs_with_metrics = [d for d in model_dirs if (d / "metrics.json").exists()]
    if model_dirs_with_metrics:
        best_dir, best_f1 = None, -1.0
        for d in model_dirs_with_metrics:
            try:
                with open(d / "metrics.json") as f:
                    data = json.load(f)
                metrics = data.get("test_metrics", {})
                f1 = metrics.get("f1", metrics.get("ere_f1", metrics.get("ner_f1", 0.0)))
                if f1 > best_f1:
                    best_f1, best_dir = f1, d
            except Exception:
                pass
        return best_dir or model_dirs_with_metrics[-1]

    model_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return model_dirs[0]


def load_json_data(dataset: CoNLL, split: str):
    """
    Load the raw JSON data for the given split so we can reconstruct original
    text and entity/relation labels per sample.
    """
    file_path = dataset.split_path(dataset.splits[split], final=False)
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def tokens_to_text(tokens: list) -> str:
    """Join tokens into readable text."""
    return "".join(tokens) if tokens else ""


def entity_to_str(ent: tuple, entity_types: list) -> str:
    """
    Convert an entity tuple (tokens, type_id, text, type_str) to a readable string.
    """
    tokens, type_id, text, type_str = ent
    type_name = type_str
    if isinstance(type_name, str) and type_name == "<ignored>":
        type_name = entity_types[type_id] if 0 <= type_id < len(entity_types) else f"type_{type_id}"
    return f"[{text}] ({type_name})"


def relation_to_str(link: tuple, entity_types: list, link_types: list) -> str:
    """
    Convert a link tuple (head_entity, link_type_id, tail_entity, ...) to string.
    """
    head, lt_id, tail = link[:3]
    lt_name = link_types[lt_id] if 0 <= lt_id < len(link_types) else f"rel_{lt_id}"
    head_str = entity_to_str(head, entity_types)
    tail_str = entity_to_str(tail, entity_types)
    return f"{head_str} --[{lt_name}]--> {tail_str}"


# ---------------------------------------------------------------------------
# Failure extraction
# ---------------------------------------------------------------------------

def analyze_sample_failures(
    sample_idx: int,
    tokens: list,
    pred_entities: list,
    gold_entities: list,
    pred_links: list,
    gold_links: list,
    entity_types: list,
    link_types: list,
    symmetric_links: set,
):
    """
    Compare predicted vs gold entities/relations and return a list of failure
    records for one sample.

    Each record is a dict with keys:
      sample_id, 原文, 类别, 标签, 模型输出, 差异分析
    """
    failures = []
    original_text = tokens_to_text(tokens)

    # ------- Entity analysis -------
    # Build comparable sets
    pred_ent_set = set(pred_entities)       # (tokens, type, text, type_str)
    gold_ent_set = set(gold_entities)

    # Boundary-level comparison (ignore type)
    pred_by_boundary = {}
    for e in pred_entities:
        pred_by_boundary[e[0]] = e
    gold_by_boundary = {}
    for e in gold_entities:
        gold_by_boundary[e[0]] = e

    # 1a) Entity FN: missed entities (in gold but not predicted)
    fn_entities = gold_ent_set - pred_ent_set
    # Check if any FN entity has a boundary match with wrong type
    fn_boundary_match_type_error = []
    fn_strict = []
    for e in sorted(fn_entities, key=lambda x: x[2]):  # sort by text
        boundary = e[0]
        if boundary in pred_by_boundary:
            # Boundary matches but type or text differs → type error
            fn_boundary_match_type_error.append((e, pred_by_boundary[boundary]))
        else:
            fn_strict.append(e)

    for gold_ent in fn_strict:
        gold_str = entity_to_str(gold_ent, entity_types) if entity_types else str(gold_ent)
        failures.append({
            "sample_id": sample_idx,
            "原文": original_text,
            "类别": "实体-FN(漏识别)",
            "标签": gold_str,
            "模型输出": "(无)",
            "差异分析": f"未识别出实体: {gold_str}",
        })

    for gold_ent, pred_ent in fn_boundary_match_type_error:
        gold_str = entity_to_str(gold_ent, entity_types) if entity_types else str(gold_ent)
        pred_str = entity_to_str(pred_ent, entity_types) if entity_types else str(pred_ent)
        failures.append({
            "sample_id": sample_idx,
            "原文": original_text,
            "类别": "实体-类型错误",
            "标签": gold_str,
            "模型输出": pred_str,
            "差异分析": f"边界正确但类型错误: 标签={gold_str}, 预测={pred_str}",
        })

    # 1b) Entity FP: spurious entities (predicted but not in gold)
    fp_entities = pred_ent_set - gold_ent_set
    for pred_ent in sorted(fp_entities, key=lambda x: x[2]):
        boundary = pred_ent[0]
        if boundary in gold_by_boundary:
            # boundary matches gold but type differs → already covered above as type error
            continue
        pred_str = entity_to_str(pred_ent, entity_types) if entity_types else str(pred_ent)
        failures.append({
            "sample_id": sample_idx,
            "原文": original_text,
            "类别": "实体-FP(误识别)",
            "标签": "(无)",
            "模型输出": pred_str,
            "差异分析": f"多识别了不存在的实体: {pred_str}",
        })

    # ------- Relation analysis -------
    if not pred_links and not gold_links:
        return failures

    # Build comparable link sets (entity-type ignored for boundary-level rel match)
    def normalize_link(link):
        """Return (head_boundary, link_type, tail_boundary) for comparison."""
        h, lt, t = link[:3]
        return (h[0], lt, t[0])

    def normalize_link_with_entity_type(link):
        """Return (head_full, link_type, tail_full) for strict comparison."""
        h, lt, t = link[:3]
        return (h, lt, t)

    pred_link_boundary_map = {}
    for l in pred_links:
        key = normalize_link(l)
        pred_link_boundary_map[key] = l

    gold_link_boundary_map = {}
    for l in gold_links:
        key = normalize_link(l)
        gold_link_boundary_map[key] = l

    pred_link_set_boundary = set(pred_link_boundary_map.keys())
    gold_link_set_boundary = set(gold_link_boundary_map.keys())

    pred_link_set_strict = set(normalize_link_with_entity_type(l) for l in pred_links)
    gold_link_set_strict = set(normalize_link_with_entity_type(l) for l in gold_links)

    # Handle symmetric relations
    def handle_sym(links, sym_ids):
        result = []
        for l in links:
            h, lt, t = l[:3]
            if lt in sym_ids:
                result.append((frozenset({h[0], lt, t[0]}), lt))
            else:
                result.append(((h[0], lt, t[0]), lt))
        return result

    # 2a) Relation FN (strict): in gold but NOT in pred (strict match)
    fn_links_strict = gold_link_set_strict - pred_link_set_strict
    fn_strict_boundary_keys = set(normalize_link(l) for l in [
        gold_links[i] for i, gl in enumerate(gold_links)
        if normalize_link_with_entity_type(gl) in fn_links_strict
    ])

    # 2b) Relation FP (strict): in pred but NOT in gold
    fp_links_strict = pred_link_set_strict - gold_link_set_strict

    # Classify each FN/FP
    for link in gold_links:
        nk = normalize_link(link)
        nk_strict = normalize_link_with_entity_type(link)
        if nk_strict not in fn_links_strict:
            continue
        gold_str = relation_to_str(link, entity_types, link_types)

        if nk in pred_link_boundary_map:
            # boundary + rel type match, but entity type differs → entity type caused failure
            pred_link = pred_link_boundary_map[nk]
            pred_str = relation_to_str(pred_link, entity_types, link_types)
            failures.append({
                "sample_id": sample_idx,
                "原文": original_text,
                "类别": "关系-实体类型不匹配",
                "标签": gold_str,
                "模型输出": pred_str,
                "差异分析": "关系边界和类型正确，但实体类型不匹配",
            })
        elif any(normalize_link(pl)[:2] == nk[:2] or normalize_link(pl)[1:] == nk[1:]
                 for pl in pred_links):
            # Partial match — one entity matches, type matches
            failures.append({
                "sample_id": sample_idx,
                "原文": original_text,
                "类别": "关系-FN(漏识别)",
                "标签": gold_str,
                "模型输出": "(无或部分匹配)",
                "差异分析": f"关系未被正确识别，实体或关系类型有误",
            })
        else:
            failures.append({
                "sample_id": sample_idx,
                "原文": original_text,
                "类别": "关系-FN(漏识别)",
                "标签": gold_str,
                "模型输出": "(无)",
                "差异分析": f"关系完全未被识别",
            })

    for link in pred_links:
        nk_strict = normalize_link_with_entity_type(link)
        if nk_strict not in fp_links_strict:
            continue
        nk = normalize_link(link)

        # Skip if boundary+type match means it's just entity type mismatch
        if nk in gold_link_boundary_map:
            continue

        pred_str = relation_to_str(link, entity_types, link_types)
        failures.append({
            "sample_id": sample_idx,
            "原文": original_text,
            "类别": "关系-FP(误识别)",
            "标签": "(无)",
            "模型输出": pred_str,
            "差异分析": f"多识别了不存在的关系: {pred_str}",
        })

    return failures


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def analyze_config(
    config_name: str,
    models_base_dir: Path,
    eval_split: str,
    eval_batch_size: int,
    device: str,
    max_errors: int = 5000,
    offline: bool = False,
) -> list:
    """Run error analysis for one config. Returns list of failure dicts."""
    display_name = CONFIG_DISPLAY_NAMES.get(config_name, config_name.split("/")[-1])

    print(f"\n{'=' * 60}")
    print(f"  Analyzing: {display_name}")
    print(f"{'=' * 60}")

    # ---- Find & load model ----
    model_path = find_best_model(config_name, models_base_dir)
    if model_path is None:
        print(f"  -> SKIPPED\n")
        return []
    print(f"  Model: {model_path}")

    load_kwargs = {"local_files_only": True} if offline else {}
    pretrained_config = ITERConfig.from_pretrained(model_path, **load_kwargs)
    model_cls = ITER if pretrained_config.is_feature_ner_only else ITERForRelationExtraction
    model = model_cls.from_pretrained(model_path, **load_kwargs)
    model = model.to(device)
    model.eval()

    # ---- Load dataset ----
    dataset = None
    load_errors = []
    for attempt_name in [config_name,
                         config_name.split("/")[-1] if "/" in config_name else None,
                         pretrained_config.dataset]:
        if attempt_name is None:
            continue
        try:
            hparams = Hparams.from_name(attempt_name, eval_batch_size=eval_batch_size)
            dataset = CoNLL.from_name(attempt_name, tokenizer=model.tokenizer)
            break
        except (ValueError, FileNotFoundError, KeyError) as e:
            load_errors.append(f"'{attempt_name}': {e}")

    if dataset is None:
        print(f"  [ERROR] Could not load dataset: {'; '.join(load_errors)}")
        return []

    if not model.config.entity_types:
        model.config.entity_types = dataset.entity_types
    if not model.config.link_types:
        model.config.link_types = dataset.link_types
    dataset.setup_dataset()

    # Load raw JSON for original text
    raw_data = load_json_data(dataset, eval_split)
    print(f"  Samples: {len(dataset[eval_split])}")

    # ---- Run inference and collect failures ----
    all_failures = []
    sample_idx = 0  # global sample counter across batches

    with torch.no_grad():
        dataloader = create_primary_dataloader(
            [dataset], eval_split, False, hparams, world_size=1, rank=0, seed=0, use_fsdp=False,
        )

        for batch in tqdm(dataloader, desc=f"Inference {eval_split}", leave=False):
            batch: Batch = batch.to(device)
            batch_size = batch["input_ids"].size(0)

            generation_output = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                entity_types=dataset.entity_types,
                link_types=dataset.link_types,
            )

            gold_entities, gold_links = decode_actions_and_pairings(
                model,
                batch["input_ids"],
                batch["actions"],
                batch["lr_pair_flag"],
                batch.get("rr_pair_flag", None),
                None,
                entity_types=dataset.entity_types,
                link_types=dataset.link_types,
            )

            for i in range(batch_size):
                if sample_idx >= len(raw_data):
                    break

                sample = raw_data[sample_idx]
                tokens = sample.get("tokens", [])

                pred_ents = generation_output.entities[i] if i < len(generation_output.entities) else []
                gold_ents = gold_entities[i] if i < len(gold_entities) else []
                pred_lnks = generation_output.links[i] if i < len(generation_output.links) else []
                gold_lnks = gold_links[i] if i < len(gold_links) else []

                failures = analyze_sample_failures(
                    sample_idx=sample_idx,
                    tokens=tokens,
                    pred_entities=pred_ents,
                    gold_entities=gold_ents,
                    pred_links=pred_lnks,
                    gold_links=gold_lnks,
                    entity_types=dataset.entity_types,
                    link_types=dataset.link_types,
                    symmetric_links=dataset.symmetric_link_types,
                )
                all_failures.extend(failures)

                sample_idx += 1

                if len(all_failures) >= max_errors:
                    print(f"  [WARN] Reached max errors cap ({max_errors}), truncating...")
                    break

            if len(all_failures) >= max_errors:
                break

    print(f"  Total failures collected: {len(all_failures)}")
    return all_failures


# ---------------------------------------------------------------------------
# Excel output
# ---------------------------------------------------------------------------

def write_errors_to_excel(
    config_failures: dict,  # {config_name: [failure_dict, ...]}
    output_path: Path,
):
    """Write error analysis to an Excel workbook with one sheet per config."""
    try:
        import openpyxl
    except ImportError:
        print("[WARN] openpyxl not installed. Writing JSON instead.")
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            # Convert to serializable format
            serializable = {
                cfg: [
                    {k: str(v) for k, v in rec.items()}
                    for rec in records
                ]
                for cfg, records in config_failures.items()
            }
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"JSON saved to: {json_path}")
        return

    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # Color definitions
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=11)
    fn_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")  # light orange
    fp_fill = PatternFill(start_color="D9E2F3", end_color="D9E2F3", fill_type="solid")  # light blue
    type_err_fill = PatternFill(start_color="FFFFCC", end_color="FFFFCC", fill_type="solid")  # light yellow
    wrap_align = Alignment(wrap_text=True, vertical="top")
    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin"),
    )

    columns = ["sample_id", "原文", "类别", "标签", "模型输出", "差异分析"]
    col_widths = [10, 50, 20, 40, 40, 45]

    for config_name, failures in config_failures.items():
        if not failures:
            continue

        # Truncate sheet name to 31 chars (Excel limit)
        sheet_name = config_name[:31]
        ws = wb.create_sheet(title=sheet_name)

        # ---- Summary row ----
        # Count by category
        cat_counts = defaultdict(int)
        for f in failures:
            cat_counts[f.get("类别", "未知")] += 1

        summary_parts = [f"{cat}: {cnt}" for cat, cnt in sorted(cat_counts.items())]
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(columns))
        summary_cell = ws.cell(row=1, column=1,
                               value=f"【{config_name}】错误总数: {len(failures)} | " + " | ".join(summary_parts))
        summary_cell.font = Font(bold=True, size=12, color="C00000")

        # ---- Header row ----
        for col_idx, header in enumerate(columns, 1):
            cell = ws.cell(row=2, column=col_idx, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
            cell.border = thin_border

        # Set column widths
        for col_idx, width in enumerate(col_widths, 1):
            ws.column_dimensions[get_column_letter(col_idx)].width = width

        # ---- Data rows ----
        for row_idx, rec in enumerate(failures, 3):
            for col_idx, key in enumerate(columns, 1):
                value = rec.get(key, "")
                cell = ws.cell(row=row_idx, column=col_idx, value=value)
                cell.alignment = wrap_align
                cell.border = thin_border

                # Color rows by error category
                category = rec.get("类别", "")
                if "FN" in category:
                    cell.fill = fn_fill
                elif "FP" in category:
                    cell.fill = fp_fill
                elif "类型" in category:
                    cell.fill = type_err_fill

        # Freeze header
        ws.freeze_panes = "A3"

        # Auto-filter
        ws.auto_filter.ref = f"A2:{get_column_letter(len(columns))}{len(failures) + 2}"

        print(f"  Sheet '{sheet_name}': {len(failures)} rows written")

    wb.save(output_path)
    print(f"\nSaved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze prediction failures for all wenyanwen configs on the test set"
    )
    parser.add_argument("--models_dir", type=Path, default=DEFAULT_MODELS_BASE_DIR,
                        help="Base directory for trained models")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config names (default: all four)")
    parser.add_argument("--split", type=str, default=DEFAULT_EVAL_SPLIT,
                        help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_EVAL_BATCH_SIZE,
                        help="Batch size for inference")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output Excel path")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto-detect)")
    parser.add_argument("--max_errors", type=int, default=_DEFAULT_MAX_ERRORS,
                        help="Max error rows per config (safety cap)")
    parser.add_argument("--offline", action="store_true", default=False,
                        help="Offline mode: do not attempt to download models from HuggingFace")
    args = parser.parse_args()

    # Apply offline mode early
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    models_base_dir = args.models_dir
    max_errors = args.max_errors

    if args.configs:
        config_names = []
        for name in args.configs.split(","):
            name = name.strip()
            if "/" not in name:
                name = f"wenyanwen/{name}"
            config_names.append(name)
    else:
        config_names = ALL_CONFIG_NAMES

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Models dir: {models_base_dir}")
    print(f"Configs: {len(config_names)}")
    print(f"Max errors per config: {max_errors}")
    print()

    # ---- Analyze each config ----
    config_failures = {}
    for config_name in config_names:
        display_name = CONFIG_DISPLAY_NAMES.get(config_name, config_name.split("/")[-1])
        failures = analyze_config(
            config_name, models_base_dir, args.split, args.batch_size, device,
            max_errors=max_errors,
            offline=args.offline,
        )
        config_failures[display_name] = failures

    total = sum(len(v) for v in config_failures.values())
    if total == 0:
        print("\nNo failures found — all predictions are perfect (or no models loaded).")
        return

    # ---- Write to Excel ----
    output_path = args.output or (
        PROJECT_ROOT / "artifacts" / "analysis" / "evaluation_error_analysis.xlsx"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_errors_to_excel(config_failures, output_path)

    # ---- Print summary ----
    print(f"\n{'=' * 60}")
    print("Summary:")
    for cfg, failures in config_failures.items():
        cat_counts = defaultdict(int)
        for f in failures:
            cat = f.get("类别", "未知")
            # Simplify category
            if "FN" in cat:
                cat_counts["FN(漏识别)"] += 1
            elif "FP" in cat:
                cat_counts["FP(误识别)"] += 1
            elif "类型" in cat:
                cat_counts["类型错误"] += 1
            else:
                cat_counts[cat] += 1
        parts = " | ".join(f"{k}: {v}" for k, v in sorted(cat_counts.items()))
        print(f"  {cfg}: {len(failures)} errors ({parts})")
    print(f"  TOTAL: {total} errors")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
