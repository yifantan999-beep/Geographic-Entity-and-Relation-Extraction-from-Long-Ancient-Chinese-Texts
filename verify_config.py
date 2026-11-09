#!/usr/bin/env python3
"""
配置链路验证脚本
=================
一键检查从 JSON 配置文件 → CoNLL → ITERConfig → Model 的完整配置传递链路。

用法:
    # 检查所有 wenyanwen 相关配置
    python verify_config.py --dataset wenyanwen

    # 检查指定配置文件
    python verify_config.py --dataset wenyanwen_parallel_no_iso

    # 检查并打印完整模型结构
    python verify_config.py --dataset wenyanwen --print-model

    # 列出所有可用配置
    python verify_config.py --list
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional


# ── Color output helpers ──────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str) -> str:
    return f"{GREEN}✅ {msg}{RESET}"


def fail(msg: str) -> str:
    return f"{RED}❌ {msg}{RESET}"


def warn(msg: str) -> str:
    return f"{YELLOW}⚠️  {msg}{RESET}"


def title(msg: str) -> str:
    return f"\n{BOLD}{'='*70}{RESET}\n{BOLD}  {msg}{RESET}\n{BOLD}{'='*70}{RESET}"


def section(msg: str) -> str:
    return f"\n{BOLD}── {msg} ──{RESET}"


# ── Feature bit definitions ───────────────────────────────────────────
FEATURE_BITS = {
    0: "NER hidden state offset",
    1: "Sum representations",
    2: "Examples without links (inverted)",
    3: "Extended context window",
    4: "LSE loss for rr_score",
    5: "Extra LR class",
    6: "T5 decoder as decoder",
    7: "Extra RR class",
    8: "Batch average loss",
    9: "Nest depth > 1",
    10: "Optimized inference",
    11: "NER Only",
    12: "T5 decoder as encoder",
    13: "Empty examples",
    14: "Negative LSE samples",
    15: "CE loss for RR pair",
    16: "Train symrels both directions",
    17: "Mimic PL Marker eval",
    18: "Negsample link types",
    19: "Behave like PL Marker",
    20: "Development flag",
    21: "Don't create rr_score",
    22: "Fact Decomposition (dual-branch B)",
    23: "Multi-Scale Geographic (dual-branch A)",
    24: "Iterative refinement",
    25: "Multi-scale consistency loss",
    26: "Gradient isolation (parallel branches)",
    27: "Hierarchical long-context NER boundary refinement",
}


def decode_features(features_int: int) -> list[tuple[int, str, bool]]:
    """Decode integer feature bitmask into list of (bit, description, enabled)."""
    results = []
    for bit, desc in sorted(FEATURE_BITS.items()):
        enabled = (features_int & (1 << bit)) != 0
        results.append((bit, desc, enabled))
    return results


# ── Step 1: Find and parse config files ───────────────────────────────
def find_config_files(dataset_name: str) -> list[Path]:
    """Find all config files matching a dataset name."""
    cfg_dir = Path("cfg")
    candidates = []

    # Direct match: cfg/dataset_name.json
    direct = cfg_dir / f"{dataset_name}.json"
    if direct.exists():
        candidates.append(direct)

    # Subdirectory match: cfg/dataset_name/*.json
    subdir = cfg_dir / dataset_name
    if subdir.is_dir():
        for f in sorted(subdir.glob("*.json")):
            candidates.append(f)

    # Template match
    template = cfg_dir / f"{dataset_name}.template.json"
    if template.exists():
        candidates.append(template)

    return candidates


def list_all_configs():
    """List all available dataset configs."""
    cfg_dir = Path("cfg")
    configs = []

    # Top-level JSON files
    for f in sorted(cfg_dir.glob("*.json")):
        name = f.stem
        if not name.endswith(".template"):
            configs.append(name)

    # Subdirectories
    for d in sorted(cfg_dir.iterdir()):
        if d.is_dir() and not d.name.startswith(".") and not d.name == "__pycache__":
            configs.append(f"{d.name}/ (directory with {len(list(d.glob('*.json')))} files)")

    return configs


# ── Step 2: Check Hparams loading ─────────────────────────────────────
def check_hparams(dataset_name: str) -> dict:
    """Load and validate Hparams from dataset name."""
    print(section("Hparams (Training Hyperparameters)"))

    try:
        from iter.datasets.training import Hparams
        hparams = Hparams.from_name(dataset_name)
        print(f"  {ok('Hparams.from_name() succeeded')}")
        print(f"  Config source: {hparams._config_path if hasattr(hparams, '_config_path') else 'N/A'}")

        fields = {
            "optimize_for": hparams.optimize_for,
            "metric_average": hparams.metric_average,
            "batch_size": hparams.batch_size,
            "dropout": hparams.dropout,
            "d_ff": hparams.d_ff,
            "activation_fn": hparams.activation_fn,
            "max_epochs": hparams.max_epochs,
            "num_epochs": hparams.num_epochs,
            "patience": hparams.patience,
            "gradient_accumulation": hparams.gradient_accumulation,
            "lr_scheduler": hparams.lr_scheduler,
            "warmup_steps": hparams.warmup_steps,
            "task_lr_scheduler": hparams.task_lr_scheduler,
            "task_warmup_steps": hparams.task_warmup_steps,
        }

        for k, v in fields.items():
            print(f"    {k}: {v}")

        return {"status": "ok", "hparams": hparams}
    except Exception as e:
        print(f"  {fail(f'Hparams.from_name() failed: {e}')}")
        return {"status": "fail", "error": str(e)}


# ── Step 3: Check CoNLL dataset loading ───────────────────────────────
def check_conll(dataset_name: str) -> dict:
    """Load and validate CoNLL dataset, checking all feature-related attrs."""
    print(section("CoNLL Dataset"))

    try:
        from iter.datasets import CoNLL
        dataset = CoNLL.from_name(dataset_name)
        print(f"  {ok('CoNLL.from_name() succeeded')}")
    except Exception as e:
        print(f"  {fail(f'CoNLL.from_name() failed: {e}')}")
        return {"status": "fail", "error": str(e)}

    # ── Check features ──
    features_val = dataset.features
    print(f"\n  Features (integer bitmask): {features_val}")
    print(f"  Features (binary): {bin(features_val)}")

    # Decode active bits
    active_bits = []
    inactive_critical = []
    decoded = decode_features(features_val)
    for bit, desc, enabled in decoded:
        if enabled:
            active_bits.append(f"bit {bit}: {desc}")
        elif bit in (22, 23, 26):  # Critical bits for dual-branch
            inactive_critical.append(f"bit {bit}: {desc}")

    if active_bits:
        print(f"\n  {GREEN}Active feature bits:{RESET}")
        for b in active_bits:
            print(f"    {GREEN}●{RESET} {b}")
    else:
        print(f"  {warn('No feature bits enabled (features=0) — using base model only')}")

    if inactive_critical:
        print(f"\n  {YELLOW}Dual-branch related bits NOT enabled:{RESET}")
        for b in inactive_critical:
            print(f"    {YELLOW}○{RESET} {b}")

    # ── Check loss weights ──
    print(f"\n  Loss & structure attributes from CoNLL:")
    critical_attrs = {
        "fact_loss_weight": (0.5, "Fact decomposition loss weight"),
        "multiscale_loss_weight": (0.1, "Multi-scale consistency loss weight"),
        "use_gradient_isolation": (True, "Gradient isolation between branches"),
        "is_feature_ner_only": (False, "NER-only mode"),
        "num_types": (None, "Number of entity types"),
        "num_links": (None, "Number of relation types"),
        "max_length": (None, "Max sequence length"),
        "entity_nest_depth": (None, "Entity nest depth"),
    }

    all_ok = True
    for attr, (default, desc) in critical_attrs.items():
        if hasattr(dataset, attr):
            val = getattr(dataset, attr)
            print(f"    {attr}: {val}  ({desc})")
        else:
            print(f"    {fail(f'MISSING: {attr} ({desc}) — will cause AttributeError in train.py')}")
            all_ok = False

    # Also check if feature property methods work
    feature_props = {
        "is_feature_fact_decomposition": (22, "Fact Decomposition"),
        "is_feature_multiscale_geo": (23, "Multi-Scale Geo"),
        "is_feature_gradient_isolation": (26, "Gradient Isolation"),
    }
    print(f"\n  Feature property checks:")
    for prop_name, (bit, desc) in feature_props.items():
        if hasattr(dataset, prop_name):
            val = getattr(dataset, prop_name)
            status = ok("ON") if val else warn("OFF")
            print(f"    {prop_name}: {status}  (bit {bit}, {desc})")
        else:
            print(f"    {fail(f'{prop_name}: NOT FOUND')}")
            all_ok = False

    result = {"status": "ok" if all_ok else "issues_found", "dataset": dataset}
    return result


# ── Step 4: Check ITERConfig creation ─────────────────────────────────
def check_iter_config(dataset, transformer: str = "bert-base-chinese") -> dict:
    """Create ITERConfig from dataset and verify all fields."""
    print(section("ITERConfig"))

    try:
        from iter import ITERConfig

        config = ITERConfig(
            transformer,
            transformer_config={"max_length": dataset.max_length},
            num_types=dataset.num_types,
            num_links=dataset.num_links,
            features=dataset.features,
            dataset=dataset.name,
            max_nest_depth=dataset.entity_nest_depth,
            dropout=0.3,
            activation_fn="relu",
            d_ff=0,
            entity_types=dataset.entity_types,
            link_types=dataset.link_types,
            fact_loss_weight=dataset.fact_loss_weight,
            multiscale_loss_weight=dataset.multiscale_loss_weight,
            use_gradient_isolation=dataset.use_gradient_isolation,
            ner_type_loss_weight=getattr(dataset, "ner_type_loss_weight", 0.0),
            ner_type_class_weights=getattr(dataset, "ner_type_class_weights", None),
            ner_span_margin_loss_weight=getattr(dataset, "ner_span_margin_loss_weight", 0.0),
            ner_span_margin=getattr(dataset, "ner_span_margin", 0.20),
            ner_span_margin_max_negatives=getattr(dataset, "ner_span_margin_max_negatives", 8),
            ner_span_margin_hard_negative_radius=getattr(dataset, "ner_span_margin_hard_negative_radius", 2),
            ner_boundary_consistency_loss_weight=getattr(
                dataset, "ner_boundary_consistency_loss_weight", 0.0
            ),
        )
        print(f"  {ok('ITERConfig created successfully')}")
    except Exception as e:
        print(f"  {fail(f'ITERConfig creation failed: {e}')}")
        return {"status": "fail", "error": str(e)}

    # ── Verify core fields ──
    checks = []
    checks.append(("features", config.features, dataset.features, "int"))
    checks.append(("fact_loss_weight", config.fact_loss_weight, dataset.fact_loss_weight, "float"))
    checks.append(("multiscale_loss_weight", config.multiscale_loss_weight, dataset.multiscale_loss_weight, "float"))
    checks.append(("use_gradient_isolation", config.use_gradient_isolation, dataset.use_gradient_isolation, "bool"))
    checks.append(("ner_type_loss_weight", config.ner_type_loss_weight, dataset.ner_type_loss_weight, "float"))
    checks.append(("ner_span_margin_loss_weight", config.ner_span_margin_loss_weight,
                   getattr(dataset, "ner_span_margin_loss_weight", 0.0), "float"))
    checks.append((
        "ner_boundary_consistency_loss_weight",
        config.ner_boundary_consistency_loss_weight,
        getattr(dataset, "ner_boundary_consistency_loss_weight", 0.0),
        "float",
    ))
    checks.append(("num_types", config.num_types, dataset.num_types, "int"))
    checks.append(("num_links", config.num_links, dataset.num_links, "int"))
    checks.append(("max_nest_depth", config.max_nest_depth, dataset.entity_nest_depth, "int"))

    print(f"\n  Field propagation CoNLL → ITERConfig:")
    all_match = True
    for field, config_val, dataset_val, _ in checks:
        if config_val == dataset_val:
            print(f"    {ok(f'{field}: {config_val}')}")
        else:
            print(f"    {fail(f'{field}: config={config_val} != dataset={dataset_val}')}")
            all_match = False

    # ── Verify feature bits ──
    print(f"\n  Feature bit checks on ITERConfig:")
    feature_checks = {
        "is_feature_fact_decomposition": 22,
        "is_feature_multiscale_geo": 23,
        "is_feature_gradient_isolation": 26,
    }
    for prop, bit in feature_checks.items():
        if hasattr(config, prop):
            val = getattr(config, prop)
            expected = (dataset.features & (1 << bit)) != 0
            if val == expected:
                status = ok("ON") if val else warn("OFF")
                print(f"    {prop}: {status}")
            else:
                print(f"    {fail(f'{prop}: {val} (expected {expected})')}")
                all_match = False
        else:
            print(f"    {fail(f'{prop}: NOT FOUND on ITERConfig')}")
            all_match = False

    # ── Warn about dead fields ──
    print(f"\n  {warn('Note: use_fact_decomposition, use_multiscale_geo fields in ITERConfig')}")
    print(f"    These constructor params are stored but NOT used by the model.")
    print(f"    The model relies on feature bits (is_feature_*). Ensure features bitmask is correct.")

    result = {"status": "ok" if all_match else "issues_found", "config": config}
    return result


# ── Step 5: Check Model initialization ────────────────────────────────
def check_model(config, print_model: bool = False) -> dict:
    """Initialize the model and verify module creation based on features."""
    print(section("Model Initialization"))

    try:
        from iter import ITER, ITERForRelationExtraction

        if config.is_feature_ner_only:
            model_cls = ITER
        else:
            model_cls = ITERForRelationExtraction

        model = model_cls(config)
        print(f"  {ok(f'Model created: {model_cls.__name__}')}")
    except Exception as e:
        print(f"  {fail(f'Model creation failed: {e}')}")
        import traceback
        traceback.print_exc()
        return {"status": "fail", "error": str(e)}

    # ── Check dual-branch modules ──
    print(f"\n  Module existence checks:")

    modules_to_check = {
        "multiscale_module": ("Multi-Scale Geo (bit 23)", 23),
        "fact_module": ("Fact Decomposition (bit 22)", 22),
    }

    for module_name, (desc, bit) in modules_to_check.items():
        exists = hasattr(model, module_name) and getattr(model, module_name) is not None
        expected = (config.features & (1 << bit)) != 0
        if exists == expected:
            status = ok("EXISTS") if exists else ok("absent (correct)")
            print(f"    {module_name}: {status}  ({desc})")
        else:
            if expected and not exists:
                print(f"    {fail(f'{module_name}: MISSING but bit {bit} is set')}")
            else:
                print(f"    {fail(f'{module_name}: EXISTS but bit {bit} is not set')}")

    # ── Check parallel branch infrastructure ──
    both_modules = (
        hasattr(model, 'multiscale_module') and model.multiscale_module is not None and
        hasattr(model, 'fact_module') and model.fact_module is not None
    )
    use_parallel = getattr(model, 'use_parallel_branches', False)

    print(f"\n  Dual-branch infrastructure:")
    print(f"    Both modules present: {ok('Yes') if both_modules else warn('No')}")
    print(f"    use_parallel_branches: {ok('True') if use_parallel else warn('False')}")

    if use_parallel:
        has_proj_ms = hasattr(model, 'proj_multiscale') and model.proj_multiscale is not None
        has_proj_fact = hasattr(model, 'proj_fact') and model.proj_fact is not None
        has_fusion_gate = hasattr(model, 'fusion_gate') and model.fusion_gate is not None
        has_fusion_ln = hasattr(model, 'fusion_layer_norm') and model.fusion_layer_norm is not None

        print(f"    proj_multiscale (task projection A): {ok('EXISTS') if has_proj_ms else fail('MISSING')}")
        print(f"    proj_fact (task projection B): {ok('EXISTS') if has_proj_fact else fail('MISSING')}")
        print(f"    fusion_gate (learned gate): {ok('EXISTS') if has_fusion_gate else fail('MISSING')}")
        print(f"    fusion_layer_norm: {ok('EXISTS') if has_fusion_ln else fail('MISSING')}")
    else:
        if both_modules:
            print(f"    {fail('INCONSISTENT: both modules present but use_parallel_branches=False')}")

    # ── Check loss weights in model ──
    print(f"\n  Loss weights in model:")
    loss_weights = {
        "fact_loss_weight": config.fact_loss_weight,
        "multiscale_loss_weight": config.multiscale_loss_weight,
    }
    for attr, expected_val in loss_weights.items():
        if hasattr(model, attr):
            val = getattr(model, attr)
            if val == expected_val:
                print(f"    {attr}: {val} {ok('')}")
            else:
                print(f"    {fail(f'{attr}: model={val} != config={expected_val}')}")
        else:
            print(f"    {fail(f'{attr}: NOT FOUND on model')}")

    # ── Check gradient isolation setting ──
    if hasattr(model, 'is_feature_gradient_isolation'):
        iso_enabled = model.is_feature_gradient_isolation
        status = ok("ON (gradients detached between branches)") if iso_enabled else warn("OFF")
        print(f"    is_feature_gradient_isolation: {status}")

    # ── Parameter count ──
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Parameter count:")
    print(f"    Total: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"    Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # ── Optional: print full model ──
    if print_model:
        print(f"\n{title('Full Model Architecture')}")
        print(model)

    return {"status": "ok", "model": model}


# ── Step 6: Forward pass smoke test ───────────────────────────────────
def check_forward_pass(model, config) -> dict:
    """Run a minimal forward pass to verify loss computation."""
    print(section("Forward Pass Smoke Test"))

    try:
        import torch

        # Create a minimal dummy input
        batch_size = 2
        seq_len = 32
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = model.to(device)
        model.eval()  # eval mode to avoid dropout randomness

        input_ids = torch.randint(100, 2000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        actions = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        lr_pair_flag = torch.zeros(batch_size, seq_len, 1, config.num_types, device=device)
        rr_pair_flag = torch.zeros(batch_size, seq_len, 1, config.num_links, device=device)

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                actions=actions,
                lr_pair_flag=lr_pair_flag,
                rr_pair_flag=rr_pair_flag,
            )

        print(f"  {ok('Forward pass succeeded')}")
        print(f"  Output keys/fields: {[k for k in dir(output) if not k.startswith('_')]}")

        # Check losses
        if hasattr(output, 'loss'):
            print(f"  Total loss: {output.loss.item():.6f}")

        loss_checks = {
            'fact_loss': 'Fact decomposition loss',
            'multiscale_loss': 'Multi-scale consistency loss',
        }

        for loss_attr, desc in loss_checks.items():
            if hasattr(output, loss_attr):
                val = getattr(output, loss_attr)
                if val is not None:
                    print(f"  {desc}: {val.item():.6f}")

        # Check that fact_loss and multiscale_loss are present when modules exist
        both_modules = getattr(model, 'use_parallel_branches', False)
        has_fact = hasattr(model, 'fact_module') and model.fact_module is not None

        if has_fact or both_modules:
            if hasattr(output, 'fact_loss') and output.fact_loss is not None:
                print(f"  {ok('fact_loss present in output (correct)')}")
            else:
                print(f"  {warn('fact_loss not in output (may be expected in eval mode)')}")

        return {"status": "ok"}

    except Exception as e:
        print(f"  {fail(f'Forward pass failed: {e}')}")
        import traceback
        traceback.print_exc()
        return {"status": "fail", "error": str(e)}


# ── Main ───────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="验证 ITER 配置链路：JSON → CoNLL → ITERConfig → Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python verify_config.py --dataset wenyanwen
  python verify_config.py --dataset wenyanwen_parallel_no_iso
  python verify_config.py --list
  python verify_config.py --dataset wenyanwen --print-model
  python verify_config.py --dataset wenyanwen --smoke-test
        """
    )
    parser.add_argument("--dataset", "-d", type=str, help="数据集配置名称")
    parser.add_argument("--transformer", "-t", type=str, default="bert-base-chinese",
                        help="Transformer 模型名称 (默认: bert-base-chinese)")
    parser.add_argument("--list", "-l", action="store_true", help="列出所有可用的数据集配置")
    parser.add_argument("--print-model", "-p", action="store_true", help="打印完整模型结构")
    parser.add_argument("--smoke-test", "-s", action="store_true", help="执行前向传播烟雾测试")
    parser.add_argument("--offline", action="store_true", help="离线模式 (不下载模型)")
    args = parser.parse_args()

    # ── List mode ──
    if args.list:
        print(title("Available Dataset Configurations"))
        configs = list_all_configs()
        for c in configs:
            print(f"  {c}")
        return

    if not args.dataset:
        parser.error("请提供 --dataset 参数，或使用 --list 查看可用配置")

    dataset_name = args.dataset

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    print(title(f"Configuration Verification: {dataset_name}"))
    print(f"  Transformer: {args.transformer}")
    print(f"  Offline mode: {args.offline}")

    all_ok = True

    # ── Run checks ──
    result = check_hparams(dataset_name)
    if result["status"] == "fail":
        all_ok = False

    result = check_conll(dataset_name)
    if result["status"] == "fail":
        print(f"\n{fail('CRITICAL: CoNLL loading failed, cannot continue')}")
        sys.exit(1)
    elif result["status"] == "issues_found":
        all_ok = False

    dataset = result["dataset"]

    result = check_iter_config(dataset, args.transformer)
    if result["status"] == "fail":
        print(f"\n{fail('CRITICAL: ITERConfig creation failed, cannot continue')}")
        sys.exit(1)
    elif result["status"] == "issues_found":
        all_ok = False

    config = result["config"]

    result = check_model(config, args.print_model)
    if result["status"] == "fail":
        print(f"\n{fail('CRITICAL: Model creation failed, cannot continue')}")
        sys.exit(1)

    model = result["model"]

    if args.smoke_test:
        result = check_forward_pass(model, config)
        if result["status"] == "fail":
            all_ok = False

    # ── Final summary ──
    print(title("Summary"))
    features_val = dataset.features

    summary_lines = []
    summary_lines.append(f"  Dataset: {dataset_name}")
    summary_lines.append(f"  Features bitmask: {features_val} (binary: {bin(features_val)})")

    # Dual-branch status
    has_fact = (features_val & (1 << 22)) != 0
    has_ms = (features_val & (1 << 23)) != 0
    has_iso = (features_val & (1 << 26)) != 0

    if has_fact and has_ms:
        branch_mode = "Parallel Dual-Branch (Branch A: Multi-Scale Geo, Branch B: Fact Decomposition)"
        if has_iso:
            branch_mode += " + Gradient Isolation"
    elif has_ms:
        branch_mode = "Sequential: Multi-Scale Geo only"
    elif has_fact:
        branch_mode = "Sequential: Fact Decomposition only"
    else:
        branch_mode = "Base ITER (no additional modules)"

    summary_lines.append(f"  Model mode: {branch_mode}")
    summary_lines.append(f"  fact_loss_weight: {dataset.fact_loss_weight}")
    summary_lines.append(f"  multiscale_loss_weight: {dataset.multiscale_loss_weight}")
    summary_lines.append(f"  All checks passed: {ok('Yes') if all_ok else fail('No — review issues above')}")

    for line in summary_lines:
        print(line)

    if all_ok:
        print(f"\n{GREEN}{BOLD}🎉 All configuration checks passed! Safe to train with:{RESET}")
        print(f"\n  python train.py --transformer {args.transformer} --dataset {dataset_name}")
    else:
        print(f"\n{RED}{BOLD}⚠️  Some checks failed. Review the issues above before training.{RESET}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
