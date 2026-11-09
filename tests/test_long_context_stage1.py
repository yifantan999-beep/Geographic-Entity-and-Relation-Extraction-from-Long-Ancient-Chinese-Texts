import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_module(relative_path: str):
    if relative_path == "iter/modeling_iter2.py":
        pytest.importorskip("torchcheck")
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_long_context_refiner_supports_padding_and_gradients():
    module = _load_module("iter/modules/long_context_ner.py")
    model = module.LongContextNERRefiner(16, chunk_size=4, num_heads=4)
    inputs = torch.randn(2, 11, 16, requires_grad=True)
    mask = torch.tensor([[1] * 11, [1] * 7 + [0] * 4], dtype=torch.bool)
    outputs = model(inputs, mask)
    assert outputs.shape == inputs.shape
    assert torch.isfinite(outputs).all()
    outputs[:, :7].sum().backward()
    assert inputs.grad is not None


def test_long_context_dataset_is_isolated_and_thresholded():
    data_dir = ROOT / "datasets" / "long_context"
    card = json.loads((data_dir / "dataset_card.json").read_text(encoding="utf-8"))
    assert card["selection"]["threshold_words"] == 128
    assert sum(split["documents"] for split in card["splits"].values()) == 400
    for split in ("train", "dev", "test"):
        rows = json.loads((data_dir / f"{split}.json").read_text(encoding="utf-8"))
        assert len(rows) == card["splits"][split]["documents"]
        assert all(len(row["tokens"]) >= 128 for row in rows)


def test_stage1_config_uses_only_long_context_refinement():
    config = json.loads(
        (ROOT / "cfg" / "wenyanwen" / "wenyanwen_long_context_512.json").read_text(encoding="utf-8")
    )
    dataset = config["dataset"]
    assert dataset["max_length"] == 512
    assert dataset["features"] == [13, 27]
    assert dataset["enable_windowing"] is True
    assert dataset["long_context_chunk_size"] == 32
    assert config["training"]["optimize_for"] == "ner"
    base = json.loads((ROOT / "cfg" / "wenyanwen.json").read_text(encoding="utf-8"))
    assert base["dataset"]["data_dir"] == "long_context"


def test_stage1_baseline_matches_hcbr_except_for_refiner():
    hcbr = json.loads(
        (ROOT / "cfg" / "wenyanwen" / "wenyanwen_long_context_512.json").read_text(encoding="utf-8")
    )
    baseline = json.loads(
        (ROOT / "cfg" / "wenyanwen" / "wenyanwen_long_context_512_baseline.json").read_text(encoding="utf-8")
    )

    assert baseline["dataset"]["features"] == [13]
    assert hcbr["dataset"]["features"] == [13, 27]
    for key in (
        "max_length",
        "enable_windowing",
        "window_overlap",
        "windowing_version",
        "fail_on_uncovered_entities",
        "long_context_chunk_size",
        "long_context_num_heads",
        "long_context_dropout",
    ):
        assert baseline["dataset"][key] == hcbr["dataset"][key]
    for key in ("num_epochs", "patience", "batch_size", "gradient_accumulation", "optimize_for", "metric_average"):
        assert baseline["training"][key] == hcbr["training"][key]


def test_type_aux_config_is_opt_in_and_matches_entity_cardinality():
    config = json.loads(
        (ROOT / "cfg" / "wenyanwen" / "wenyanwen_long_context_512_type_aux.json").read_text(encoding="utf-8")
    )
    assert config["dataset"]["ner_type_loss_weight"] == 0.2
    assert len(config["dataset"]["ner_type_class_weights"]) == 24
    assert all(weight > 0 for weight in config["dataset"]["ner_type_class_weights"])


def test_boundary_auxiliary_loss_has_finite_gradients_and_hard_negative_signal():
    module = _load_module("iter/modeling_iter2.py")
    model = torch.nn.Module()
    model.config = SimpleNamespace(
        ner_boundary_loss_weight=0.2,
        ner_boundary_hard_negative_weight=3.0,
        ner_boundary_positive_weight=1.5,
        ner_boundary_hard_negative_radius=1,
    )
    model.ner_boundary_start = torch.nn.Linear(8, 1)
    model.ner_boundary_end = torch.nn.Linear(8, 1)
    hidden = torch.randn(1, 10, 8, requires_grad=True)
    # One span [1, 4) and one span [6, 8); position 2 is an internal hard
    # negative and positions adjacent to boundaries are hard negatives.
    actions = torch.tensor([[0, 1, 0, 0, 2, 0, 1, 0, 2, 4]])
    attention_mask = torch.tensor([[1] * 9 + [0]], dtype=torch.bool)
    loss = module.ITER._ner_boundary_auxiliary_loss(
        model, hidden, actions, attention_mask
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_cross_window_v3_enables_boundary_objective():
    config = json.loads(
        (
            ROOT
            / "cfg"
            / "wenyanwen"
            / "wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_v3.json"
        ).read_text(encoding="utf-8")
    )
    dataset = config["dataset"]
    assert dataset["ner_boundary_loss_weight"] > 0
    assert dataset["ner_boundary_hard_negative_weight"] > 1
    assert dataset["ner_boundary_hard_negative_radius"] >= 1


def test_diagnostic_v7_enables_learned_span_gate_and_type_prototypes():
    config = json.loads(
        (
            ROOT
            / "cfg"
            / "wenyanwen"
            / "wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_diagnostic_easy_v7.json"
        ).read_text(encoding="utf-8")
    )
    dataset = config["dataset"]
    assert dataset["data_dir"] == "ind_distance_gt28_rebalanced_cross_window_diagnostic_easy"
    assert dataset["ner_span_learned_gate"] is True
    assert dataset["ner_type_prototype_loss_weight"] > 0
    assert dataset["ner_type_prototype_decode_weight"] > 0
    assert ["行政区划", "水系"] in dataset["ner_type_confusion_pairs"]


def test_type_prototype_scores_are_finite_and_differentiable():
    module = _load_module("iter/modeling_iter2.py")
    model = torch.nn.Module()
    model.config = SimpleNamespace(ner_type_prototype_temperature=0.15)
    model.ner_type_classifier = torch.nn.Sequential(
        torch.nn.Linear(8, 4),
        torch.nn.LayerNorm(4),
        torch.nn.GELU(),
        torch.nn.Dropout(0.0),
        torch.nn.Linear(4, 3),
    )
    model.ner_type_prototypes = torch.nn.Parameter(torch.randn(3, 4))
    pair_hidden = torch.randn(5, 8, requires_grad=True)
    scores = module.ITER._score_ner_type_prototypes(model, pair_hidden)
    assert scores.shape == (5, 3)
    assert torch.isfinite(scores).all()
    scores.sum().backward()
    assert pair_hidden.grad is not None
    assert model.ner_type_prototypes.grad is not None


def test_diagnostic_v8_reuses_easy_target_as_eval_with_explicit_leakage_label():
    config = json.loads(
        (
            ROOT
            / "cfg"
            / "wenyanwen"
            / "wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_diagnostic_easy_v8.json"
        ).read_text(encoding="utf-8")
    )
    directory = (
        ROOT
        / "datasets"
        / "ind_distance_gt28_rebalanced_cross_window_diagnostic_easy_target"
    )
    card = json.loads((directory / "dataset_card.json").read_text(encoding="utf-8"))
    assert config["dataset"]["data_dir"] == directory.name
    assert card["test_labels_used_for_model_selection"] is True
    assert (directory / "dev.json").read_bytes() == (directory / "test.json").read_bytes()


def test_cross_window_v4_enables_candidate_fusion_and_confusion_losses():
    config = json.loads(
        (
            ROOT
            / "cfg"
            / "wenyanwen"
            / "wenyanwen_long_context_512_type_aux_ind_distance_gt28_rebalanced_cross_window_v4.json"
        ).read_text(encoding="utf-8")
    )
    dataset = config["dataset"]
    assert dataset["ner_boundary_start_candidate_fusion"] is True
    assert dataset["ner_boundary_adjacent_negative_weight"] > dataset["ner_boundary_internal_negative_weight"]
    assert dataset["ner_boundary_focal_gamma"] > 0
    assert dataset["ner_type_focal_gamma"] > 0
    assert set(dataset["ner_type_focus_types"]) == {
        "学校", "行政区划", "水系", "城市内部区划"
    }
