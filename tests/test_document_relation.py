import importlib.util
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "iter" / "modules" / "document_relation.py"
SPEC = importlib.util.spec_from_file_location("document_relation_test_module", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
DocumentRelationConfig = MODULE.DocumentRelationConfig
DocumentRelationModel = MODULE.DocumentRelationModel
evidence_kl_loss = MODULE.evidence_kl_loss
non_negative_pu_loss = MODULE.non_negative_pu_loss
distance_weighted_positive_loss = MODULE.distance_weighted_positive_loss
relation_balanced_positive_loss = MODULE.relation_balanced_positive_loss
pair_type_compatibility_loss = MODULE.pair_type_compatibility_loss
long_distance_pair_ranking_loss = MODULE.long_distance_pair_ranking_loss
relation_positive_focal_loss = MODULE.relation_positive_focal_loss
relation_recall_margin_loss = MODULE.relation_recall_margin_loss
distance_expert_pu_loss = MODULE.distance_expert_pu_loss
silver_evidence_distribution = MODULE.silver_evidence_distribution
compute_relation_statistics = MODULE.compute_relation_statistics


def _model(fusion_mode="fixed"):
    config = DocumentRelationConfig(
        hidden_size=16,
        entity_types=["place", "road"],
        relation_types=["located_in", "connects"],
        type_embedding_size=4,
        distance_embedding_size=4,
        dropout=0.0,
        fusion_mode=fusion_mode,
        fusion_hidden_size=8,
        fusion_relation_embedding_size=3,
    )
    return DocumentRelationModel(config)


def test_document_relation_forward_and_evidence_distribution():
    model = _model()
    entity_repr = torch.randn(3, 16)
    type_probs = torch.tensor([[1.0, 0.0], [0.2, 0.8], [1.0, 0.0]])
    positions = torch.tensor([2.0, 20.0, 50.0])
    window_repr = torch.randn(2, 16)
    window_ranges = torch.tensor([[0.0, 32.0], [24.0, 64.0]])
    output = model(entity_repr, type_probs, positions, window_repr, window_ranges)

    assert output.pairs.shape == (6, 2)
    assert output.logits.shape == (6, 2)
    assert output.evidence_attention.shape == (6, 2)
    assert torch.allclose(output.evidence_attention.sum(dim=-1), torch.ones(6))

    target = silver_evidence_distribution(output.pairs, positions, window_ranges)
    positive_pairs = torch.tensor([True, False, True, False, False, False])
    loss = evidence_kl_loss(output.evidence_attention, target, positive_pairs)
    assert torch.isfinite(loss)


def test_non_negative_pu_loss_has_finite_gradients():
    logits = torch.randn(10, 2, requires_grad=True)
    targets = torch.zeros(10, 2)
    targets[0, 0] = 1
    targets[3, 1] = 1
    loss = non_negative_pu_loss(logits, targets, torch.tensor([0.1, 0.05]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_distance_weighted_positive_loss_emphasizes_long_pairs():
    logits = torch.tensor([[0.0], [-2.0]], requires_grad=True)
    targets = torch.ones_like(logits)
    pairs = torch.tensor([[0, 1], [1, 2]])
    positions = torch.tensor([0.0, 10.0, 200.0])
    loss = distance_weighted_positive_loss(
        logits,
        targets,
        positions,
        pairs,
        boundaries=[32, 64, 128],
        weights=[1.0, 1.25, 1.5, 2.0],
    )
    expected = (torch.nn.functional.softplus(torch.tensor(0.0)) + 2.0 * torch.nn.functional.softplus(torch.tensor(2.0))) / 3.0
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_relation_balanced_positive_loss_has_finite_gradients():
    logits = torch.zeros((4, 2), requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    loss = relation_balanced_positive_loss(
        logits, targets, torch.tensor([0.01, 0.0001]), max_weight=4.0
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_document_relation_save_and_load_roundtrip(tmp_path):
    model = _model().eval()
    directory = tmp_path / "document_relation"
    model.save_pretrained(directory)
    restored = DocumentRelationModel.from_pretrained(directory).eval()
    assert restored.config.entity_types == model.config.entity_types
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name])


def test_unseen_type_pairs_have_neutral_compatibility_bias():
    documents = [{
        "entities": [
            {"start": 0, "end": 1, "type": "place"},
            {"start": 2, "end": 3, "type": "place"},
        ],
        "relations": [{"head": 0, "tail": 1, "type": "located_in"}],
    }]
    compatibility, priors = compute_relation_statistics(
        documents,
        ["place", "road"],
        ["located_in", "connects"],
        set(),
    )
    assert torch.equal(compatibility[1, 1], torch.zeros(2))
    assert torch.all(priors > 0)


def test_fixed_fusion_path_remains_global_only():
    model = _model("fixed").eval()
    output = model(
        torch.randn(3, 16),
        torch.tensor([[1.0, 0.0], [0.2, 0.8], [1.0, 0.0]]),
        torch.tensor([2.0, 20.0, 50.0]),
        torch.randn(2, 16),
        torch.tensor([[0.0, 32.0], [24.0, 64.0]]),
    )
    assert output.fused_logits is None
    assert output.fused_probabilities is None
    assert output.fusion_gate is None


def test_learned_fusion_blends_available_logits_and_falls_back_to_global():
    model = _model("learned").eval()
    entity_repr = torch.randn(3, 16)
    type_probs = torch.tensor([[1.0, 0.0], [0.2, 0.8], [1.0, 0.0]])
    positions = torch.tensor([2.0, 20.0, 50.0])
    window_repr = torch.randn(2, 16)
    window_ranges = torch.tensor([[0.0, 32.0], [24.0, 64.0]])
    pairs = model.all_ordered_pairs(3, "cpu")
    local_probabilities = torch.full((pairs.size(0), 2), 0.8)
    local_available = torch.zeros_like(local_probabilities, dtype=torch.bool)
    local_available[0, 0] = True

    output = model(
        entity_repr,
        type_probs,
        positions,
        window_repr,
        window_ranges,
        pairs=pairs,
        local_probabilities=local_probabilities,
        local_available=local_available,
    )

    expected = 0.5 * torch.logit(local_probabilities[0, 0]) + 0.5 * output.logits[0, 0]
    assert torch.allclose(output.fused_logits[0, 0], expected)
    assert torch.allclose(output.fusion_gate[0, 0], torch.tensor(0.5))
    assert torch.allclose(output.fused_logits[~local_available], output.logits[~local_available])
    assert torch.equal(output.fusion_gate[~local_available], torch.zeros_like(output.fusion_gate[~local_available]))


def test_learned_fusion_save_and_load_roundtrip(tmp_path):
    model = _model("learned").eval()
    directory = tmp_path / "learned_fusion"
    model.save_pretrained(directory)
    restored = DocumentRelationModel.from_pretrained(directory).eval()
    assert restored.config.fusion_mode == "learned"
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name])


def test_long_distance_extensions_have_finite_outputs_and_losses():
    config = DocumentRelationConfig(
        hidden_size=16,
        entity_types=["学校", "园林", "行政区划", "水系", "城市内部区划", "街道"],
        relation_types=["包含", "隶属"],
        type_embedding_size=4,
        distance_embedding_size=4,
        dropout=0.0,
        fusion_mode="learned",
        distance_conditioned_fusion=True,
        fusion_long_distance_boundary=128,
        long_distance_head=True,
        long_distance_boundary=128,
        pair_type_compatibility=True,
    )
    model = DocumentRelationModel(config)
    model.set_statistics(torch.zeros(6, 6, 2), torch.tensor([0.1, 0.1]))
    entity_repr = torch.randn(4, 16)
    type_probs = torch.softmax(torch.randn(4, 6), dim=-1)
    positions = torch.tensor([0.0, 100.0, 220.0, 400.0])
    pairs = model.all_ordered_pairs(4, "cpu")
    output = model(
        entity_repr,
        type_probs,
        positions,
        torch.randn(2, 16),
        torch.tensor([[0.0, 256.0], [256.0, 512.0]]),
        pairs=pairs,
        local_probabilities=torch.full((pairs.size(0), 2), 0.7),
        local_available=torch.ones((pairs.size(0), 2), dtype=torch.bool),
    )
    targets = torch.zeros_like(output.logits)
    targets[0, 0] = 1.0
    type_loss = pair_type_compatibility_loss(
        output.pair_type_logits,
        targets,
        type_probs[pairs[:, 0]],
        type_probs[pairs[:, 1]],
        model.pair_type_focus_matrix,
        model.class_priors,
    )
    ranking_loss = long_distance_pair_ranking_loss(
        output.long_distance_logits, targets, positions, pairs
    )
    assert output.fusion_gate.shape == output.logits.shape
    assert torch.isfinite(type_loss)
    assert torch.isfinite(ranking_loss)


def test_v5_direction_position_and_positive_focal_extensions():
    config = DocumentRelationConfig(
        hidden_size=12,
        entity_types=["place", "road"],
        relation_types=["r1", "r2"],
        type_embedding_size=3,
        distance_embedding_size=3,
        dropout=0.0,
        fusion_mode="learned",
        use_direction_embedding=True,
        use_window_position_encoding=True,
    )
    model = DocumentRelationModel(config).eval()
    model.set_statistics(torch.zeros(2, 2, 2), torch.tensor([0.1, 0.1]))
    pairs = torch.tensor([[0, 1], [1, 0], [0, 2]])
    positions = torch.tensor([0.0, 20.0, 180.0])
    output = model(
        torch.randn(3, 12),
        torch.softmax(torch.randn(3, 2), dim=-1),
        positions,
        torch.randn(3, 12),
        torch.tensor([[0.0, 64.0], [64.0, 128.0], [128.0, 256.0]]),
        pairs=pairs,
        local_probabilities=torch.full((3, 2), 0.4),
        local_available=torch.ones((3, 2), dtype=torch.bool),
    )
    assert torch.isfinite(output.fused_logits).all()
    targets = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    loss = relation_positive_focal_loss(output.fused_logits, targets, positions, pairs)
    assert torch.isfinite(loss)


def test_distance_experts_use_smooth_normalized_gates_and_auxiliary_loss():
    config = DocumentRelationConfig(
        hidden_size=12,
        entity_types=["place", "road"],
        relation_types=["contains", "located_in"],
        type_embedding_size=3,
        distance_embedding_size=3,
        dropout=0.0,
        distance_experts=True,
        distance_expert_boundaries=[32, 64, 128],
        focus_relation_names=["contains"],
        focus_relation_positive_weight=1.5,
    )
    model = DocumentRelationModel(config)
    model.set_statistics(torch.zeros(2, 2, 2), torch.tensor([0.1, 0.1]))
    positions = torch.tensor([0.0, 20.0, 50.0, 100.0, 250.0])
    pairs = torch.tensor([[0, 1], [0, 2], [0, 3], [0, 4]])
    output = model(
        torch.randn(5, 12),
        torch.softmax(torch.randn(5, 2), dim=-1),
        positions,
        torch.randn(2, 12),
        torch.tensor([[0.0, 128.0], [128.0, 512.0]]),
        pairs=pairs,
    )
    assert output.distance_expert_logits.shape == (4, 4, 2)
    assert output.distance_expert_weights.shape == (4, 4)
    assert torch.allclose(
        output.distance_expert_weights.sum(dim=-1), torch.ones(4), atol=1e-6
    )
    targets = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    )
    loss = distance_expert_pu_loss(
        output.distance_expert_logits,
        targets,
        positions,
        pairs,
        [32, 64, 128],
        model.class_priors,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.distance_expert_classifiers[0][-1].weight.grad is not None


def test_distance_expert_active_bucket_preserves_other_bucket_logits():
    config = DocumentRelationConfig(
        hidden_size=8,
        entity_types=["place", "road"],
        relation_types=["contains"],
        type_embedding_size=2,
        distance_embedding_size=2,
        dropout=0.0,
        distance_experts=True,
        distance_expert_boundaries=[32, 64, 128],
        distance_expert_active_buckets=[2],
    )
    model = DocumentRelationModel(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.distance_expert_classifiers[2][-1].bias.fill_(2.0)
    entities = torch.randn(3, 8)
    types = torch.softmax(torch.randn(3, 2), dim=-1)
    positions = torch.tensor([0.0, 20.0, 100.0])
    windows = torch.randn(1, 8)
    ranges = torch.tensor([[0.0, 256.0]])
    short = model(
        entities, types, positions, windows, ranges, pairs=torch.tensor([[0, 1]])
    )
    middle = model(
        entities, types, positions, windows, ranges, pairs=torch.tensor([[0, 2]])
    )
    # The expert residual is exactly disabled outside bucket 2, while the
    # 64-127 pair receives a positive expert residual.
    assert torch.allclose(short.logits, torch.zeros_like(short.logits), atol=1e-6)
    assert middle.logits.item() > 0.0


def test_relation_focus_weights_increase_positive_focal_penalty():
    logits = torch.zeros(2, 2)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    positions = torch.tensor([0.0, 20.0, 50.0])
    pairs = torch.tensor([[0, 1], [1, 2]])
    plain = relation_positive_focal_loss(logits, targets, positions, pairs)
    focused = relation_positive_focal_loss(
        logits,
        targets,
        positions,
        pairs,
        relation_weights=torch.tensor([2.0, 1.0]),
    )
    assert focused > plain


def test_recall_margin_loss_has_finite_gradients_and_weights_short_pairs():
    logits = torch.zeros((2, 2), requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    positions = torch.tensor([0.0, 10.0, 200.0])
    pairs = torch.tensor([[0, 1], [1, 2]])
    loss = relation_recall_margin_loss(
        logits,
        targets,
        positions,
        pairs,
        threshold=0.24,
        margin=0.35,
        short_distance_boundary=64,
        short_distance_weight=2.0,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_recall_head_starts_as_zero_residual():
    config = DocumentRelationConfig(
        hidden_size=8,
        entity_types=["place"],
        relation_types=["r1", "r2"],
        type_embedding_size=2,
        distance_embedding_size=2,
        dropout=0.0,
        recall_head=True,
    )
    model = DocumentRelationModel(config).eval()
    output = model(
        torch.randn(2, 8),
        torch.ones(2, 1),
        torch.tensor([0.0, 20.0]),
        torch.randn(1, 8),
        torch.tensor([[0.0, 32.0]]),
        pairs=torch.tensor([[0, 1]]),
    )
    assert torch.equal(output.recall_head_logits, torch.zeros_like(output.recall_head_logits))


def test_recall_head_config_roundtrip(tmp_path):
    config = DocumentRelationConfig(
        hidden_size=8,
        entity_types=["place"],
        relation_types=["r1"],
        recall_head=True,
        recall_head_size=16,
        recall_head_weight=0.25,
    )
    model = DocumentRelationModel(config)
    model.save_pretrained(tmp_path)
    restored = DocumentRelationModel.from_pretrained(tmp_path)
    assert restored.config.recall_head is True
    assert restored.config.recall_head_size == 16


def test_legacy_checkpoint_without_relation_focus_buffer_is_loadable(tmp_path):
    model = _model()
    model.save_pretrained(tmp_path)
    weights_path = tmp_path / MODULE.WEIGHTS_NAME
    weights = {
        name: tensor.clone() for name, tensor in MODULE.load_file(weights_path).items()
    }
    weights.pop("relation_focus_weights")
    MODULE.save_file(weights, weights_path)
    restored = DocumentRelationModel.from_pretrained(tmp_path)
    assert torch.equal(
        restored.relation_focus_weights,
        torch.ones_like(restored.relation_focus_weights),
    )


def test_v8_re_config_targets_only_mid_distance_and_common_relations():
    path = (
        Path(__file__).resolve().parents[1]
        / "cfg"
        / "wenyanwen"
        / "wenyanwen_document_relation_distance_experts_v8.json"
    )
    import json

    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["distance_expert_active_buckets"] == [2]
    assert config["distance_expert_weight"] < 0.5
    assert config["focus_relation_names"] == ["包含", "隶属"]
