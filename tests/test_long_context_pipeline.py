import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_function(relative_path, function_name):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


build_document_windows = _load_function("iter/datasets/windowing.py", "build_document_windows")
evaluate_documents = _load_function("iter/misc/document_metrics.py", "evaluate_documents")


class FakeEncoding(dict):
    pass


class FakeTokenizer:
    """One source token -> one model token, plus two special tokens."""

    def __call__(self, tokens, **kwargs):
        return FakeEncoding(input_ids=[101, *range(len(tokens)), 102])


def test_windowing_preserves_entities_and_reindexes_relations():
    item = {
        "tokens": list("abcdefghijklmnop"),
        "entities": [
            {"start": 2, "end": 5, "type": "A"},
            {"start": 6, "end": 8, "type": "B"},
            {"start": 13, "end": 15, "type": "C"},
        ],
        "relations": [
            {"head": 0, "tail": 1, "type": "near"},
            {"head": 0, "tail": 2, "type": "far"},
        ],
    }
    windows, manifest, coverage = build_document_windows(
        item, FakeTokenizer(), max_length=10, overlap_words=4
    )
    assert len(windows) >= 2
    assert coverage["uncovered_entity_ids"] == []
    assert 0 in coverage["covered_relation_ids"]
    assert 1 in coverage["uncovered_relation_ids"]
    for window in windows:
        for relation in window["relations"]:
            assert relation["head"] < len(window["entities"])
            assert relation["tail"] < len(window["entities"])
    assert all(row["tokenized_length"] <= 10 for row in manifest)


def test_window_boundaries_do_not_depend_on_gold_labels():
    tokens = list("abcdefghijklmnop")
    annotated = {
        "tokens": tokens,
        "entities": [{"start": 7, "end": 10, "type": "A"}],
        "relations": [],
    }
    unannotated = {"tokens": tokens, "entities": [], "relations": []}
    _, annotated_manifest, _ = build_document_windows(
        annotated, FakeTokenizer(), max_length=10, overlap_words=4
    )
    _, unannotated_manifest, _ = build_document_windows(
        unannotated, FakeTokenizer(), max_length=10, overlap_words=4
    )
    annotated_ranges = [(x["word_start"], x["word_end"]) for x in annotated_manifest]
    unannotated_ranges = [(x["word_start"], x["word_end"]) for x in unannotated_manifest]
    assert annotated_ranges == unannotated_ranges


def test_metrics_use_global_offsets_for_repeated_surface_forms():
    document = {
        "word_count": 300,
        "gold": {
            "entities": [
                {"start": 0, "end": 2, "type": "地名", "text": "同名"},
                {"start": 100, "end": 102, "type": "地名", "text": "同名"},
            ],
            "relations": [],
        },
        "prediction": {
            "entities": [{"start": 0, "end": 2, "type": "地名", "text": "同名"}],
            "relations": [],
        },
    }
    metrics = evaluate_documents([document], long_threshold=256)
    assert metrics["overall"]["ner_strict"]["tp"] == 1
    assert metrics["overall"]["ner_strict"]["fn"] == 1
    assert metrics["long_documents"]["documents"] == 1


def test_metrics_report_type_accuracy_and_confusion():
    document = {
        "word_count": 20,
        "gold": {
            "entities": [{"start": 1, "end": 3, "type": "河流"}],
            "relations": [],
        },
        "prediction": {
            "entities": [{"start": 1, "end": 3, "type": "道路"}],
            "relations": [],
        },
    }
    metrics = evaluate_documents([document])
    assert metrics["overall"]["ner_boundary"]["tp"] == 1
    assert metrics["overall"]["classification_accuracy"]["ner_type_on_matched_boundaries"] == 0
    assert metrics["type_confusions"]["entities"]["河流"]["道路"] == 1


def test_relation_metrics_separate_window_observable_gold():
    entities = [
        {"start": 0, "end": 2, "type": "地点"},
        {"start": 20, "end": 22, "type": "地点"},
        {"start": 200, "end": 202, "type": "地点"},
    ]
    document = {
        "word_count": 300,
        "gold": {
            "entities": entities,
            "relations": [
                {"head": 0, "tail": 1, "type": "相邻", "symmetric": True, "window_observable": True},
                {"head": 0, "tail": 2, "type": "通往", "symmetric": False, "window_observable": False},
            ],
        },
        "prediction": {
            "entities": entities,
            "relations": [
                {"head": 1, "tail": 0, "type": "相邻", "symmetric": True},
            ],
        },
    }
    metrics = evaluate_documents([document])
    assert metrics["overall"]["re_strict_end_to_end"]["tp"] == 1
    assert metrics["overall"]["re_strict_end_to_end"]["fn"] == 1
    assert metrics["window_relation_coverage"]["observable_recall"] == 1.0
    assert metrics["window_relation_coverage"]["unobservable_gold"] == 1
