import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "build_cross_window_dataset.py"
SPEC = importlib.util.spec_from_file_location("cross_window_builder", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_merge_bin_adds_provenance_and_preserves_relation_scope():
    first = {
        "tokens": ["a", "b"],
        "entities": [{"start": 0, "end": 1, "type": "x"}, {"start": 1, "end": 2, "type": "y"}],
        "relations": [{"head": 0, "tail": 1, "type": "r"}],
    }
    second = {
        "tokens": ["c", "d"],
        "entities": [{"start": 0, "end": 1, "type": "x"}, {"start": 1, "end": 2, "type": "y"}],
        "relations": [{"head": 0, "tail": 1, "type": "r"}],
    }
    document, manifest = MODULE.merge_bin(
        [("train", 3, first), ("train", 4, second)], "|"
    )
    assert len(manifest) == 2
    assert [entity["source_document_id"] for entity in document["entities"]] == [
        "train:3", "train:3", "train:4", "train:4"
    ]
    for relation in document["relations"]:
        assert document["entities"][relation["head"]]["source_document_id"] == document["entities"][relation["tail"]]["source_document_id"]
