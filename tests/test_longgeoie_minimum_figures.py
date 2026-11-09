import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_longgeoie_minimum_figures.py"
SPEC = importlib.util.spec_from_file_location("build_longgeoie_minimum_figures", SCRIPT)
FIGURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIGURES)


def _document(*, fusion_trace=None):
    entities = [
        {"start": 0, "end": 2, "type": "A"},
        {"start": 130, "end": 132, "type": "B"},
        {"start": 300, "end": 302, "type": "C"},
    ]
    prediction = {
        "entities": entities,
        "relations": [
            {
                "head": 0,
                "tail": 1,
                "type": "near",
                "symmetric": False,
                "fusion_gate": 0.25,
                "local_probability": 0.8,
            }
        ],
    }
    if fusion_trace is not None:
        prediction["fusion_trace"] = fusion_trace
    return {
        "document_id": "doc-1",
        "word_count": 400,
        "windows": [
            {"word_start": 0, "word_end": 200},
            {"word_start": 180, "word_end": 400},
        ],
        "gold": {
            "entities": entities,
            "relations": [
                {
                    "head": 0,
                    "tail": 1,
                    "type": "near",
                    "symmetric": False,
                    "window_observable": True,
                },
                {
                    "head": 0,
                    "tail": 2,
                    "type": "far",
                    "symmetric": False,
                    "window_observable": False,
                },
            ],
        },
        "prediction": prediction,
    }


def test_legacy_gate_fallback_converts_local_weight_to_global_weight():
    rows = FIGURES.build_gate_trace("LongGeoIE", [_document()])
    assert len(rows) == 1
    assert rows[0]["local_gate_weight"] == 0.25
    assert rows[0]["global_gate_weight"] == 0.75
    assert rows[0]["scope"] == "recalled_gold_relations_fallback"


def test_full_trace_keeps_missing_endpoints_and_exact_gate_values():
    trace = [
        {
            "gold_relation_id": 0,
            "relation_type": "near",
            "distance": 130,
            "local_gate_weight": 0.25,
            "global_gate_weight": 0.75,
            "endpoint_match": True,
            "local_available": True,
            "is_recalled": True,
            "same_window": True,
        },
        {
            "gold_relation_id": 1,
            "relation_type": "far",
            "distance": 300,
            "local_gate_weight": None,
            "global_gate_weight": None,
            "endpoint_match": False,
            "local_available": False,
            "is_recalled": False,
            "same_window": False,
        },
    ]
    rows = FIGURES.build_gate_trace("LongGeoIE", [_document(fusion_trace=trace)])
    assert len(rows) == 2
    assert rows[0]["bucket"] == "128-191"
    assert rows[0]["global_gate_weight"] == 0.75
    assert rows[1]["bucket"] == ">=256"
    assert rows[1]["global_gate_weight"] is None
    assert rows[1]["endpoint_match"] is False


def test_window_analysis_uses_window_observable_not_distance():
    rows = FIGURES.build_window_rows({"LongGeoIE": [_document()]})
    by_category = {row["window_category"]: row for row in rows}
    assert by_category["Same-window"]["support"] == 1
    assert by_category["Same-window"]["tp"] == 1
    assert by_category["Same-window"]["f1"] == 1.0
    assert by_category["Cross-window"]["support"] == 1
    assert by_category["Cross-window"]["tp"] == 0
    assert by_category["Cross-window"]["f1"] == 0.0
