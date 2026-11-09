import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_efficiency_benchmark.py"
SPEC = importlib.util.spec_from_file_location("render_efficiency_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _payload():
    return {
        "method": "LongGeoIE",
        "parameters": {
            "stage1": {"total": 100_000_000},
            "stage2": {"total": 5_000_000},
            "total_unique": {"total": 105_000_000},
        },
        "summary": {
            "pure_inference_seconds": {"mean": 10.0, "std": 0.25},
            "milliseconds_per_document": {"mean": 20.0},
            "documents_per_second": {"mean": 50.0},
            "peak_cuda_allocated_bytes": {"max": 4 * 1024**3},
        },
        "dataset": {"documents": 500, "windows": 600, "batch_size": 2},
        "hardware": {"device": "cuda:0", "gpu_name": "Test GPU"},
        "warmup_runs": 1,
        "measured_runs": [{"run": 1}],
    }


def test_build_row_formats_model_size_and_throughput():
    row = MODULE.build_row(_payload())
    assert row["Stage-1 Params (M)"] == 100.0
    assert row["Stage-2 Params (M)"] == 5.0
    assert row["Total Params (M)"] == 105.0
    assert row["Time Mean (s)"] == 10.0
    assert row["Peak VRAM (GiB)"] == 4.0


def test_rendered_tables_include_measurement_scope():
    payload = _payload()
    row = MODULE.build_row(payload)
    assert "LongGeoIE" in MODULE.render_markdown(row)
    latex = MODULE.render_latex(row, payload)
    assert "IND test set" in latex
    assert "Checkpoint loading" in latex
    assert "Test GPU" in latex
