#!/usr/bin/env python3
"""Render a LongGeoIE efficiency benchmark as CSV, Markdown, and LaTeX."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _millions(value: int) -> float:
    return value / 1_000_000.0


def _gibibytes(value: float) -> float:
    return value / (1024.0 ** 3)


def build_row(payload: dict) -> dict[str, str | int | float]:
    parameters = payload["parameters"]
    summary = payload["summary"]
    dataset = payload["dataset"]
    return {
        "Method": payload.get("method", "LongGeoIE"),
        "Stage-1 Params (M)": round(_millions(parameters["stage1"]["total"]), 3),
        "Stage-2 Params (M)": round(_millions(parameters["stage2"]["total"]), 3),
        "Total Params (M)": round(_millions(parameters["total_unique"]["total"]), 3),
        "Time Mean (s)": round(summary["pure_inference_seconds"]["mean"], 3),
        "Time Std (s)": round(summary["pure_inference_seconds"]["std"], 3),
        "ms/doc": round(summary["milliseconds_per_document"]["mean"], 3),
        "docs/s": round(summary["documents_per_second"]["mean"], 3),
        "Peak VRAM (GiB)": round(
            _gibibytes(summary["peak_cuda_allocated_bytes"]["max"]), 3
        ),
        "Documents": int(dataset["documents"]),
        "Windows": int(dataset["windows"]),
        "Batch Size": int(dataset["batch_size"]),
    }


def render_markdown(row: dict) -> str:
    columns = (
        "Method",
        "Stage-1 Params (M)",
        "Stage-2 Params (M)",
        "Total Params (M)",
        "Time Mean (s)",
        "Time Std (s)",
        "ms/doc",
        "docs/s",
        "Peak VRAM (GiB)",
    )
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" if index == 0 else "---:" for index in range(len(columns))) + " |"
    values = "| " + " | ".join(str(row[column]) for column in columns) + " |"
    return "\n".join((header, separator, values, ""))


def render_latex(row: dict, payload: dict) -> str:
    time_cell = f'{row["Time Mean (s)"]:.3f} $\\pm$ {row["Time Std (s)"]:.3f}'
    gpu_name = payload["hardware"].get("gpu_name") or payload["hardware"]["device"]
    return "\n".join(
        (
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Model size and inference efficiency on the IND test set.}",
            r"\label{tab:ind_efficiency}",
            r"\begin{tabular}{lrrrrrr}",
            r"\hline",
            r"Method & S1 Params & S2 Params & Total Params & Time (s) & ms/doc & Peak VRAM \\",
            r"\hline",
            (
                f'{row["Method"]} & {row["Stage-1 Params (M)"]:.3f}M & '
                f'{row["Stage-2 Params (M)"]:.3f}M & {row["Total Params (M)"]:.3f}M & '
                f'{time_cell} & {row["ms/doc"]:.3f} & {row["Peak VRAM (GiB)"]:.3f} GiB \\\\'
            ),
            r"\hline",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\linewidth}\footnotesize "
            f'Measured on {gpu_name} with batch size {row["Batch Size"]} over '
            f'{row["Documents"]} IND test documents ({row["Windows"]} windows). '
            f'The result is the mean of {payload["benchmark_repeats"] if "benchmark_repeats" in payload else len(payload["measured_runs"])} '
            f'measured runs after {payload["warmup_runs"]} warm-up run(s). '
            r"Checkpoint loading, dataset preparation, metric computation, and result serialization are excluded."
            r"\end{minipage}",
            r"\end{table}",
            "",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.benchmark.read_text(encoding="utf-8"))
    if not payload.get("measured_runs") or not payload.get("summary"):
        parser.error("Benchmark JSON contains no measured runs")
    output_dir = args.output_dir or args.benchmark.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    row = build_row(payload)

    with (output_dir / "benchmark_runs.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=payload["measured_runs"][0].keys())
        writer.writeheader()
        writer.writerows(payload["measured_runs"])

    with (output_dir / "benchmark_table.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    (output_dir / "benchmark_table.md").write_text(
        render_markdown(row), encoding="utf-8"
    )
    (output_dir / "benchmark_table.tex").write_text(
        render_latex(row, payload), encoding="utf-8"
    )
    print(f"Efficiency tables written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
