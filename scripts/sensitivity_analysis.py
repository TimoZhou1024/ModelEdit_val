#!/usr/bin/env python
"""
Sensitivity analysis for ViT-tiny 3090 parameter search summaries.

For each dataset, this script:
1) Uses a pre-defined best exp_idx as anchor.
2) Fixes two parameters from the anchor and max_edits=3000.
3) Sweeps one factor among three points and plots two metrics:
   - edit_accuracy_after
   - test_accuracy_delta

Outputs:
- Filtered CSV tables per dataset/factor
- 3 figures per dataset (PNG + PDF), ready for paper insertion
- Aggregated summary tables across all datasets
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_DATASETS = [
    "pathmnist",
    "dermamnist",
    "bloodmnist",
    "organamnist",
    "retinamnist",
    "liver2a",
    "liver2s",
]


BEST_EXP_IDX: Dict[str, int] = {
    "pathmnist": 49,
    "dermamnist": 52,
    "bloodmnist": 19,
    "organamnist": 10,
    "retinamnist": 39,
    "liver2a": 27,
    "liver2s": 51,
}

# BEST_EXP_IDX: Dict[str, int] = {
#     "pathmnist": 46,
#     "dermamnist": 19,
#     "bloodmnist": 26,
#     "organamnist": 1,
#     "retinamnist": 21,
#     "liver2a": 3,
#     "liver2s": 6,
# }

FACTOR_CONFIG = {
    "num_edit_layers": {
        "fixed": ["v_grad_steps", "projection_samples"],
        "candidates": [1, 3, 5],
        "xlabel": "Number of Edited Layers",
    },
    "v_grad_steps": {
        "fixed": ["num_edit_layers", "projection_samples"],
        "candidates": [5, 15, 25],
        "xlabel": "V-Grad Steps",
    },
    "projection_samples": {
        "fixed": ["num_edit_layers", "v_grad_steps"],
        "candidates": [500, 1000, 1500],
        "xlabel": "Projection Samples",
    },
}


REQUIRED_COLUMNS = [
    "exp_idx",
    "num_edit_layers",
    "v_grad_steps",
    "projection_samples",
    "max_edits",
    "edit_accuracy_after",
    "test_accuracy_delta",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Sensitivity analysis from vit_tiny_3090_sensitivity CSV summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(project_root / "report" / "0214" / "vit_tiny_3090_sensitivity"),
        help="Directory containing param_search_summary_{dataset}.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "results" / "sensitivity_analysis"),
        help="Directory to save filtered tables and figures",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to process",
    )
    parser.add_argument(
        "--max-edits",
        type=int,
        default=3000,
        help="Fixed max_edits for all sensitivity plots (default: 3000)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any factor does not have exactly 3 data points",
    )
    return parser.parse_args()


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )


def validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")


def cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in REQUIRED_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def get_anchor_row(df: pd.DataFrame, dataset: str) -> pd.Series:
    if dataset not in BEST_EXP_IDX:
        raise KeyError(f"No best exp_idx configured for dataset: {dataset}")

    target_idx = BEST_EXP_IDX[dataset]
    anchor_rows = df[df["exp_idx"] == target_idx]
    if anchor_rows.empty:
        raise ValueError(
            f"Dataset {dataset}: configured exp_idx={target_idx} not found in CSV"
        )
    return anchor_rows.iloc[0]


def filter_for_factor(
    df: pd.DataFrame,
    anchor: pd.Series,
    factor: str,
    max_edits: int,
) -> pd.DataFrame:
    cfg = FACTOR_CONFIG[factor]
    fixed_a, fixed_b = cfg["fixed"]
    candidates = cfg["candidates"]

    sub = df[
        (df["max_edits"] == max_edits)
        & (df[fixed_a] == float(anchor[fixed_a]))
        & (df[fixed_b] == float(anchor[fixed_b]))
        & (df[factor].isin(candidates))
    ].copy()

    sub = sub.sort_values(by=factor)
    return sub


def plot_factor(
    df_plot: pd.DataFrame,
    dataset: str,
    factor: str,
    output_dir: Path,
    max_edits: int,
) -> Tuple[Path, Path]:
    cfg = FACTOR_CONFIG[factor]
    x = df_plot[factor].to_list()
    y_edit = df_plot["edit_accuracy_after"].to_list()
    y_delta = df_plot["test_accuracy_delta"].to_list()

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.plot(x, y_edit, marker="o", color="#1f77b4", label="edit_accuracy_after")
    ax.plot(x, y_delta, marker="s", color="#d62728", label="test_accuracy_delta")

    ax.set_xlabel(cfg["xlabel"])
    ax.set_ylabel("Metric Value")
    ax.set_title(f"{dataset} | Sensitivity on {factor} (max_edits={max_edits})")
    ax.set_xticks(cfg["candidates"])
    ax.grid(True, axis="both", linestyle="--", alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    png_path = output_dir / f"{dataset}_{factor}_sensitivity.png"
    pdf_path = output_dir / f"{dataset}_{factor}_sensitivity.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_plot_style()

    aggregate_rows: List[pd.DataFrame] = []
    anchor_summary: List[dict] = []
    summary_records: List[dict] = []

    for dataset in args.datasets:
        csv_path = input_dir / f"param_search_summary_{dataset}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        validate_columns(df, csv_path)
        df = cast_numeric(df)
        df = df.dropna(subset=REQUIRED_COLUMNS)

        anchor = get_anchor_row(df, dataset)
        anchor_summary.append(
            {
                "dataset": dataset,
                "best_exp_idx": int(anchor["exp_idx"]),
                "anchor_num_edit_layers": int(anchor["num_edit_layers"]),
                "anchor_v_grad_steps": int(anchor["v_grad_steps"]),
                "anchor_projection_samples": int(anchor["projection_samples"]),
                "anchor_max_edits": int(anchor["max_edits"]),
                "anchor_edit_accuracy_after": float(anchor["edit_accuracy_after"]),
                "anchor_test_accuracy_delta": float(anchor["test_accuracy_delta"]),
            }
        )

        dataset_out = output_dir / dataset
        dataset_out.mkdir(parents=True, exist_ok=True)

        for factor in FACTOR_CONFIG:
            filtered = filter_for_factor(
                df=df,
                anchor=anchor,
                factor=factor,
                max_edits=args.max_edits,
            )

            if args.strict and len(filtered) != 3:
                raise ValueError(
                    f"Dataset {dataset}, factor {factor}: expected 3 points, got {len(filtered)}"
                )

            filtered = filtered.copy()
            filtered["dataset"] = dataset
            filtered["factor"] = factor
            filtered["best_exp_idx"] = int(anchor["exp_idx"])
            preferred_order = ["dataset", "factor", "best_exp_idx"]
            remaining_cols = [c for c in filtered.columns if c not in preferred_order]
            filtered = filtered[preferred_order + remaining_cols]

            table_path = dataset_out / f"{dataset}_{factor}_filtered.csv"
            filtered.to_csv(table_path, index=False)
            aggregate_rows.append(filtered)

            png_path, pdf_path = plot_factor(
                df_plot=filtered,
                dataset=dataset,
                factor=factor,
                output_dir=dataset_out,
                max_edits=args.max_edits,
            )

            summary_records.append(
                {
                    "dataset": dataset,
                    "factor": factor,
                    "num_points": len(filtered),
                    "table_path": str(table_path),
                    "figure_png": str(png_path),
                    "figure_pdf": str(pdf_path),
                }
            )

    if aggregate_rows:
        pd.concat(aggregate_rows, ignore_index=True).to_csv(
            output_dir / "all_filtered_points.csv", index=False
        )

    pd.DataFrame(anchor_summary).to_csv(output_dir / "best_exp_idx_anchor_summary.csv", index=False)
    pd.DataFrame(summary_records).to_csv(output_dir / "artifact_index.csv", index=False)

    print(f"Sensitivity analysis finished. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
