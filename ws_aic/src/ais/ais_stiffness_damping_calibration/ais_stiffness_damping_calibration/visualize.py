#!/usr/bin/env python3
"""Visualize stiffness/damping sweep CSV results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_WEIGHTS = (
    ("final_error_norm_mm", "final error", 0.30),
    ("overshoot_norm_mm", "overshoot", 0.35),
    ("tail_xy_peak_to_peak_mm", "tail xy", 0.20),
    ("z_peak_mmps", "z peak", 0.15),
)


def candidate_output_roots() -> list[Path]:
    package_root = Path(__file__).resolve().parents[1]
    return [
        Path("ais/ais_stiffness_damping_calibration/outputs"),
        Path("ws_aic/src/ais/ais_stiffness_damping_calibration/outputs"),
        package_root / "outputs",
    ]


def find_latest_csv() -> Path:
    candidates: list[Path] = []
    for root in candidate_output_roots():
        if root.exists():
            candidates.extend(root.glob("*/stiffness_damping_sweep.csv"))
    if not candidates:
        roots = ", ".join(str(root) for root in candidate_output_roots())
        raise FileNotFoundError(f"No sweep CSV found under: {roots}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def option_label(row: pd.Series) -> str:
    return (
        f"k=({row.stiffness_x:.0f},{row.stiffness_y:.0f},{row.stiffness_z:.0f}) "
        f"d=({row.damping_x:.0f},{row.damping_y:.0f},{row.damping_z:.0f})"
    )


def compact_option_label(option: str) -> str:
    return option.replace("k=(", "k").replace(") d=(", " d").replace(")", "")


def delta_label(row: pd.Series) -> str:
    values_mm = np.array(
        [row.requested_delta_x_m, row.requested_delta_y_m, row.requested_delta_z_m],
        dtype=float,
    ) * 1000.0
    nonzero = np.flatnonzero(np.abs(values_mm) > 1e-6)
    if len(nonzero) == 1:
        axis = ("x", "y", "z")[int(nonzero[0])]
        return f"{axis}={values_mm[nonzero[0]]:.0f}mm"
    return f"({values_mm[0]:.0f},{values_mm[1]:.0f},{values_mm[2]:.0f})mm"


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "delta_index",
        "requested_delta_norm_m",
        "requested_delta_x_m",
        "requested_delta_y_m",
        "requested_delta_z_m",
        "stiffness_x",
        "stiffness_y",
        "stiffness_z",
        "damping_x",
        "damping_y",
        "damping_z",
        "final_error_norm_mm",
        "tail_xy_peak_to_peak_mm",
        "peak_abs_velocity_z_mps",
        "max_overshoot_x_mm",
        "max_overshoot_y_mm",
        "max_overshoot_z_mm",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    df = df.copy()
    df["option"] = df.apply(option_label, axis=1)
    df["option_compact"] = df["option"].map(compact_option_label)
    df["delta_label"] = df.apply(delta_label, axis=1)
    df["distance_mm"] = df["requested_delta_norm_m"] * 1000.0
    df["overshoot_norm_mm"] = np.sqrt(
        df[
            ["max_overshoot_x_mm", "max_overshoot_y_mm", "max_overshoot_z_mm"]
        ].pow(2).sum(axis=1)
    )
    df["z_peak_mmps"] = df["peak_abs_velocity_z_mps"] * 1000.0
    return df


def score_results(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked_parts = []
    for _, group in df.groupby("delta_index", sort=False):
        ranked = group[
            ["option", "option_compact", "delta_index", "delta_label", "distance_mm"]
        ].copy()
        weighted_score = np.zeros(len(group), dtype=float)
        for metric, _, weight in METRIC_WEIGHTS:
            rank = group[metric].rank(method="average", ascending=True, pct=True)
            ranked[f"{metric}_rank_pct"] = rank.to_numpy()
            weighted_score += weight * rank.to_numpy()
        ranked["score"] = weighted_score
        ranked_parts.append(ranked)

    score_by_delta = pd.concat(ranked_parts, ignore_index=True)
    mean_metrics = df.groupby("option", sort=False).agg(
        option_compact=("option_compact", "first"),
        mean_final_error_mm=("final_error_norm_mm", "mean"),
        mean_overshoot_mm=("overshoot_norm_mm", "mean"),
        mean_tail_xy_mm=("tail_xy_peak_to_peak_mm", "mean"),
        mean_z_peak_mmps=("z_peak_mmps", "mean"),
        max_overshoot_mm=("overshoot_norm_mm", "max"),
    )
    score_summary = score_by_delta.groupby("option", sort=False).agg(
        mean_score=("score", "mean"),
        worst_score=("score", "max"),
        score_std=("score", "std"),
    )
    summary = score_summary.join(mean_metrics).sort_values(
        ["mean_score", "worst_score", "mean_final_error_mm"]
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary, score_by_delta


def ordered_delta_columns(score_by_delta: pd.DataFrame) -> list[str]:
    ordered = (
        score_by_delta[["delta_index", "delta_label"]]
        .drop_duplicates()
        .sort_values("delta_index")
    )
    return ordered["delta_label"].tolist()


def plot_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    top_options: Sequence[str],
    metric: str,
    ylabel: str,
) -> None:
    for option in top_options:
        part = df[df["option"] == option].sort_values("delta_index")
        ax.plot(
            part["distance_mm"],
            part[metric],
            marker="o",
            linewidth=2,
            label=compact_option_label(option),
        )
    ax.set_xlabel("requested distance (mm)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def write_plot(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    score_by_delta: pd.DataFrame,
    output_path: Path,
    top_n: int,
) -> None:
    ordered_options = summary.index.tolist()
    ordered_compact = [compact_option_label(option) for option in ordered_options]
    ordered_deltas = ordered_delta_columns(score_by_delta)
    heatmap = (
        score_by_delta.pivot_table(index="option", columns="delta_label", values="score")
        .reindex(index=ordered_options, columns=ordered_deltas)
    )

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.0, 1.0])

    ax_heatmap = fig.add_subplot(grid[0, 0])
    image = ax_heatmap.imshow(heatmap.to_numpy(), aspect="auto", cmap="viridis_r")
    ax_heatmap.set_title("Weighted rank score by distance (lower is better)")
    ax_heatmap.set_xticks(np.arange(len(ordered_deltas)), labels=ordered_deltas)
    ax_heatmap.set_yticks(np.arange(len(ordered_compact)), labels=ordered_compact)
    ax_heatmap.tick_params(axis="x", rotation=35)
    for y_index in range(heatmap.shape[0]):
        for x_index in range(heatmap.shape[1]):
            value = heatmap.iat[y_index, x_index]
            ax_heatmap.text(
                x_index,
                y_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=9,
            )
    fig.colorbar(image, ax=ax_heatmap, label="score")

    ax_bar = fig.add_subplot(grid[0, 1])
    bar_data = summary.sort_values("mean_score", ascending=True)
    colors = ["#2a9d8f"] + ["#8d99ae"] * (len(bar_data) - 1)
    ax_bar.barh(bar_data["option_compact"], bar_data["mean_score"], color=colors)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("mean weighted rank score")
    ax_bar.set_title("Overall ranking")
    ax_bar.grid(True, axis="x", alpha=0.25)

    top_options = ordered_options[: max(1, min(top_n, len(ordered_options)))]
    ax_final = fig.add_subplot(grid[1, 0])
    plot_metric(ax_final, df, top_options, "final_error_norm_mm", "final error (mm)")
    ax_final.set_title("Final target error")

    ax_over = fig.add_subplot(grid[1, 1])
    plot_metric(ax_over, df, top_options, "overshoot_norm_mm", "overshoot norm (mm)")
    ax_over.set_title("Overshoot")

    ax_tail = fig.add_subplot(grid[2, 0])
    plot_metric(ax_tail, df, top_options, "tail_xy_peak_to_peak_mm", "tail xy p2p (mm)")
    ax_tail.set_title("Tail oscillation")

    ax_z = fig.add_subplot(grid[2, 1])
    plot_metric(ax_z, df, top_options, "z_peak_mmps", "z peak velocity (mm/s)")
    ax_z.set_title("Z motion coupling")
    ax_z.legend(loc="best", fontsize=9)

    fig.suptitle("AIC Cartesian stiffness/damping sweep", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> str:
    table = df.loc[:, columns].copy()
    return table.to_markdown(index=False, floatfmt=".3f")


def write_summary(
    csv_path: Path,
    output_dir: Path,
    summary: pd.DataFrame,
    score_by_delta: pd.DataFrame,
    plot_path: Path,
) -> Path:
    best_option = str(summary.index[0])
    best_rows = score_by_delta[score_by_delta["option"] == best_option].sort_values(
        "delta_index"
    )
    summary_for_md = summary.reset_index().rename(columns={"index": "option"})
    summary_for_md = summary_for_md[
        [
            "rank",
            "option",
            "mean_score",
            "worst_score",
            "mean_final_error_mm",
            "mean_overshoot_mm",
            "mean_tail_xy_mm",
            "mean_z_peak_mmps",
        ]
    ]

    best_by_delta = best_rows[
        ["delta_label", "score"]
    ].merge(
        summary_for_md[summary_for_md["option"] == best_option][
            ["option", "mean_score"]
        ],
        how="cross",
    )

    path = output_dir / "summary.md"
    weights_text = ", ".join(
        f"{label} {weight:.0%}" for _, label, weight in METRIC_WEIGHTS
    )
    path.write_text(
        "\n".join(
            [
                "# Stiffness/Damping Sweep Summary",
                "",
                f"Input CSV: `{csv_path}`",
                f"Overview plot: `{plot_path}`",
                "",
                f"Scoring: each metric is ranked within each requested delta; lower is better. Weights: {weights_text}.",
                "",
                f"Best overall: `{best_option}`",
                "",
                "## Ranked Options",
                "",
                markdown_table(summary_for_md, summary_for_md.columns),
                "",
                "## Best Option Score By Delta",
                "",
                markdown_table(best_by_delta, best_by_delta.columns),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create plots and ranking tables from a stiffness/damping sweep CSV."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        help="Path to stiffness_damping_sweep.csv. Defaults to the latest sweep output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for sweep_overview.png, ranked_options.csv, and summary.md.",
    )
    parser.add_argument("--top-n", type=int, default=3)
    return parser.parse_args()


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    csv_path = args.csv if args.csv is not None else find_latest_csv()
    csv_path = csv_path.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else csv_path.parent

    df = load_results(csv_path)
    summary, score_by_delta = score_results(df)

    ranked_csv = output_dir / "ranked_options.csv"
    summary.to_csv(ranked_csv)

    score_csv = output_dir / "score_by_delta.csv"
    score_by_delta.to_csv(score_csv, index=False)

    plot_path = output_dir / "sweep_overview.png"
    write_plot(df, summary, score_by_delta, plot_path, args.top_n)
    summary_path = write_summary(csv_path, output_dir, summary, score_by_delta, plot_path)

    best = summary.iloc[0]
    print(f"Best overall: {summary.index[0]}")
    print(
        "mean_score={:.3f}, final={:.3f}mm, overshoot={:.3f}mm, "
        "tail_xy={:.3f}mm, z_peak={:.3f}mm/s".format(
            best["mean_score"],
            best["mean_final_error_mm"],
            best["mean_overshoot_mm"],
            best["mean_tail_xy_mm"],
            best["mean_z_peak_mmps"],
        )
    )
    print(f"Wrote plot: {plot_path}")
    print(f"Wrote ranked CSV: {ranked_csv}")
    print(f"Wrote summary: {summary_path}")
    return plot_path, ranked_csv, summary_path


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
