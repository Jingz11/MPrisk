"""Render Figure S13 from the regional quality-restriction results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BLUE = "#377eb8"
GRAY = "#646e75"
GRID = "#e0e5e8"


def basin_panel(ax, basins, metric, title, show_names):
    positions = np.arange(len(basins))
    before = basins[f"{metric}_before"]
    after = basins[f"{metric}_after"]
    ax.set_title(title, loc="left", pad=10)
    ax.set(xlim=(-3, 103), ylim=(len(basins)-.5, -.5), xticks=np.arange(0, 101, 20), yticks=positions)
    ax.set_yticklabels(basins.label if show_names else [])
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=.65)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=3)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    ax.axhline(2.5, linewidth=.8, color=GRID)
    for i in positions[after.notna()]:
        ax.plot([before.iloc[i], after.iloc[i]], [i-.13, i+.13], color="#9ba9b2", linewidth=.9)
    ax.scatter(before, positions-.13, s=28, facecolor="white", edgecolor=BLUE, linewidth=.95, zorder=3)
    ax.scatter(after, positions+.13, s=22, marker="D", color=BLUE, zorder=3)


def render(data, output):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(12.1, 8.3), facecolor="white")
    handles = [
        Line2D([], [], linestyle="none", marker="o", markeredgecolor=BLUE,
               markerfacecolor="white", markersize=5.5, label="All Primary records"),
        Line2D([], [], linestyle="none", marker="D", color=BLUE, markersize=5,
               label="Quality-restricted records"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .905), ncol=2, frameon=False)

    basins = data.loc[data.scope.eq("Basin, river water")].sort_values(
        ["rank_before", "mean_Z_before"], ascending=[True, False]
    )
    basins = basins.assign(label=basins.region.str.replace(" River Basin", "", regex=False))
    assert len(basins) == 8 and basins.rank_before.notna().sum() == 3
    basin_panel(fig.add_axes((.168, .401, .277, .402)), basins, "mean_Z",
                r"$\bf{(a)}$  River-water mean Z (points)", True)
    basin_panel(fig.add_axes((.487, .401, .265, .402)), basins, "high_pct",
                r"$\bf{(b)}$  River-water G4–G5 share (%)", False)

    support = fig.add_axes((.775, .401, .203, .402))
    support.set(ylim=(len(basins)-.5, -.5), xlim=(0, 1))
    support.axis("off")
    support.set_title("Research groups / Rank", fontsize=9, pad=8)
    support.axhline(2.5, linewidth=.8, color=GRID)
    for i, row in enumerate(basins.itertuples(index=False)):
        support.text(.12, i, f"{row.studies_before} → {row.studies_after}", ha="center", va="center")
        if pd.notna(row.rank_after):
            label = f"{row.rank_before:.0f} → {row.rank_after:.0f}"
        else:
            label = "No eligible records" if row.n_after == 0 else "Insufficient support"
        support.text(.71, i, label, ha="center", va="center",
                     color="black" if pd.notna(row.rank_after) else GRAY, fontsize=9)

    coverage = fig.add_axes((.168, .158, .584, .138))
    scope_labels = {"Basin, river water": "River-water basins", "Province, river": "River provinces",
                    "Province, lake": "Lake provinces"}
    counts = data.groupby("scope").rank_after.agg(["size", "count"]).loc[list(scope_labels)]
    np.testing.assert_array_equal(counts["size"], [8, 21, 9])
    np.testing.assert_array_equal(counts["count"], [3, 3, 0])
    positions = np.arange(len(counts))
    shares = 100 * counts["count"] / counts["size"]
    coverage.barh(positions, 100, height=.52, color="#e8edf1")
    coverage.barh(positions, shares, height=.52, color=BLUE)
    coverage.set(xlim=(0, 100), ylim=(2.5, -.5), xticks=range(0, 101, 20), yticks=positions)
    coverage.set_yticklabels(list(scope_labels.values()))
    coverage.set_title(r"$\bf{(c)}$  Regions eligible for ranking in both cohorts", loc="left", pad=8)
    coverage.set_xlabel("Share of displayed regions (%)")
    coverage.tick_params(axis="y", length=0, pad=6)
    coverage.tick_params(axis="x", length=3)
    coverage.spines[["top", "right", "left"]].set_visible(False)
    coverage.spines["bottom"].set_color(GRAY)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"Figure_S13.{extension}", dpi=300, facecolor="white")
    plt.close(fig)
