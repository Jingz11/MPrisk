"""Versioned Figure 5 with split-to-split variation in panel (a)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
ORDER = [
    ('agri_film_prov_t', 'AgriFilm'), ('IPME', 'IPME'), ('fertilizer_prov_10kt', 'Fertilizer'),
    ('PopDensity_urban_construction', 'PopDensity'), ('agri_output_prov_100mCNY', 'AgriOutput'),
    ('temp_L8_climatology_degC', 'Temp (L8)'), ('Waste_Treat', 'Waste_Treat'), ('GRP_pc', 'GRP_pc'),
    ('elevation_2p5min_m', 'ALT'), ('WW_Discharge', 'WW_Discharge'), ('VIIRS_valid_3km_nW_cm2_sr', 'VIIRS'),
]
MODELS = ['ElasticNet', 'Ridge', 'HGB', 'GB', 'ExtraTrees', 'RF']
DISPLAY = {'HGB': 'HistGB', 'GB': 'GradientBoosting', 'RF': 'RandomForest'}
PALE = ['#a8cee1', '#fcc0a7']
DARK = ['#3583ba', '#d85859']

plt.rcParams.update({"font.family": "Arial", "font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})


def performance_panels(fig, statistics):
    chart = fig.add_axes([.191, .739, .773, .208])
    xpos = np.arange(len(MODELS))
    legend = []
    display = [DISPLAY.get(m, m) for m in MODELS]
    for i, habitat in enumerate(("river", "lake")):
        t = statistics.query("habitat == @habitat").set_index("model").loc[MODELS]
        x = xpos + (i - .5) * .33
        chart.bar(x, t.CV_pooled_R2_mean, .31, color=PALE[i], zorder=2)
        chart.errorbar(x, t.CV_pooled_R2_mean, yerr=t.CV_pooled_R2_SD, fmt="none",
                       ecolor=DARK[i], alpha=.60, capsize=2.7, lw=1, zorder=3)
        chart.errorbar(x, t.test_R2_mean, yerr=t.test_R2_SD, fmt="x", color=DARK[i],
                       ms=6.2, mew=1.5, capsize=3.1, lw=1.15, zorder=4)
        legend.extend([Patch(color=PALE[i], label=f"{habitat.title()} CV mean ± SD"),
                       Line2D([0], [0], color=DARK[i], marker="x", ls="",
                              label=f"{habitat.title()} test mean ± SD")])
    floor = min(0, min((statistics[f"{key}_mean"] - statistics[f"{key}_SD"]).min()
                       for key in ("CV_pooled_R2", "test_R2")))
    chart.set_xticks(xpos, display, rotation=35, ha="right", rotation_mode="anchor")
    chart.set(ylim=(floor - .025 if floor < 0 else 0, 1.03), ylabel="Coefficient of determination ($R^2$)")
    chart.grid(axis="y", color="#ececec", linestyle="--", linewidth=.6)
    chart.set_axisbelow(True)
    chart.legend(handles=legend, ncol=2, loc="upper left", fontsize=8.2, framealpha=.96,
                 handletextpad=.55, columnspacing=1.1)


def importance_panels(fig, importance):
    width = .368
    pi_lo = min(0., (importance.PI_mean - importance.PI_SD).min())
    pi_hi = max(.05, (importance.PI_mean + importance.PI_SD).max())
    pad = (pi_hi - pi_lo) * .07
    mdi_hi = importance.MDI.max() * 1.13
    for i, (habitat, left) in enumerate(zip(("river", "lake"), (.191, .596))):
        q = importance.query("habitat == @habitat").set_index("variable").loc[[v for v, _ in ORDER]]
        y = np.arange(len(q))
        b = fig.add_axes([left, .365, width, .234])
        c = fig.add_axes([left, .070, width, .224])
        b.set_title(habitat.title(), fontsize=14)
        b.barh(y, q.PI_mean, color=PALE[i], height=.72)
        b.errorbar(q.PI_mean, y, xerr=q.PI_SD, fmt="none", ecolor=DARK[i], capsize=3, lw=.8)
        b.axvline(0, color="#808080", linewidth=.65)
        b.set(xlim=(pi_lo - pad if pi_lo < 0 else 0, pi_hi + pad),
              xlabel="Permutation importance (ΔR²)")
        c.scatter(q.MDI, y, s=53 + 660 * q.MDI, alpha=.63, color=DARK[i], linewidth=0)
        c.set(xlim=(0, mdi_hi), xlabel="MDI importance")
        for ax in (b, c):
            ax.set_yticks(y, q.label if i == 0 else [""] * len(q))
            ax.set_ylim(len(q) - .35, -.65)
            ax.grid(axis="x", color="#ededed", linestyle=":", linewidth=.65)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=11)
            ax.xaxis.label.set_size(13)
            ax.xaxis.label.set_weight("bold")
            for tick in ax.get_yticklabels():
                tick.set_weight("bold")


def render(stats, importance, output):
    fig = plt.figure(figsize=(12.6, 14.2), facecolor="white")
    performance_panels(fig, stats)
    importance_panels(fig, importance)
    for letter, y in [("a", .972), ("b", .624), ("c", .312)]:
        fig.axes[0].annotate(f"({letter})", (.034, y), xycoords=fig.transFigure, fontsize=16, va="center")
    for extension in ("png", "pdf", "svg"):
        options = {"metadata": {"CreationDate": None, "ModDate": None}} if extension == "pdf" else {}
        fig.savefig(output / f"Figure_5.{extension}", dpi=400, facecolor="white", **options)
    plt.close(fig)
