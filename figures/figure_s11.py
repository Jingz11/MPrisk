from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

BANDS = ["#e8f1f5", "#e8f2ea", "#f5f2df", "#f9ecdf", "#f7e1df"]
STYLE = {
    "no_effect": ("o", "#5b7b93", "No detected change"),
    "development": ("P", "#5b7b93", "Development-source bioassay"),
    "limited": ("^", "#ad731e", "Taxon-specific change"),
    "impairment": ("^", "#b95046", "Adverse endpoint"),
    "guidance": ("s", "#217b81", "Management benchmark"),
    "ssd": ("D", "#6f4c8a", "SSD benchmark"),
}

def render(points, output):
    with plt.rc_context({"font.family": "Arial", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": .7}):
        fig, axes = plt.subplots(2, 1, figsize=(14.7, 8.6))
        fig.subplots_adjust(left=.065, right=.966, bottom=.223, top=.936, hspace=.49)
        for ax, medium in zip(axes, ("water", "sediment")):
            panel = points.loc[points.medium.eq(medium)].sort_values(["plot_row", "concentration"])
            is_reference = panel.dose_type.eq("external reference").to_numpy()
            x_ticks = np.arange(len(panel), dtype=float) + .60 * is_reference
            divider = (x_ticks[~is_reference][-1] + x_ticks[is_reference][0]) / 2
            for i, band in enumerate(BANDS):
                ax.axhspan(20 * i, 20 * (i + 1), color=band, lw=0)
                ax.text(1.008, 20 * i + 10, f"G{i + 1}", transform=ax.get_yaxis_transform(),
                        ha="left", va="center", color="black")
            ax.axvline(divider, color="#91999e", lw=.8, linestyle=(0, (3, 3)))
            for xs, label in ((x_ticks[~is_reference], "Bioassays"),
                              (x_ticks[is_reference], "Published benchmarks")):
                ax.text(xs.mean(), 1.03, label, transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", color="black", fontsize=10)
            for x, (_, row) in zip(x_ticks, panel.iterrows()):
                y = (row.z_low + row.z_high) / 2
                symbol, color, _ = STYLE[row.status]
                open_marker = row.status in {"no_effect", "development"}
                has_range = row.z_high - row.z_low > .015
                if has_range:
                    ax.errorbar(x, y, yerr=(row.z_high - row.z_low) / 2, color=color, fmt="none",
                                capsize=3.6, lw=1.35, zorder=3)
                ax.scatter(x, y, marker=symbol, s=37, edgecolor=color,
                           facecolor="white" if open_marker else color, linewidth=1,
                           zorder=4, clip_on=False)
            ax.set_xticks(x_ticks, panel.axis_label, fontsize=9)
            ax.tick_params(axis="x", length=3, pad=5)
            ax.tick_params(axis="y", length=3, pad=3)
            ax.set(xlim=(-.62, x_ticks[-1] + .66), ylim=(0, 101), yticks=np.arange(0, 101, 20),
                   ylabel="E–C–H score (Z)")
            ax.set_title("(a) Water" if medium == "water" else "(b) Sediment", loc="left", fontweight="bold", pad=11)
            ax.spines[["top", "right"]].set_visible(False)
        handles = []
        for key in ("no_effect", "development", "limited", "impairment", "guidance", "ssd"):
            marker, color, label = STYLE[key]
            handles.append(Line2D([], [], linestyle="none", marker=marker, color=color,
                                  markerfacecolor="white" if key in {"no_effect", "development"} else color, markersize=6, label=label))
        handles.append(Line2D([], [], color="#666666", lw=1.2, label="Composition scenario range"))
        axes[-1].set_xlabel("External exposure concentration and particle type", labelpad=10)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.50, .065), ncol=3,
                   frameon=False, columnspacing=2, handletextpad=.45, fontsize=9)
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(output / f"Figure_S11.{suffix}", dpi=450)
        plt.close(fig)
