"""Restore the manuscript's continuous bar bases in Figure 6(c,d)."""

from pathlib import Path
import sys

from figures_6_7_shared import FEATURES, GOLD, GREEN, outline, panel, pd, plt, np, ribbon
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HABITATS = ("river", "lake")
COMP = ("E_ln_count", "C_ln_opportunity", "H_ln_multiplier")
METHODS = ("Direct RF", "Cascade RF")
COLORS = (GREEN, GOLD)
INK = ("#14786d", "#9b520c")


def cascade_scatter(fig, cell, data, performance, letter):
    gs = cell.subgridspec(2, 2, height_ratios=(.22, 1), width_ratios=(1, .22), hspace=.025, wspace=.023)
    mid = fig.add_subplot(gs[1, 0])
    top = fig.add_subplot(gs[0, 0], sharex=mid)
    right = fig.add_subplot(gs[1, 1], sharey=mid)
    for role, color, label in (("train", GREEN, "Training fit"),
                               ("test", GOLD, "Test")):
        q = data.loc[data.role.eq(role)]
        mid.scatter(q.Z, q.predicted, color=color, s=8.8, alpha=.56, linewidth=0,
                    label=label, rasterized=True)
        top.hist(q.Z, bins=np.arange(0, 104, 4), color=color, alpha=.67)
        right.hist(q.predicted, bins=np.arange(0, 104, 4), color=color, alpha=.67, orientation="horizontal")
    mid.plot([0, 100], [0, 100], color="#777", ls="--", lw=.65, label="1:1")
    mid.set(xlim=(0, 100), ylim=(0, 100), xlabel="Actual composite score Z", ylabel="Predicted composite score Z")
    mid.legend(loc="upper left", fontsize=7.8, title="Cascade RF", title_fontsize=7.8)
    for axis in (top, right):
        axis.axis("off")
    panel(top, letter, -.20, 1.07)


def test_bars(fig, cell, values, letter):
    gs = cell.subgridspec(2, 1, height_ratios=(1.12, 1), hspace=.055)
    hi = fig.add_subplot(gs[0])
    lo = fig.add_subplot(gs[1], sharex=hi)
    for i, (model, color) in enumerate(zip(METHODS, COLORS)):
        offset = (i - .5) * .25
        r = values.loc[model]
        error_x = np.array([1, 2]) + offset
        errors = [r.RMSE, r.MAE]
        lo.bar(np.r_[offset, error_x], [r.R2, *errors], .23, bottom=0, color=color)
        # The same error bars continue above the axis break. Both axes clip
        # the bars drawn from zero to the portion shown in their own range.
        hi.bar(error_x, errors, .23, bottom=0, color=color, label=model)
    hi.set(ylim=(8, 16), yticks=[10, 12, 14, 16])
    lo.set(ylim=(0, 1.04), yticks=[0, .5, 1])
    hi.set_ylabel("Error (RMSE / MAE)", fontsize=9)
    lo.set_ylabel("$R^2$")
    lo.set_xlim(-.55, 2.55)
    lo.set_xticks(range(3), ["Test $R^2$", "Test RMSE", "Test MAE"])
    hi.tick_params(axis="x", bottom=False, labelbottom=False)
    hi.spines["bottom"].set_visible(False)
    hi.legend(fontsize=7.3, loc="upper left", frameon=False, borderaxespad=.2)
    hi.set_title("Test metrics", fontsize=9, pad=4)
    for ax in (hi, lo):
        ax.grid(axis="y", alpha=.14)
        ax.set_axisbelow(True)
    for ax, y in ((hi, 0), (lo, 1)):
        for x in (0, 1):
            ax.plot([x-.012, x+.012], [y-.036, y+.036], transform=ax.transAxes,
                    clip_on=False, color="#777", lw=.62)
    panel(hi, letter, -.20, 1.31)


def direct_shapley(ax, values, habitat, letter, limit):
    index = values.set_index("predictor").reindex(list(FEATURES)).sort_values(
        "group_mean_absolute_shapley", ascending=False)
    assert index.group_mean_absolute_shapley.notna().all()
    i = HABITATS.index(habitat)
    y = np.arange(len(index))
    ax.barh(y, index.group_mean_absolute_shapley, color=COLORS[i], height=.72)
    ax.set(yticks=y, yticklabels=[FEATURES[key] for key in index.index], xlim=(0, limit),
           xlabel="Study-mean |test Shapley value| (Z points)")
    ax.tick_params(axis="y", labelsize=8.2)
    ax.invert_yaxis()
    ax.set_title(f"{habitat.title()} · Direct RF", color=INK[i], weight="bold", pad=4)
    ax.grid(axis="x", alpha=.15)
    panel(ax, letter, -.35, 1.10)


def component_heatmap(fig, ax, shap):
    keys = list(FEATURES)
    for i, habitat in enumerate(HABITATS):
        values = shap.loc[shap.habitat.eq(habitat)].pivot(
            index="output", columns="predictor", values="group_mean_absolute_shapley"
        ).reindex(index=COMP, columns=keys)
        assert values.notna().all().all() and (values.max(axis=1) > 0).all()
        values = values.div(values.max(axis=1), axis=0)
        cmap = LinearSegmentedColormap.from_list(habitat, ["white", COLORS[i]])
        im = ax.imshow(values, extent=[-.5, len(keys)-.5, 3*i+2.5, 3*i-.5], cmap=cmap,
                       vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0 + pos.width*.54*i, .075, pos.width*.46, .007])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, .5, 1])
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(habitat.title() + " · row-scaled importance", fontsize=7.1)
    ax.axhline(2.5, color="white", lw=3)
    ax.set(ylim=(5.5, -.5), yticks=range(6), yticklabels=list("ECH")*2)
    ax.set_xticks(range(len(keys)), [FEATURES[v] for v in keys], rotation=55, ha="right", fontsize=7.4)
    ax.set_title("Component Shapley values", fontsize=9, pad=5)
    panel(ax, "g", -.15)


def permutation_groups(ax, perm):
    cats = perm.category.drop_duplicates().tolist()
    colors = ("#23857c", "#a28233")
    markers = ("o", "s", "^")
    for k, cat in enumerate(cats):
        ax.axhspan(k-.47, k+.47, color=("#edf2fa", "#eef6ec", "#fcf1e8", "#f1ebf8")[k], alpha=.76, zorder=-2)
    for i, habitat in enumerate(HABITATS):
        for j, code in enumerate("ECH"):
            v = perm.loc[perm.habitat.eq(habitat) & perm.component.eq(code)].set_index("category").loc[cats]
            ax.errorbar(v.delta_r2, np.arange(len(cats)) + (3*i+j-2.5)*.133, xerr=v.repeat_sd,
                        fmt=markers[j], ms=3.8, color=colors[i], capsize=1.65, lw=.75)
    ax.axvline(0, color="#888", ls=":", lw=.7)
    ax.set(yticks=range(len(cats)), yticklabels=cats, ylim=(len(cats)-.51, -.49),
           xlabel="ΔR² (group permutation)")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Component group permutation", fontsize=9, pad=5)
    handles = [Patch(color=c, label=h.title()) for c, h in zip(colors, HABITATS)]
    handles += [Line2D([], [], marker=m, color="#555", ls="", label=c, markersize=4) for m, c in zip(markers, "ECH")]
    ax.legend(handles=handles, fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(.5, -.58), frameon=False)
    panel(ax, "h", -.37)


def draw(perf, scatter, shap, perm, output):
    assert len(perf) == 4 and len(perm) == 24 and set(scatter.seed) == {102}
    assert set(shap.loc[shap.output.eq("Z"), "seed"]) == {42}
    assert set(shap.loc[shap.output.isin(COMP), "seed"]) == {102}
    fig = plt.figure(figsize=(12.1, 14.2))
    gs = fig.add_gridspec(4, 2, left=.177, right=.963, top=.927, bottom=.150,
                         wspace=.37, hspace=.49, height_ratios=(1.17, .89, .92, .77))
    direct_shap = shap.loc[shap.output.eq("Z")]
    shap_limit = direct_shap.loc[direct_shap.predictor.isin(FEATURES), "group_mean_absolute_shapley"].max() * 1.08
    for j, habitat in enumerate(HABITATS):
        p = perf.loc[perf.habitat.eq(habitat)].set_index("model").loc[list(METHODS)]
        cascade_scatter(fig, gs[0, j], scatter.loc[scatter.habitat.eq(habitat)], p.loc["Cascade RF"], "ab"[j])
        test_bars(fig, gs[1, j], p, "cd"[j])
        direct_shapley(fig.add_subplot(gs[2, j]), direct_shap.loc[direct_shap.habitat.eq(habitat)],
                       habitat, "ef"[j], shap_limit)
        ribbon(fig, (.112, .565)[j], .951, .405, habitat.title(), ("#e0f3ef", "#fcf2e0")[j], INK[j])
        outline(fig, (.109, .562)[j], .516, .411, .435, ("#abd7cf", "#edc8a1")[j])
    component_heatmap(fig, fig.add_subplot(gs[3, 0]), shap.loc[shap.output.isin(COMP)])
    permutation_groups(fig.add_subplot(gs[3, 1]), perm)
    outline(fig, .098, .052, .875, .449, "#a0a0a0")
    for suffix in ("png", "pdf", "svg"):
        metadata = {"metadata": {"CreationDate": None, "ModDate": None}} if suffix == "pdf" else {}
        fig.savefig(output / f"Figure_6.{suffix}", dpi=350, facecolor="white", **metadata)
    plt.close(fig)
