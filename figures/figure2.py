"""Draw the final water PLS-DA panel in the abundance and composition plate."""
import io
import re
from lxml import etree
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image, ImageChops

NS = "{http://www.w3.org/2000/svg}"
COLORS = {"river": "#309b70", "lake": "#e9a23b"}

def replace_vector_panel(figure, base, output, panel_id="Water_PLSDA"):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="svg", transparent=True)
    layer = etree.fromstring(buffer.getvalue())
    original = etree.parse(str(base.with_suffix(".svg")))
    old = original.xpath('//*[@id="axes_7"]')[0]
    assert any("13.07%" in value for value in old.itertext())

    ids = {node.attrib["id"]: panel_id + "_" + node.attrib["id"] for node in layer.iter() if "id" in node.attrib}
    for node in layer.iter():
        for name, value in list(node.attrib.items()):
            if name == "id":
                node.attrib[name] = ids[value]
            elif name.endswith("href") and value.startswith("#") and value[1:] in ids:
                node.attrib[name] = "#" + ids[value[1:]]
            else:
                node.attrib[name] = re.sub(r"url\(#([^)]+)\)", lambda m: f"url(#{ids[m[1]]})", value)
    replacement = etree.Element(NS + "g", id=panel_id)
    for node in list(layer):
        if node.tag in {NS + "g", NS + "defs"}:
            replacement.append(node)
    old.getparent().replace(old, replacement)
    original.write(str(output.with_suffix(".svg")), encoding="UTF-8", xml_declaration=True)


def replace_raster_panel(figure, base, output):
    source = Image.open(base.with_suffix(".png")).convert("RGBA")
    width, height = source.size
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=width / 11, transparent=True)
    buffer.seek(0)
    layer = Image.open(buffer).convert("RGBA")
    assert layer.size == source.size
    box = (0, round(height * 0.668), round(width * 0.49), height)
    left, top, right, bottom = box
    active = layer.getbbox()
    assert left <= active[0] and top <= active[1] and right >= active[2] and bottom >= active[3]
    result = source.copy()
    result.paste(Image.new("RGBA", (right-left, bottom-top), "white"), box)
    result = Image.alpha_composite(result, layer).convert("RGB")
    assert ImageChops.difference(result.crop((0, 0, width, top)), source.convert("RGB").crop((0, 0, width, top))).getbbox() is None
    assert ImageChops.difference(result.crop((right, top, width, height)), source.convert("RGB").crop((right, top, width, height))).getbbox() is None
    result.save(output.with_suffix(".png"), dpi=(width / 11, width / 11))


def draw_water_panel(scores, correlations):
    plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "axes.linewidth": 0.9,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure = plt.figure(figsize=(11, 12))
    ax = figure.add_axes((0.084, 0.077, 0.381, 0.200))
    for habitat, color in COLORS.items():
        observations = scores.loc[scores.habitat.eq(habitat)]
        xy = observations[["axis1", "axis2"]].to_numpy()
        centre = xy.mean(axis=0)
        covariance = np.cov(xy, rowvar=False)
        eigenvalues, vectors = np.linalg.eigh(covariance)
        residuals = xy - centre
        distances = np.einsum("ij,jk,ik->i", residuals, np.linalg.inv(covariance), residuals)
        height, width = 2 * np.sqrt(eigenvalues * np.quantile(distances, 0.9))
        angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
        ax.add_patch(Ellipse(centre, width, height, angle=angle, facecolor=color,
                             edgecolor=color, alpha=0.18, linewidth=0.9))
        ax.scatter(*xy.T, color=color, s=8, alpha=0.30, linewidths=0,
                   label=f"{habitat.title()} (n={len(xy):,})")
        study = observations.groupby("group")[["axis1", "axis2"]].mean()
        ax.scatter(*study.to_numpy().T, color=color, marker="D", s=16,
                   edgecolors="#333333", linewidths=0.4, zorder=4)

    ax.axhline(0, color="#999999", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#999999", linewidth=0.6, zorder=0)
    ax.margins(0.08)
    # Use the strongest four two-axis correlations to fit the smaller main-text panel.
    directions = correlations.set_index("feature")[["axis1_r", "axis2_r"]]
    chosen = directions.loc[(directions**2).sum(axis=1).nlargest(4).index]
    reference = np.quantile(abs(scores[["axis1", "axis2"]]), 0.93, axis=0)
    endpoints = chosen * (0.84 * reference / np.abs(chosen.to_numpy()).max())
    labels = {"Fiber": "Fibres", "Fragment": "Fragments", "0–100 μm": "0–100 µm",
              "500–5000 μm": "500–5,000 µm"}
    offsets = {"Fiber": (4, 5), "Fragment": (-5, 12), "0–100 μm": (-5, -10),
               "500–5000 μm": (5, -10)}
    for feature, (x, y) in endpoints.iterrows():
        ax.annotate("", (x, y), (0, 0), arrowprops={"arrowstyle": "-|>", "lw": 0.9,
                                                  "color": "#333333"}, zorder=5)
        ax.annotate(labels[feature], (x, y), xytext=offsets[feature], textcoords="offset points",
                    ha="left" if x > 0 else "right", va="center", fontsize=9, zorder=6)

    ax.set_title("(e) Water", fontsize=12, pad=6)
    ax.set_xlabel("PLS-DA LV1", fontsize=11, labelpad=3)
    ax.set_ylabel("PLS-DA LV2", fontsize=11, labelpad=2)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(labelsize=9.5, pad=2, length=3)
    legend = ax.legend(loc="upper left", fontsize=9, frameon=False, handletextpad=0.2)
    for marker in legend.legend_handles:
        marker.set_alpha(1)
    return figure


def render(base, tables, output):
    scores = pd.read_csv(tables / "ordination_coordinates.csv")
    correlations = pd.read_csv(tables / "ordination_correlations.csv")
    scores = scores.loc[scores.medium.eq("water") & scores.model.eq("PLS-DA")]
    correlations = correlations.loc[correlations.medium.eq("water") & correlations.model.eq("PLS-DA")]
    assert scores.id.is_unique and len(scores) == 1589 and scores.group.nunique() == 85
    assert scores.groupby("habitat").size().to_dict() == {"lake": 247, "river": 1342}
    figure = draw_water_panel(scores, correlations)
    replace_vector_panel(figure, base, output / "Figure_2")
    replace_raster_panel(figure, base, output / "Figure_2")
    plt.close(figure)
