"""Render ECH Figures 3 and 4 with the accepted manuscript panel design."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
VARIANTS = {'Combined': 'WaterSediment', 'water': 'Water', 'sediment': 'Sediment'}
HABITATS = ('river', 'lake')
GRADE_NAMES = ('Very Low', 'Low', 'Medium', 'High', 'Very High')
F3_COLORS = ('#48966c', '#75b976', '#fde39b', '#fd9b76', '#d95a65')
F4_COLORS = ('#d3d9df', '#b0b8c0', '#7e8790', '#3c89a6', '#c65033')
SCENARIOS = ('Baseline','High H','High C','High E','Maximum','Minimum','Sharper gate')
LABELS = {'size_profile': 'Size profile', 'shape_effects': 'Shape effects',
          'color_effects': 'Color effects', 'material_effects': 'Material effects',
          'prior_E': 'E prior', 'prior_C': 'C prior', 'prior_H': 'H prior',
          'temperature': 'Temperature'}
STYLE = {'font.family': 'Arial', 'font.size': 11, 'pdf.fonttype': 42,
         'svg.fonttype': 'none', 'axes.spines.top': False,
         'axes.spines.right': False, 'axes.linewidth': .75}
plt.rcParams.update(STYLE)


def save(fig, stem, dpi=300):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=dpi, facecolor='white')
    plt.close(fig)


def panel(ax, letter, x=-.22, y=1.11):
    ax.text(x, y, f'({letter})', transform=ax.transAxes, fontsize=18,
            ha='left', va='top', clip_on=False)


def correlations(frame, feature):
    cols = [feature, 'E', 'Z']
    values = frame[cols].to_numpy()
    weights = 1 / frame.groupby('group').id.transform('size').to_numpy()
    weights /= weights.sum()
    values = values - weights @ values
    cov = values.T @ (values * weights[:, None])
    sd = np.sqrt(cov.diagonal())
    r = cov / np.outer(sd, sd)
    return r[0, 1], r[0, 2], r[1, 2]


def figure3(data, medium, stem):
    frame = data if medium == 'Combined' else data[data.medium.eq(medium)]
    combined = medium == 'Combined'
    fig = plt.figure(figsize=(11.75, 9.9))
    grid = fig.add_gridspec(3, 2, left=.104, right=.894, bottom=.080, top=.918,
                            wspace=.43, hspace=.57, height_ratios=[1.15, 1., 1.])
    color_map = plt.get_cmap('coolwarm')
    scale = Normalize(0, 1)
    audit = []
    for col, habitat in enumerate(HABITATS):
        sample = frame[frame.habitat.eq(habitat)]
        ax = fig.add_subplot(grid[0, col])
        counts = np.bincount(sample.grade, minlength=6)[1:]
        assert counts.sum() == len(sample)
        bottom = np.zeros(5)
        for matrix, hatch in [('water', ''), ('sediment', '///')]:
            g = sample.loc[sample.medium.eq(matrix), 'grade']
            if len(g):
                height = np.bincount(g, minlength=6)[1:]
                ax.bar(np.arange(5), height, bottom=bottom, color=F3_COLORS, edgecolor='#4b4b4b',
                       hatch=hatch if combined else None, width=.83, lw=.65)
                bottom += height
        ax.set_xticks(np.arange(5), GRADE_NAMES, rotation=31, ha='right')
        ax.set(ylabel='Number of samples', ylim=(0, 1.27 * max(counts)))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.grid(axis='y', color='#efefef', lw=.55)
        ax.set_axisbelow(True)
        pos = ax.get_position()
        ax_title = fig.add_axes([pos.x0, pos.y1+.027, pos.width, .034], facecolor='none')
        ax_title.add_patch(Rectangle((0, 0), 1, 1, facecolor=('#e5f4f5', '#f5e5ee')[col], edgecolor='none'))
        ax_title.text(.5,.49,habitat.title(),ha='center',va='center',fontsize=17)
        ax_title.set_axis_off()
        panel(ax, 'ab'[col], -.22, 1.30)
        if combined and col == 0:
            ax.legend(handles=[Patch(facecolor='white', edgecolor='#666', label='Water'),
                               Patch(facecolor='white', edgecolor='#666', hatch='///', label='Sediment')],
                      loc='upper right', fontsize=7.4)
        audit.extend(dict(medium=medium, habitat=habitat, grade=i+1, n=int(n)) for i, n in enumerate(counts))
        for row, feature in [(1, 'H'), (2, 'C')]:
            ax = fig.add_subplot(grid[row, col])
            n = 0
            for matrix, mark in [('water', 'o'), ('sediment', '^')]:
                q = sample[sample.medium.eq(matrix)]
                if len(q):
                    dots = ax.scatter(q.E, q.Z, c=q[feature], cmap=color_map, norm=scale,
                                      marker=mark if combined else 'o', s=11, alpha=.69,
                                      edgecolors='white', lw=.12, rasterized=True)
                    n += len(dots.get_offsets())
            assert n == len(sample)
            ax.set(xlim=(-.025, 1.025), ylim=(-3, 103), xticks=np.linspace(0, 1, 6),
                   yticks=np.arange(0, 101, 20), xlabel='Exposure (E)', ylabel='Composite score Z')
            for spine in ax.spines.values():
                spine.set_visible(True)
            panel(ax, ('cd' if row == 1 else 'ef')[col], -.22, 1.10)
            if col:
                keys = [Line2D([], [], ls='', marker='o', color=color_map(scale(v)), markersize=4.0,
                               label=f'{v:.1f}') for v in np.linspace(0,1,6)]
                ax.legend(handles=keys, title=feature, bbox_to_anchor=(1.023, 1.009),
                          loc='upper left', borderpad=.33, fontsize=8, title_fontsize=10,
                          handlelength=.8, handletextpad=.37, labelspacing=.23, framealpha=.97)
    save(fig, f'Figure_3_{stem}')
    return audit


def figure4(sobol, stress, medium, stem, xlim, ylim):
    m = sobol[sobol.medium.eq(medium)]
    rank = m[m.habitat.eq('all')].sort_values('ST', ascending=False)
    order = rank.parameter.tolist()
    fig = plt.figure(figsize=(11.65, 12.5))
    grid = fig.add_gridspec(3, 2, left=.238, right=.949, bottom=.195, top=.906,
                            wspace=.48, hspace=.69, height_ratios=[1.24, .85, .85])
    for j, kind in enumerate(('S1','ST')):
        ax = fig.add_subplot(grid[0,j])
        for i, habitat in enumerate(HABITATS):
            p = m[m.habitat.eq(habitat)].set_index('parameter').loc[order]
            estimate = p[kind].to_numpy()
            errors = np.maximum(0, [estimate-p[kind+'_min'].to_numpy(), p[kind+'_max'].to_numpy()-estimate])
            color = (('#287b9d','#a1c9db'),('#b44125','#eaa896'))[j][i]
            ax.barh(np.arange(len(order)) + (.16 if i == 0 else -.16), estimate,
                    height=.30, color=color, edgecolor='#555', lw=.40, xerr=errors,
                    error_kw=dict(lw=.64,capsize=1.65), label=habitat.title(), zorder=3)
        ax.set_yticks(np.arange(len(order)), [LABELS[v] for v in order])
        ax.invert_yaxis()
        ax.set_xlim(*xlim)
        ax.set_title('$S_1$' if j == 0 else '$S_T$', loc='left', fontsize=18, pad=6)
        ax.set_xlabel('First-order sensitivity index' if j == 0 else 'Total-effect sensitivity index')
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.grid(axis='x', alpha=.17)
        ax.axvline(0, color='#969da0', lw=.65)
        ax.legend(loc='lower right', fontsize=8, framealpha=.96)
        ax.text(-.46, 1.10, f'({"ab"[j]})', transform=ax.transAxes, fontsize=18, va='top')
    for j, habitat in enumerate(HABITATS):
        p = stress[stress.medium.eq(medium)&stress.habitat.eq(habitat)].set_index('scenario').loc[list(SCENARIOS)]
        x = np.arange(len(p))
        labels = [name.replace('Sharper gate', 'Sharper\ngate') for name in SCENARIOS]
        box = grid[1,j].get_position(fig)
        ax = fig.add_axes([box.x0-.049, box.y0, box.width+.049, box.height])
        mu, lo, hi = (p[field].to_numpy() for field in ('study_mean_Z','Z_low','Z_high'))
        ax.errorbar(x, mu, yerr=[mu-lo,hi-mu], fmt='o', color='#161616',
                    ecolor='#2683a8', ms=4.7, lw=1.55, zorder=3)
        ax.axhline(mu[0], color='#c64c2e', lw=1.1, ls='--', label='Baseline')
        ax.set(ylabel='Composite score Z', ylim=ylim, xlim=(-.45,6.45))
        ax.set_title(habitat.title(), fontweight='bold', fontsize=15, pad=7)
        ax.set_xticks(x, labels, rotation=33, ha='right', rotation_mode='anchor', fontsize=9)
        ax.grid(alpha=.17)
        ax.legend(loc='upper left', fontsize=7.8)
        ax.yaxis.set_major_locator(MaxNLocator(6))
        panel(ax, 'cd'[j], -.20, 1.15)
        box = grid[2,j].get_position(fig)
        ax = fig.add_axes([box.x0-.049, box.y0, box.width+.049, box.height])
        bot = np.zeros(len(p))
        for grade, color in enumerate(F4_COLORS, 1):
            share = p[f'G{grade}_study_fraction'].to_numpy()
            ax.bar(x, share, bottom=bot, width=.80, color=color, edgecolor='#60666a', lw=.38, zorder=3)
            bot += share
        np.testing.assert_allclose(bot, 1, atol=1e-9)
        ax.set(ylim=(0,1.01), xlim=(-.45,6.45), ylabel='Study-weighted proportion')
        ax.set_xticks(x, labels, rotation=33, ha='right', rotation_mode='anchor', fontsize=9)
        ax.yaxis.set_major_locator(MultipleLocator(.2))
        ax.grid(axis='y', alpha=.17)
        panel(ax, 'ef'[j], -.20)
    for x0, y0, width, height, border in [(.077,.643,.894,.326,'#287b9d'),(.077,.077,.894,.551,'#b44125')]:
        fig.add_artist(Rectangle((x0,y0), width,height, transform=fig.transFigure, fill=False,
                                 edgecolor=border, linestyle=(0,(3,3)), lw=1.55))
    fig.legend(handles=[Patch(facecolor=c,edgecolor='#666',label=g) for c,g in zip(F4_COLORS,GRADE_NAMES)],
               loc='lower center', bbox_to_anchor=(.55,.084), ncol=5, fontsize=8.7, columnspacing=.9)
    save(fig, f'Figure_4_{stem}')
