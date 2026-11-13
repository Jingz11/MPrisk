"""Figure 7 original panel layout with the new river/lake and basin fits."""

from pathlib import Path
import sys

from figures_6_7_shared import BASINS, GOLD, GREEN, RED, panel, np, pd, plt, resolve_vars
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

BASIN_ORDER = ("Yangtze River Basin", "Pearl River Basin", "Yellow River Basin")
BASIN_COLORS = ("#3e80aa", "#dc7540", "#83a449")
FOREST_ORDER = ("Fertilizer", "PopDensity", "GRP_pc", "AgriOutput", "Waste_Treat", "IPME", "AgriFilm", "WW_Discharge")


def national(fig, ax, estimates, variables):
    labels = list(variables)
    matrix, stars = np.empty((2, len(labels))), np.zeros((2, len(labels)), dtype=bool)
    for i, habitat in enumerate(("river", "lake")):
        rows = estimates.loc[estimates.habitat.eq(habitat)].set_index("variable").loc[list(variables.values())]
        matrix[i] = rows.theta_standardized
        stars[i] = rows.p.lt(.05)
    edge = max(.25, abs(matrix[np.isfinite(matrix)]).max())
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eaeaea")
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, norm=TwoSlopeNorm(0, -edge, edge))
    for i, j in np.ndindex(matrix.shape):
        value = matrix[i, j]
        label = f"{value:.2f}" + ("*" if stars[i, j] else "") if np.isfinite(value) else "—"
        ax.text(j, i, label, ha="center", va="center", fontsize=11, weight="bold",
                color="white" if np.isfinite(value) and abs(value)>.66*edge else "#262626")
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks([0, 1], ["River", "Lake"])
    cb = fig.colorbar(im, ax=ax, fraction=.026, pad=.020, aspect=10)
    cb.ax.set_title("DML association\n(α)", fontsize=8)
    panel(ax, "a", -.23, 1.22)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)


def basin_forest(fig, cell, estimates, variables):
    grid = cell.subgridspec(1, 3, wspace=.22)
    for i, (basin, color) in enumerate(zip(BASIN_ORDER, BASIN_COLORS)):
        ax = fig.add_subplot(grid[i])
        rows = estimates.loc[estimates.scope.eq(basin)].set_index("variable").loc[[variables[v] for v in FOREST_ORDER]]
        extent = np.nanmax(np.abs(rows[["theta_standardized", "ci_low", "ci_high"]].to_numpy()))
        limit = max(.4, extent*1.12)
        for j, r in enumerate(rows.itertuples()):
            if j % 2:
                ax.axhspan(j-.46, j+.46, color="#f3f5f8", zorder=0)
            if not np.isfinite(r.theta_standardized):
                ax.text(.93, j, "—", ha="right", va="center", fontsize=9, transform=ax.get_yaxis_transform())
            elif np.isfinite(r.ci_low) and np.isfinite(r.ci_high):
                starred = r.p < .05
                ax.hlines(j, r.ci_low, r.ci_high, color=color, alpha=.88 if starred else .53, lw=1.45)
                ax.scatter(r.theta_standardized, j, marker="*" if starred else "o", s=67 if starred else 28,
                           color=color, edgecolors="#303030" if starred else color, linewidths=.55, zorder=3)
            else:
                ax.scatter(r.theta_standardized, j, marker="o", s=30, facecolors="white", edgecolors=color,
                           linewidths=1.1, zorder=3)
        ax.axvline(0, color="#7e868a", ls="--", lw=.8, zorder=1)
        ax.set(xlim=(-limit, limit), ylim=(len(rows)-.48, -.52), yticks=range(len(rows)),
               yticklabels=FOREST_ORDER if i == 0 else [""]*len(rows), xlabel="DML association (α)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="y", length=0, labelsize=9.2)
        ax.grid(axis="x", alpha=.16)
        ax.set_title(basin, color=color, fontweight="bold", fontsize=11.5)
        if i == 0:
            ax.set_ylabel("Predictor variables")
            panel(ax, "c", -.70, 1.13)
        if i == 2:
            ax.legend(handles=[Line2D([], [], color="#555", marker="o", label="95% interval", ms=4),
                               Line2D([], [], color="#555", marker="o", label="Point only", ms=4,
                                      markerfacecolor="white", ls="")],
                      fontsize=7.3, frameon=False, loc="upper right", bbox_to_anchor=(1.01, -.16), ncol=2)


def signed_profiles(fig,cell,data):
    grid=cell.subgridspec(1,3,width_ratios=(1.05,1.6,.43),wspace=.33)
    score,contrib,support=[fig.add_subplot(grid[i]) for i in range(3)]
    q=data.set_index('region').loc[list(BASINS)]
    y=np.arange(len(q))
    for ax in (score,contrib,support):
        for j in y:
            if j%2:ax.axhspan(j-.46,j+.46,color='#f1f4f6',zorder=0)
        ax.set_ylim(len(q)-.48,-.52)
        ax.tick_params(axis='y',length=0)
    score.hlines(y,0,q.study_equal_Z,color='#173f59',lw=1.2)
    score.scatter(q.study_equal_Z,y,color='#173f59',s=28,zorder=3)
    score.axvline(50,color='#222',lw=.8,ls=(0,(4,4)))
    score.set(xlim=(0,111),xticks=[0,50,100],yticks=y,yticklabels=BASINS,xlabel='Score (points)')
    score.tick_params(axis='y',labelsize=9.2)
    score.set_title('Mean Z',fontsize=11,fontweight='bold',pad=16)
    score.spines['left'].set_visible(False)
    colors=['#31998f','#eac968','#c85131'];positive=np.zeros(len(q));negative=positive.copy()
    for k,color in zip('ECH',colors):
        value=q['study_equal_phi_'+k].to_numpy();left=np.where(value>=0,positive,negative)
        contrib.barh(y,value,left=left,color=color,height=.51,label=k,zorder=2)
        positive+=np.maximum(0,value);negative+=np.minimum(0,value)
    contrib.axvline(0,color='#222',lw=.8)
    contrib.set(xlim=(-50,50),xticks=[-40,-20,0,20,40],yticks=[],xlabel='Contribution relative to 50 (points)')
    contrib.spines['left'].set_visible(False)
    contrib.set_title('Signed Shapley contributions',fontsize=11,fontweight='bold',pad=24)
    contrib.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,.995),handlelength=1,fontsize=9,columnspacing=1.1)
    support.set(xlim=(0,1),yticks=[],xticks=[])
    support.spines[['left','bottom']].set_visible(False)
    support.set_title('Support',fontsize=11,fontweight='bold',pad=16)
    support.text(.17,1.01,'n',transform=support.transAxes,ha='center',fontsize=9,fontweight='bold')
    support.text(.77,1.01,'Groups',transform=support.transAxes,ha='center',fontsize=9,fontweight='bold')
    for j,r in enumerate(q.itertuples()):
        support.text(.17,j,str(r.rows),ha='center',va='center',fontsize=9)
        support.text(.77,j,str(r.studies),ha='center',va='center',fontsize=9)
    panel(score,'b',-.67,1.13)

def render(estimates,regions,output):
    variables=resolve_vars(set(estimates.variable))
    fig=plt.figure(figsize=(12.1,10.2))
    gs=fig.add_gridspec(3,1,left=.186,right=.968,top=.935,bottom=.115,height_ratios=(.43,1.19,1.15),hspace=.63)
    national(fig,fig.add_subplot(gs[0]),estimates.query("scope=='National'"),variables)
    signed_profiles(fig,gs[1],regions)
    basin_forest(fig,gs[2],estimates.query("scope!='National'"),variables)
    for ext in ['png','pdf','svg']:fig.savefig(output/f'Figure_7.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
