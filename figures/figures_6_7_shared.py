"""Apply the historical manuscript geometry to the ECH downstream tables."""
from figures_3_4 import plt, pd, np, panel
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

HAB = ('river','lake')
GREEN, GOLD, RED = '#319c90', '#eac66e', '#c65033'
MODELS = ('Direct','Pure','Hybrid II')
FEATURES = dict([
    ('agri_film_prov_t','AgriFilm'), ('PopDensity_urban_construction','PopDensity'),
    ('GRP_pc','GRP_pc'), ('temp_L8_climatology_degC','Temp (L8)'),
    ('agri_output_prov_100mCNY','AgriOutput'), ('elevation_2p5min_m','ALT'),
    ('Waste_Treat','Waste_Treat'), ('fertilizer_prov_10kt','Fertilizer'),
    ('WW_Discharge','WW_Discharge'), ('IPME','IPME'), ('VIIRS_valid_3km_nW_cm2_sr','VIIRS'),
])
COMP = ('E_ln_count','C_ln_opportunity','H_ln_multiplier')
VAR = dict([
    ('fertilizer_prov_10kt','Fertilizer'), ('AgriOutput_nominal_province','AgriOutput'),
    ('PopDensity_urban_construction','PopDensity'), ('GRP_pc','GRP_pc'),
    ('WW_Discharge','WW_Discharge'), ('Waste_Treat','Waste_Treat'),
    ('AgriFilm','AgriFilm'), ('IPME','IPME'),
])
BASINS = ('Indian Ocean Basin','Yellow River Basin','Pearl River Basin','Yangtze River Basin',
          'Yarlung Zangbo River Basin','Hai River Basin','Songhua River Basin','Tarim River Basin')


def outline(fig, x, y, w, h, color):
    fig.add_artist(Rectangle((x,y),w,h,transform=fig.transFigure,fill=False,
                             edgecolor=color,linestyle=(0,(3,2)),lw=1.1))


def ribbon(fig,x,y,w,label,fill,ink):
    fig.add_artist(Rectangle((x,y),w,.027,transform=fig.transFigure,
                             facecolor=fill,edgecolor=ink,lw=.35))
    ax = fig.add_axes([x, y, w, .027], frameon=False)
    ax.set_axis_off()
    ax.set_title(label, fontsize=15, fontweight='bold', color=ink, pad=0, y=.12)


def resolve_vars(observed):
    preferences = [('Fertilizer','fertilizer_prov_10kt','Fertilizer_nutrient_province'),
                   ('AgriOutput','AgriOutput_nominal_province'),('PopDensity','PopDensity_urban_construction'),
                   ('GRP_pc','GRP_pc'),('WW_Discharge','WW_Discharge'),('Waste_Treat','Waste_Treat'),
                   ('AgriFilm','AgriFilm','MulchFilm_province','agri_film_prov_t'),('IPME','IPME')]
    return {c:next((v for v in vals if v in observed),vals[0]) for c,*vals in preferences}
