import pandas as pd
import figures_3_4 as v


def render(data, pairs, rms, finish):
    def save_main(fig,stem,dpi=300):
     axes=[a for a in fig.axes if a.get_subplotspec() is not None and a.get_subplotspec().rowspan.start==2]
     axes=sorted(axes,key=lambda a:a.get_position().x0)
     for ax in axes:ax.clear()
     a,b=axes
     for i,(key,col,name) in enumerate([('C','#3274A1','Contact'),('H','#D48437','Material'),('CH','#5B8165','Joint C–H')]):
      g=pairs[pairs.contrast.eq(key)].copy();g['a']=abs(g.delta_Z);g['w']=1/g.groupby('group').id1.transform('size');g=g.sort_values('a');g['w']/=g.w.sum()
      a.step(g.a,g.w.cumsum(),where='post',color=col,lw=1.8,label=name)
      r=rms.loc[key];b.errorbar(r.RMS,i,xerr=[[r.RMS-r.RMS_low],[r.RMS_high-r.RMS]],fmt='o',ms=6,c=col,lw=1.8,capsize=3)
     a.set(xlabel='Absolute matched-exposure ΔZ',ylabel='Cumulative proportion',ylim=(0,1.03),xlim=(0,27))
     a.legend(frameon=False,fontsize=9,loc='lower right');a.set_title('Matched exposure',fontsize=11,pad=7)
     b.set(yticks=[0,1,2],yticklabels=['Contact','Material','Joint C–H'],xlabel='RMS score difference (points)',xlim=(0,6.4),ylim=(2.6,-.6))
     b.set_title('Nominal parameters',fontsize=11,pad=7)
     for ax,letter in zip(axes,'ef'):
      v.panel(ax,letter,-.22,1.10);ax.tick_params(labelsize=10);ax.xaxis.label.set_size(11);ax.yaxis.label.set_size(11)
      ax.grid(axis='x',color='#e7e7e7',lw=.6);ax.set_axisbelow(True)
     finish(fig,'Figure_3')
    previous = v.save
    v.save = save_main
    try:
        v.figure3(data, 'Combined', 'WaterSediment')
    finally:
        v.save = previous
