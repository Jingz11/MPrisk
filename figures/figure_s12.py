from figures_6_7_shared import np, pd, plt, panel


def render(estimates, diagnostics, blocked, spatial, output):
    main=estimates.query("scope=='National' and habitat=='lake' and variable=='IPME'").iloc[0]
    s=spatial.query("scope=='National' and habitat=='lake' and variable=='IPME'").set_index('bandwidth_km')
    support=diagnostics.query('k==4').status.eq('Estimated').sum()
    fitted=blocked.query("status=='estimated'")
    signs=int((np.sign(fitted.theta_standardized)!=np.sign(fitted.main_theta_standardized)).sum())
    intervals=pd.DataFrame([dict(Inference='Cluster',coefficient=main.theta_standardized,low=main.ci_low,high=main.ci_high,p_Holm=main.p_Holm),
                           *[dict(Inference=f'Spatial {km} km',coefficient=s.loc[km,'theta_standardized'],low=s.loc[km,'spatial_low'],high=s.loc[km,'spatial_high'],p_Holm=s.loc[km,'spatial_p_Holm']) for km in [200,500]]])
    fig,axs=plt.subplots(1,2,figsize=(10.4,4.1),gridspec_kw={'width_ratios':[1,1.15]})
    ax=axs[0];counts=[support,len(fitted),signs];labels=['Spatial diagnostics','Province-block fits','Sign changes']
    ax.barh(range(3),counts,color=['#9ecae1','#6baed6','#2171b5'],height=.6)
    ax.set(yticks=range(3),yticklabels=labels,xlim=(0,43),xlabel='Number of specifications');ax.invert_yaxis();panel(ax,'a',-.12,1.10)
    ax=axs[1]
    for y,r in enumerate(intervals.itertuples()):ax.errorbar(r.coefficient,y,xerr=[[r.coefficient-r.low],[r.high-r.coefficient]],fmt='o',capsize=3,color='#2171b5',lw=1.5)
    ax.axvline(0,color='black',lw=.8)
    ax.set(yticks=range(3),yticklabels=intervals.Inference,xlim=(-1.025,.05),xlabel='Standardized lake IPME coefficient');ax.invert_yaxis();panel(ax,'b',-.12,1.10)
    fig.subplots_adjust(left=.18,right=.98,bottom=.20,top=.90,wspace=.52)
    for ext in ['png','pdf','svg']:fig.savefig(output/f'Figure_S12.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
