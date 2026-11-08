"""Redraw two source-backed benchmark figures; no fitting or inferential tests."""
import os
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'results/figures_2026_11_08'
QA = ROOT / 'provenance/figures_2026_11_08'
OUT = ROOT / 'work/figure_redesign_2026_11_08'
SKILL = Path(os.environ['NATURE_FIGURE_SKILL'])
sys.path.insert(0, str(SKILL / 'scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7.2, 'axes.labelsize': 7.2, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7.2,
    'axes.linewidth': .55, 'lines.linewidth': .8, 'axes.spines.top': False, 'legend.frameon': False,
    'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})
COL = {'scaffold': '#7C8389', 'descriptor': '#207F88', 'morgan': '#CE795D'}
REG = ['scaffold', 'descriptor', 'morgan']
TASK = ['BBB', 'HIA', 'CYP2D6', 'DILI', 'PPBR', 'Solubility']
MODELS = ['physchem:logistic', 'maccs:logistic', 'morgan:logistic', 'morgan:rf',
          'morgan:mlp', 'graph:gnn', 'chemprop:dmpnn']
LABELS = ['Physchem lin.', 'MACCS lin.', 'Morgan lin.', 'Morgan RF', 'Morgan MLP',
          'Graph control', 'Chemprop']
METRIC = {e: 'mae' if e in ['PPBR', 'Solubility'] else 'auprc' if e == 'CYP2D6'
          else 'auroc' for e in TASK}
ML = {'auroc': 'AUROC', 'auprc': 'Average precision', 'mae': 'MAE'}

def panel(ax, letter, title):
    ax.set_gid(letter)
    ax.text(0, 1, letter, transform=ax.transAxes + ScaledTranslation(-8/72, 7/72,
            ax.figure.dpi_scale_trans), ha='left', va='bottom', fontsize=8.5, weight='bold')
    ax.set_title(title, loc='left', pad=8, fontweight='bold')
    ax.tick_params(length=2.5, width=.55, pad=2)

def export(fig, name, **alignment):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,
        json_out=QA/(name+'.alignment.json'), overlay_svg=QA/(name+'.alignment.svg'),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True,
        **alignment)
    fig.savefig(OUT/(name+'.pdf'), format='pdf')
    fig.savefig(OUT/(name+'.svg'), format='svg')
    fig.savefig(OUT/(name+'.png'), format='png', dpi=600)
    plt.close(fig)

def benchmark():
    raw = pd.read_csv(DATA/'figure1_bbb_absolute_display.csv')
    delta = pd.read_csv(DATA/'figure1_endpoint_contrasts_display.csv')
    assert len(raw) == 20 and len(delta) == 78
    fig = plt.figure(figsize=(170/25.4, 185/25.4))
    grid = fig.add_gridspec(3, 3, left=.157, right=.976, bottom=.113, top=.914,
                           height_ratios=[1.15, 1, 1], wspace=.90, hspace=.83)
    hero = fig.add_subplot(grid[0, :])
    for j, reg in enumerate(REG):
        for i, model in enumerate(MODELS):
            rows = raw[(raw.regime == reg) & (raw.model == model)]
            if rows.empty: continue
            row = rows.iloc[0]
            hero.errorbar(i+(j-1)*.19, row['mean'],
                yerr=[[max(0, row['mean']-row.lower)], [max(0, row.upper-row['mean'])]],
                fmt=['s','o','^'][j], color=COL[reg], ms=3.8, capsize=1.7,
                elinewidth=.8, markeredgewidth=.65, zorder=3)
    hero.axhline(.5, color='.7', ls=(0,(2,3)), lw=.55, zorder=0)
    hero.set_ylim(.20, 1.01)
    hero.set_xlim(-.65, 6.65)
    hero.set_xticks(range(7), ['Physchem\nlinear', 'MACCS\nlinear', 'Morgan\nlinear',
                             'Morgan\nRF', 'Morgan\nMLP', 'Graph\ncontrol', 'Chemprop'])
    hero.set_ylabel('AUROC')
    hero.set_yticks([.25,.50,.75,1.00])
    panel(hero, 'a', 'BBB | Absolute performance across split regimes')
    forests=[]
    for k, e in enumerate(TASK):
        ax=fig.add_subplot(grid[1+k//3, k%3]); forests.append(ax)
        d=delta[delta.endpoint == e]
        for i, model0 in enumerate(MODELS):
            model=model0.replace('logistic','ridge') if e in ['PPBR','Solubility'] else model0
            for j, reg in enumerate(['descriptor','morgan']):
                q=d[(d.frontier == reg) & (d.model == model)]
                if q.empty: continue
                row=q.iloc[0]; val=row.delta_frontier_minus_scaffold
                ax.errorbar(val, i+(j-.5)*.23,
                    xerr=[[max(0,val-row.lower)], [max(0,row.upper-val)]],
                    fmt='o' if j==0 else '^', color=COL[reg], ms=3, capsize=1.3,
                    elinewidth=.75, markeredgewidth=.6)
        low=min(0, d.lower.min());high=max(0,d.upper.max());pad=(high-low)*.12
        ax.set_xlim(low-pad,high+pad)
        ax.axvline(0, color='.65', ls=(0,(2,3)), lw=.55, zorder=0)
        ax.set_yticks(range(7), LABELS)
        ax.set_ylim(6.65,-.65)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        xlabel='Δ'+ML[METRIC[e]]
        if e=='CYP2D6':xlabel='ΔAverage precision'
        if e=='PPBR':xlabel='ΔMAE (percentage points)'
        if e=='Solubility':xlabel='ΔMAE (log molar)'
        ax.set_xlabel(xlabel, labelpad=3)
        panel(ax, 'bcdefg'[k], e)
    fig.legend(handles=[Line2D([],[],marker=m,ls='',color=COL[r],ms=4,label=l)
        for r,m,l in zip(REG,['s','o','^'],['Scaffold control','Descriptor frontier','Morgan frontier'])],
        loc='upper center',bbox_to_anchor=(.56,.988),ncol=3,columnspacing=1.3,handletextpad=.5)
    fig.text(.157,.033,'Negative ΔAUROC/AP denotes a drop; positive ΔMAE denotes larger error.',fontsize=7)
    export(fig,'review_benchmark',
           row_groups=[{'id':'forest-row-1','panels':['b','c','d']},
                       {'id':'forest-row-2','panels':['e','f','g']}],
           column_groups=[{'id':'forest-col-1','panels':['b','e']},
                          {'id':'forest-col-2','panels':['c','f']},
                          {'id':'forest-col-3','panels':['d','g']}])

def allocation_panel(ax, e, raw, summary, hero=False):
    rows=[('descriptor','physchem'),('descriptor','morgan'),('morgan','physchem'),('morgan','morgan')]
    ypos=np.array([0,1.25,2.9,4.15])
    for y,(geometry,rep) in zip(ypos,rows):
        c=raw[(raw.endpoint==e)&(raw.geometry==geometry)&(raw.representation==rep)]
        vals=c[c['mode']=='exact_random'].display_value.to_numpy()
        assert len(vals)==50 and np.isfinite(vals).all()
        s=summary[(summary.endpoint==e)&(summary.geometry==geometry)&(summary.representation==rep)].iloc[0]
        assert np.isclose(vals.mean(),s.random_mean,atol=1e-12)
        assert np.allclose(np.quantile(vals,[.025,.975]),[s.random_q025,s.random_q975],atol=1e-12)
        color=COL[geometry]
        # KDE is descriptive; its support is clipped to the observed values.
        domain=np.linspace(vals.min(), vals.max(), 160)
        kde=gaussian_kde(vals,bw_method='scott')(domain)
        density=kde/kde.max()*.32
        ax.fill_between(domain,y-.12-density,y-.12,color=color,alpha=.23,lw=0)
        ax.plot(vals,np.full(len(vals),y+.32),'|',color=color,ms=2.8,alpha=.65,mew=.55)
        ax.plot([s.random_q025,s.random_q975],[y,y],color=color,lw=1.8,alpha=.7,zorder=2)
        ax.plot(s.random_mean,y,'s',mfc='white',mec=color,mew=.8,ms=3.2,zorder=4)
        ax.plot(s.frontier_reference,y-.22,'o',color=color,ms=4 if hero else 3.4,zorder=4)
        if np.isfinite(s.reverse_mean_over_allocation_records):
            ax.plot(s.reverse_mean_over_allocation_records,y+.22,'^',color=color,
                    ms=4 if hero else 3.4,zorder=4)
    ax.set_yticks(ypos,['Desc. / Physchem','Desc. / Morgan','Morgan / Physchem','Morgan / Morgan'])
    ax.set_ylim(4.85,-.8)
    ax.xaxis.set_major_locator(MaxNLocator(4 if hero else 3))
    d=summary[summary.endpoint==e]
    values=np.concatenate([raw[(raw.endpoint==e)&(raw['mode']=='exact_random')].display_value.to_numpy(),
                           d.frontier_reference.to_numpy(),
                           d.reverse_mean_over_allocation_records.to_numpy()])
    # Preserve missing reverse AUROC records in source data; omit only these
    # non-finite values when establishing numeric axis limits (HIA: 2 of 4).
    values=values[np.isfinite(values)]
    lo=values.min();hi=values.max();ax.set_xlim(max(0,lo-(hi-lo)*.09),hi+(hi-lo)*.09)
    label=ML[METRIC[e]]
    if e=='PPBR':label='MAE (percentage points)'
    if e=='Solubility':label='MAE (log molar)'
    ax.set_xlabel(label,labelpad=3)
    ax.tick_params(axis='y',length=0)
    ax.spines['left'].set_visible(False)

def allocation():
    raw=pd.read_csv(DATA/'figure3_allocation_observations_display.csv')
    summary=pd.read_csv(DATA/'figure3_allocation_summary_display.csv')
    assert len(raw)==1344 and len(summary)==24
    fig=plt.figure(figsize=(170/25.4,178/25.4))
    grid=fig.add_gridspec(3,3,left=.172,right=.974,bottom=.115,top=.936,
                          height_ratios=[1.1,1,1],wspace=1.02,hspace=.83)
    hero=fig.add_subplot(grid[0,:]);allocation_panel(hero,'DILI',raw,summary,hero=True)
    panel(hero,'a','DILI | Allocation variation and frontier references')
    axes=[]
    for k,e in enumerate(['BBB','HIA','CYP2D6','PPBR','Solubility']):
        ax=fig.add_subplot(grid[1+k//3,k%3]);axes.append(ax)
        allocation_panel(ax,e,raw,summary);panel(ax,'bcdef'[k],e)
    key=fig.add_subplot(grid[2,2]);key.axis('off')
    handles=[Line2D([],[],marker='o',ls='',color='.25',label='High frontier'),
             Line2D([],[],marker='^',ls='',color='.25',label='Reverse mean'),
             Line2D([],[],marker='s',ls='-',mfc='white',mec='.25',color='.5',label='Random mean / 95% range'),
             Line2D([],[],marker='|',ls='',color='.5',label='All 50 random allocations')]
    key.legend(handles=handles,loc='upper left',bbox_to_anchor=(-.22,1.02),fontsize=7,
               handlelength=1.2,handletextpad=.6,labelspacing=1.0)
    key.text(-.17,.35,'Teal: descriptor geometry\nCoral: Morgan geometry\n\nHIA descriptor reverse:\nAUROC undefined',transform=key.transAxes,fontsize=7,linespacing=1.4,va='top')
    fig.text(.172,.035,'Ranges describe allocation variation, not confidence intervals; shapes show descriptive density.',fontsize=7)
    export(fig,'review_allocation',exclude_axes=[key],
           row_groups=[{'id':'validation-row-1','panels':['b','c','d']},
                       {'id':'validation-row-2','panels':['e','f']}],
           column_groups=[{'id':'validation-col-1','panels':['b','e']},
                          {'id':'validation-col-2','panels':['c','f']}])

if __name__=='__main__':
    for p in [DATA,QA,OUT]:p.mkdir(parents=True,exist_ok=True)
    from prepare_benchmark_allocation import prepare
    prepare()
    benchmark()
    allocation()
    print('Exported benchmark and allocation candidate figures.')
