"""Evidence-led composition and DILI figures from saved outputs; no model fitting."""
import os
from pathlib import Path
import hashlib
import json
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'results/review_2026_10_14'
D = ROOT / 'results/figures_2026_11_08'
Q = ROOT / 'provenance/figures_2026_11_08'
F = ROOT / 'work/figure_redesign_2026_11_08'
SKILL = Path(os.environ['NATURE_FIGURE_SKILL'])
sys.path.insert(0, str(SKILL / 'scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment
for p in [D, Q, F]:
    p.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': .6, 'lines.linewidth': 1.15, 'lines.markersize': 4,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white',
})
REG = ['scaffold', 'descriptor', 'morgan']
COL = {'scaffold': '#7C8389', 'descriptor': '#207F88', 'morgan': '#CE795D'}
LAB = {'scaffold': 'Scaffold control', 'descriptor': 'Descriptor frontier', 'morgan': 'Morgan frontier'}
SHORT = ['Scaf.', 'Desc.', 'Morgan']
MODEL = ['physchem:logistic', 'physchem:mlp', 'maccs:logistic', 'maccs:mlp',
         'morgan:logistic', 'morgan:mlp', 'combined:logistic', 'combined:mlp']
MLABEL = ['Physchem LR', 'Physchem MLP', 'MACCS LR', 'MACCS MLP',
          'Morgan LR', 'Morgan MLP', 'Combined LR', 'Combined MLP']
MCOL = ['#3D7C9A', '#3D7C9A', '#168584', '#168584', '#CB6954', '#CB6954', '#9276A9', '#9276A9']
trace = []


def read(path):
    p = R / path
    df = pd.read_csv(p, float_precision='round_trip')
    trace.append({'source': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'source_rows': len(df)})
    return df


def panel(ax, letter, title):
    ax.set_gid(letter)
    ax.set_title(title, loc='left', pad=9, fontweight='normal')
    ax.text(0, 1, letter, transform=ax.transAxes + ScaledTranslation(-19/72, 8/72, ax.figure.dpi_scale_trans),
            fontsize=9, fontweight='bold', ha='left', va='bottom')
    ax.tick_params(length=2.5, width=.6, pad=3)


def ci(ax, x, y, lo, hi, color, marker='o', ms=4):
    ax.errorbar(x, y, xerr=[[max(0, x-lo)], [max(0, hi-x)]], fmt=marker,
                color=color, ecolor=color, capsize=2, ms=ms, lw=1, zorder=3)


def save(fig, stem, **groups):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=Q/(stem+'.alignment.json'),
        overlay_svg=Q/(stem+'.alignment.svg'), tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True, **groups)
    fig.savefig(F/(stem+'.pdf'), format='pdf')
    fig.savefig(F/(stem+'.svg'), format='svg')
    fig.savefig(F/(stem+'.png'), format='png', dpi=600)
    plt.close(fig)
    for name, command in [('text', [str(SKILL/'scripts/audit_pdf_text.py'), str(F/(stem+'.pdf')), '--min-pt', '7']),
                          ('collision', [str(SKILL/'scripts/audit_figure_collisions.py'), str(F/(stem+'.pdf')), '--json-out', str(Q/(stem+'.collision.json')), '--overlay-pdf', str(Q/(stem+'.collision.pdf'))])]:
        result = subprocess.run([sys.executable, *command], text=True, capture_output=True)
        (Q/(stem+'.'+name+'.txt')).write_text(result.stdout + result.stderr)
        result.check_returncode()


def composition():
    comp = read('composition/split_composition.csv')
    ints0 = read('benchmark_uncertainty/model_intervals.csv')
    ints = ints0[(ints0.scheme=='cluster') & (ints0.subset=='all') & (ints0.model=='morgan:mlp')].copy()
    mol0 = read('composition/independent_support_molecules.csv.gz')
    mol = mol0[mol0.endpoint.isin(['PPBR', 'Solubility'])].copy()
    mol.to_csv(D/'review_composition_ecdf_records.csv.gz', index=False)
    comp.to_csv(D/'review_composition_summary_records.csv', index=False)
    descriptive_points = []
    for endpoint in ['CYP2D6', 'PPBR', 'Solubility']:
        for reg in REG:
            rows = comp[(comp.endpoint==endpoint) & (comp.regime==reg)]
            for metric in ['positive_fraction'] if endpoint=='CYP2D6' else ['model_mae', 'median_baseline_mae']:
                descriptive_points.append(dict(endpoint=endpoint, regime=reg, metric=metric, mean=rows[metric].mean(), n_records=len(rows)))
    pd.DataFrame(descriptive_points).to_csv(D/'review_composition_descriptive_points.csv', index=False)
    ints.to_csv(D/'review_composition_model_intervals.csv', index=False)
    fig = plt.figure(figsize=(6.6929133858, 7.0078740157))
    # Mixed-span quantitative layout: distribution hero row, classification context,
    # then reference-predictor controls; comparable groups declared explicitly.
    axa = fig.add_axes([.115, .710, .355, .205])
    axb = fig.add_axes([.615, .710, .355, .205])
    axc = fig.add_axes([.115, .410, .205, .175])
    axd = fig.add_axes([.505, .410, .465, .175])
    axe = fig.add_axes([.115, .105, .190, .175])
    axf = fig.add_axes([.445, .105, .190, .175])
    axg = fig.add_axes([.775, .105, .195, .175])
    for ax, endpoint, letter in [(axa, 'PPBR', 'a'), (axb, 'Solubility', 'b')]:
        for reg, ls in zip(REG, ['--', '-', ':']):
            vals = np.sort(mol.loc[(mol.endpoint==endpoint) & (mol.regime==reg), 'y_true'].to_numpy())
            assert len(vals)>0 and np.isfinite(vals).all()
            ax.step(vals, np.arange(1, len(vals)+1)/len(vals), where='post', color=COL[reg], ls=ls, lw=1.4)
        ax.set_ylim(0, 1.03)
        ax.set_yticks([0, .5, 1])
        ax.set_xlabel('Test target (percentage points)' if endpoint=='PPBR' else 'Test target (log molar)')
        ax.set_ylabel('Cumulative fraction')
        if endpoint=='PPBR':
            ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
        panel(ax, letter, endpoint+' target population')
    for i, reg in enumerate(REG):
        frac = comp.loc[(comp.endpoint=='CYP2D6') & (comp.regime==reg), 'positive_fraction'].mean()
        axc.bar(i, 1-frac, bottom=frac, width=.58, color='#E8ECEC', edgecolor='white', lw=.3)
        axc.bar(i, frac, width=.58, color=COL[reg], edgecolor='white', lw=.3)
    axc.set_xticks(range(3), SHORT); axc.set_ylim(0,1); axc.set_yticks([0, .5, 1]); axc.set_ylabel('Test-label fraction')
    panel(axc, 'c', 'CYP2D6 class mix')
    axc.text(.5, -.36, 'Colour: positive; pale: negative', transform=axc.transAxes, ha='center', va='top', fontsize=7)
    for k, metric in enumerate(['auprc', 'auroc']):
        for j, reg in enumerate(REG):
            row = ints[(ints.endpoint=='CYP2D6') & (ints.metric==metric) & (ints.regime==reg)].iloc[0]
            ci(axd, row['mean'], k + (j-1)*.18, row.lower, row.upper, COL[reg], marker='o')
    axd.set_yticks([0,1], ['Average precision', 'AUROC']); axd.set_ylim(1.4,-.45)
    axd.set_xlim(.25,.9); axd.set_xticks([.3,.5,.7,.9]); axd.set_xlabel('Morgan-MLP score')
    panel(axd, 'd', 'CYP2D6 discrimination and precision')
    for ax, endpoint, letter in [(axe, 'PPBR', 'e'), (axf, 'Solubility', 'f')]:
        for i, reg in enumerate(REG):
            rows = comp[(comp.endpoint==endpoint) & (comp.regime==reg)]
            model = rows.model_mae.mean(); median = rows.median_baseline_mae.mean()
            ax.plot([model, median], [i,i], color=COL[reg], alpha=.7, lw=1.1)
            ax.plot(model, i, 'o', color=COL[reg], ms=4)
            ax.plot(median, i, 's', mfc='white', mec=COL[reg], ms=4)
        ax.set_yticks(range(3), SHORT); ax.set_ylim(2.6,-.6)
        ax.set_xlim(left=0); ax.set_xlabel('MAE (percentage points)' if endpoint=='PPBR' else 'MAE (log molar)')
        panel(ax, letter, endpoint+' reference error')
    skill_rows=[]
    for j, endpoint in enumerate(['PPBR','Solubility']):
        for i, reg in enumerate(REG):
            row = ints[(ints.endpoint==endpoint) & (ints.regime==reg) & (ints.metric=='mae_skill')].iloc[0]
            ci(axg, row['mean'], i+j*4, row.lower, row.upper, COL[reg], marker='o')
            skill_rows.append(row)
    axg.axvline(0, color='.68', ls='--', lw=.7, zorder=0)
    axg.set_yticks([0,1,2,4,5,6], ['PPBR: Scaf.','Desc.','Morgan','Sol.: Scaf.','Desc.','Morgan'])
    axg.set_ylim(6.7,-.7); axg.set_xlim(-.5,.5); axg.set_xticks([-.5,0,.5]); axg.set_xlabel('MAE skill')
    panel(axg, 'g', 'Baseline-relative skill')
    fig.legend(handles=[Line2D([],[],color=COL[r],ls=ls,label=LAB[r]) for r,ls in zip(REG,['--','-',':'])],
               loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.52,.992), handlelength=2.3, columnspacing=1.8)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color='.3',label='Morgan MLP'),
                        Line2D([],[],marker='s',ls='',mfc='white',mec='.3',label='Training median')],
               loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.37,.014))
    save(fig,'review_composition', row_groups=[['a','b'],['c','d'],['e','f','g']],
         column_groups=[['a','c','e']],
         exemptions=[{'panels':['a','c','e'], 'checks':['column'],
                      'reason':'Intentional mixed-span evidence hierarchy. Left plot edges and label anchors share a fixed x; widths differ for distributions, class mix, and reference controls.'},
                     {'panels':['g'], 'checks':['panel-width'], 'reason':'Skill forest uses a slightly narrower plot area to reserve space for endpoint and regime labels.'}])
    return {'ecdf_source_rows':len(mol0), 'ecdf_selected_rows':len(mol),
            'ecdf_filter':'endpoint in [PPBR, Solubility], all regimes, seeds and test records retained',
            'ecdf_counts':mol.groupby(['endpoint','regime']).size().to_dict(),
            'interval_source_rows':len(ints0), 'interval_selected_rows':len(ints),
            'interval_filter':'scheme == cluster & subset == all & model == morgan:mlp',
            'uncertainty':'Existing conditional 95% test-group bootstrap intervals, fitted models and partitions fixed',
            'transform':'ECDF from every saved selected test record; descriptive means from all saved fitting/allocation records. No smoothing, binning, sampling, or new inferential estimates.'}


def ranking():
    a0=read('statistics/model_intervals.csv'); t0=read('statistics/top1_frequency.csv')
    a=a0[(a0.scheme=='cluster') & (a0.family=='grid')].copy(); t=t0[t0.scheme=='cluster'].copy()
    a.to_csv(D/'review_dili_ranking_model_intervals.csv',index=False)
    t.to_csv(D/'review_dili_ranking_top1_frequency.csv',index=False)
    regs=['random','scaffold','descriptor','morgan']; rlabels=['Random','Scaffold','Descriptor','Morgan']
    fig=plt.figure(figsize=(6.6929133858,6.6535433071))
    axa=fig.add_axes([.105,.58,.34,.315]); axb=fig.add_axes([.665,.58,.305,.315])
    axc=fig.add_axes([.205,.115,.28,.315]); axd=fig.add_axes([.69,.115,.28,.315])
    for i,model in enumerate(MODEL):
        pts=[t[(t.model==model)&(t.regime==r)].point_rank.iloc[0] for r in regs]
        marker='o' if model.endswith('logistic') else '^'; ls='-' if marker=='o' else '--'
        axa.plot(range(4),pts,marker=marker,ls=ls,color=MCOL[i],ms=4,lw=1.15,alpha=.9)
    axa.set_yticks(range(1,9)); axa.set_ylim(8.5,.5); axa.set_xlim(-.15,3.15)
    axa.set_xticks(range(4),rlabels,rotation=35,ha='right',rotation_mode='anchor')
    axa.set_ylabel('Point rank (1 = highest AUROC)'); panel(axa,'a','Allocation-dependent point ordering')
    vals=np.array([[t[(t.model==m)&(t.regime==r)].top1_frequency.iloc[0]*100 for r in regs] for m in MODEL])
    cmap=LinearSegmentedColormap.from_list('frequency',['#F8FAFA','#C6DFDE','#168584'])
    axb.imshow(vals,cmap=cmap,vmin=0,vmax=100,aspect='auto',interpolation='nearest')
    axb.set_xticks(range(4),rlabels,rotation=35,ha='right',rotation_mode='anchor')
    axb.set_yticks(range(8),MLABEL); axb.tick_params(length=0)
    for (i,j),v in np.ndenumerate(vals):
        axb.text(j,i,format(v,'.1f'),ha='center',va='center',fontsize=7,color='white' if v>=65 else '#333333')
    for spine in axb.spines.values():spine.set_visible(False)
    panel(axb,'b','Top-rank frequency (%)')
    for ax,reg,letter in [(axc,'descriptor','c'),(axd,'morgan','d')]:
        for k in range(4):
            ax.axhspan(k*2-.5,k*2+1.5,color=MCOL[k*2],alpha=.055,zorder=0,lw=0)
        for i,model in enumerate(MODEL):
            row=a[(a.regime==reg)&(a.model==model)].iloc[0]
            ci(ax,row.mean_auroc,i,row.lower,row.upper,MCOL[i],marker='o' if model.endswith('logistic') else '^')
        ax.set_yticks(range(8),MLABEL); ax.set_ylim(7.6,-.6); ax.set_xlim(.3,1)
        ax.set_xticks([.3,.5,.7,.9]); ax.axvline(.5,color='.65',ls=':',lw=.8,zorder=0)
        ax.set_xlabel('AUROC'); panel(ax,letter,reg.capitalize()+' frontier')
    fig.legend(handles=[Line2D([],[],color=c,lw=1.8,label=l) for c,l in zip(MCOL[::2],['Physchem','MACCS','Morgan','Combined'])]
        +[Line2D([],[],color='.3',marker='o',ls='-',label='LR'),Line2D([],[],color='.3',marker='^',ls='--',label='MLP')],
        loc='upper center',ncol=6,frameon=False,bbox_to_anchor=(.52,.99),handlelength=1.6,columnspacing=1.0)
    save(fig,'review_dili_ranking',row_groups=[['a','b'],['c','d']],column_groups=[['a','c'],['b','d']],
         exemptions=[{'panels':['a','c','b','d'], 'checks':['column'],
                      'reason':'Overview row reserves separate model labels beside the heatmap; lower AUROC plots reserve labels on each forest. Heights align by row, and their widths are intentionally different by evidence role.'}])
    return {'model_interval_source_rows':len(a0),'model_interval_selected_rows':len(a),
            'interval_filter':'scheme == cluster & family == grid',
            'ranking_source_rows':len(t0),'ranking_selected_rows':len(t),
            'ranking_filter':'scheme == cluster; all eight models and all four regimes retained',
            'uncertainty':'Existing conditional 95% group-bootstrap intervals and joint top-rank frequencies; ties share winner credit. Point-rank paths are descriptive only.',
            'transform':'Model ordering and percentages read directly from saved rows; top1_frequency multiplied by 100 for display. No ranks or intervals recalculated.'}


if __name__=='__main__':
    c=composition();d=ranking()
    c['ecdf_counts']={str(k):v for k,v in c['ecdf_counts'].items()}
    (Q/'composition_ranking_sources.json').write_text(json.dumps({'sources':trace,'review_composition':c,'review_dili_ranking':d},indent=2)+'\n')
    import pymupdf
    from PIL import Image
    exports = {}
    for stem in ['review_composition', 'review_dili_ranking']:
        with pymupdf.open(F/(stem+'.pdf')) as document:
            page = document[0]
            dimensions = [page.rect.width*25.4/72, page.rect.height*25.4/72]
        with Image.open(F/(stem+'.png')) as raster:
            pixels = list(raster.size); dpi = raster.info.get('dpi')
        exports[stem] = {'width_mm':dimensions[0], 'height_mm':dimensions[1], 'png_pixels':pixels, 'png_dpi':dpi,
                         'svg_text_objects':(F/(stem+'.svg')).read_text().count('<text'),
                         'assets':{ext:hashlib.sha256((F/(stem+'.'+ext)).read_bytes()).hexdigest() for ext in ['pdf','svg','png']},
                         'alignment':json.loads((Q/(stem+'.alignment.json')).read_text())['verdict'],
                         'collision':json.loads((Q/(stem+'.collision.json')).read_text())['verdict']}
    (Q/'composition_ranking_export_checks.json').write_text(json.dumps(exports, indent=2)+'\n')
    print('Staged composition and DILI ranking figures; no final assets changed.')
