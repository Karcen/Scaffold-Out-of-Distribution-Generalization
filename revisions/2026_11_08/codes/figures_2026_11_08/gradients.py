"""Difficulty and training-size evidence from completed runs; no model fitting.
Backend: Python. All selected input records are retained in exported source data.
"""
import os
from pathlib import Path
import hashlib,json,sys,subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
ROOT=Path(__file__).resolve().parents[2]
SKILL=Path(os.environ['NATURE_FIGURE_SKILL'])
sys.path.insert(0,str(SKILL/'scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment
OUT=ROOT/'work/figure_redesign_2026_11_08'; DATA=ROOT/'results/figures_2026_11_08'; QA=ROOT/'provenance/figures_2026_11_08'
for p in [OUT,DATA,QA]:p.mkdir(parents=True,exist_ok=True)
TASK=['BBB','HIA','CYP2D6','DILI','PPBR','Solubility']; REG=['scaffold','descriptor','morgan']
COL={'scaffold':'#7C8389','descriptor':'#207F88','morgan':'#CE795D'}
LAB={'scaffold':'Scaffold control','descriptor':'Descriptor frontier','morgan':'Morgan frontier'}
STYLE={'scaffold':('s','--'),'descriptor':('o','-'),'morgan':('^',':')}
METRIC={'BBB':'AUROC','HIA':'AUROC','DILI':'AUROC','CYP2D6':'Average precision','PPBR':'MAE\n(percentage points)','Solubility':'MAE\n(log molar)'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':7.5,'axes.labelsize':7,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1.1,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
SRC=ROOT/'results/revision/analysis/difficulty_distance_runs.csv'
d=pd.read_csv(SRC); q=d[d.regime.isin(REG)].copy()
assert q.groupby(['endpoint','regime','test_target']).size().eq(5).all()
q.to_csv(DATA/'gradient_all_runs.csv',index=False)
a=q.groupby(['endpoint','regime','test_target'],sort=False).agg(performance_mean=('performance','mean'),performance_sd=('performance','std'),distance_mean=('morgan_geometry_nn_mean','mean'),train_mean=('train_n','mean'),train_sd=('train_n','std')).reset_index()
a.to_csv(DATA/'gradient_display.csv',index=False)
contract={'question':'How does changing the allocation fraction alter performance, training-set size, and representation-specific distance?', 'claim':'Allocation gradients change performance together with training size and selected test populations; distance alone is not isolated.', 'archetype':'quantitative grid','backend':'python','reuse':'Build anew from existing scientific data and statistical definitions','size_mm':[170,180], 'exports':['pdf','svg','png'], 'glyph_floor_pt':7,'source':str(SRC.relative_to(ROOT)),'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'records_before':len(d),'records_after':len(q),'filter':'regime in scaffold, descriptor, morgan; these are the requested primary regimes; all endpoints, targets, seeds retained','aggregation':'Arithmetic mean and sample SD across five saved seed records in each endpoint/regime/test_target cell. No model refitting or bootstrap.','panels_gradient':'a,c,e,g,i,k: primary performance mean±sample seed SD; b,d,f,h,j,l: aligned mean training count±sample seed SD, exposing changing training size. All six endpoints.','panels_distance':'a–f: the same saved primary performance mean±sample seed SD against mean reduced-Morgan nearest-training distance; marker area identifies target test fraction; trajectories follow increasing target fraction, not sorted distance.','limits':'Seed SD is not an independent-domain CI. Regression units and distance scales remain endpoint-specific. Lines are guides through evaluated fractions, not interpolated estimates.'}
(QA/'gradient_contract.json').write_text(json.dumps(contract,indent=2)+'\n')

def label(ax,l):
 ax.text(0,1,l,transform=ax.transAxes+ScaledTranslation(-42/72,7/72,ax.figure.dpi_scale_trans),fontweight='bold',fontsize=9,va='bottom',ha='left')

def clean(ax):
 ax.tick_params(length=2.5,pad=2);ax.grid(axis='y',color='#E9EBED',lw=.45,zorder=0)

def save(fig,stem,rows,columns,notes):
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,json_out=QA/f'{stem}.alignment.json',overlay_svg=QA/f'{stem}.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True,require_panel_labels=True,row_groups=rows,column_groups=columns)
 fig.savefig(OUT/f'{stem}.pdf')
 fig.savefig(OUT/f'{stem}.svg')
 fig.savefig(OUT/f'{stem}.png',dpi=300)
 plt.close(fig)
 for script,args,name in [('validate_figure.py',[str(Path(__file__).resolve()),'--json'],'source'),('audit_pdf_text.py',[str(OUT/f'{stem}.pdf'),'--min-pt','7','--json'],'text'),('audit_figure_collisions.py',[str(OUT/f'{stem}.pdf'),'--json-out',str(QA/f'{stem}.collision.json'),'--overlay-pdf',str(QA/f'{stem}.collision.pdf')],'collision_stdout')]:
  p=subprocess.run([sys.executable,str(SKILL/'scripts'/script),*args],capture_output=True,text=True)
  (QA/f'{stem}.{name}.txt').write_text(p.stdout+p.stderr)
  if p.returncode:raise RuntimeError(stem+' '+script+' '+str(p.returncode))
 (QA/f'{stem}.qa_notes.json').write_text(json.dumps(notes,indent=2)+'\n')

def gradient():
 fig=plt.figure(figsize=(170/25.4,180/25.4));gs=fig.add_gridspec(3,2,left=.14,right=.97,bottom=.12,top=.91,wspace=.58,hspace=.80)
 rows=[];cols=[[],[],[],[]]; axes=[];labels=[]
 for i,task in enumerate(TASK):
  r,c=divmod(i,2);inner=gs[r,c].subgridspec(2,1,height_ratios=[3,1],hspace=.18);top=fig.add_subplot(inner[0]);bottom=fig.add_subplot(inner[1],sharex=top); lt,lb='abcdefghijkl'[2*i:2*i+2]; labels.extend([lt,lb]);axes.extend([top,bottom]);cols[c*2].append(lt);cols[c*2+1].append(lb);label(top,lt);label(bottom,lb)
  for reg in REG:
   v=a[(a.endpoint==task)&(a.regime==reg)].sort_values('test_target');x=v.test_target.to_numpy()*100;y=v.performance_mean.to_numpy();sd=v.performance_sd.to_numpy(); marker,ls=STYLE[reg]
   top.fill_between(x,y-sd,y+sd,color=COL[reg],alpha=.10,lw=0)
   top.errorbar(x,y,yerr=sd,fmt=marker,ls=ls,color=COL[reg],markersize=3,capsize=1.5,lw=1)
   bottom.errorbar(x,v.train_mean/1000,yerr=v.train_sd/1000,fmt=marker,ls=ls,color=COL[reg],markersize=2.5,capsize=1,lw=.8)
  top.set_title(task,loc='left',pad=9,fontweight='bold');top.set_ylabel(METRIC[task],labelpad=4);top.tick_params(labelbottom=False);top.set_xticks([10,20,30,40]);clean(top);clean(bottom)
  bottom.set_ylabel('Train n\n(×1,000)',labelpad=10,rotation=0,ha='right',va='center');bottom.set_xlabel('Target test fraction (%)',labelpad=3);bottom.set_xlim(7,43);bottom.set_xticks([10,20,30,40]);bottom.yaxis.set_major_locator(MaxNLocator(nbins=1));top.yaxis.set_major_locator(MaxNLocator(nbins=3));top.margins(y=.12)
 for r in range(3):rows.extend([[labels[4*r],labels[4*r+2]],[labels[4*r+1],labels[4*r+3]]])
 fig.legend(handles=[Line2D([],[],color=COL[r],marker=STYLE[r][0],ls=STYLE[r][1],label=LAB[r]) for r in REG],loc='lower center',bbox_to_anchor=(.5,.025),ncol=3,frameon=False,columnspacing=1.2,handlelength=2)
 fig.text(.14,.985,'Performance and the changing training pool',fontsize=9,fontweight='bold',va='top')
 save(fig,'fig3_gradient',rows,cols,{'uncertainty':'Every performance and training-count mean has sample seed SD; bands and bars are the same SD, not separate uncertainty estimates. Zero SD bars can be invisible.','intentional_overlap':'Transparent SD ribbons overlay their own mean trajectories. Adjacent endpoint panels use distinct native metric scales.','source_data':'results/figures_2026_11_08/gradient_all_runs.csv; gradient_display.csv'})

def distance():
 fig,axes=plt.subplots(3,2,figsize=(170/25.4,171/25.4));fig.subplots_adjust(left=.125,right=.97,bottom=.17,top=.93,wspace=.58,hspace=.68)
 for ax,task,l in zip(axes.flat,TASK,'abcdef'):
  for reg in REG:
   v=a[(a.endpoint==task)&(a.regime==reg)].sort_values('test_target');marker,ls=STYLE[reg];x=v.distance_mean.to_numpy();y=v.performance_mean.to_numpy()
   # Preserve target-fraction traversal even where distance is non-monotone.
   ax.errorbar(x,y,yerr=v.performance_sd.to_numpy(),fmt='none',ecolor=COL[reg],capsize=1.5,lw=.7)
   ax.plot(x,y,ls=ls,color=COL[reg],lw=1)
   for xx,yy,t in zip(x,y,v.test_target):ax.scatter([xx],[yy],s=12+90*t,marker=marker,color=COL[reg],zorder=3,linewidth=.6,edgecolors='white')
  ax.set_title(task,loc='left',fontweight='bold',pad=8);ax.set_xlabel('Mean nearest-training distance',labelpad=4);ax.set_ylabel(METRIC[task],labelpad=4);label(ax,l);clean(ax);ax.locator_params(axis='x',nbins=4)
 fig.text(.125,.985,'The gradient in reduced-Morgan distance space',fontsize=9,fontweight='bold',va='top')
 fig.legend(handles=[Line2D([],[],color=COL[r],marker=STYLE[r][0],ls=STYLE[r][1],label=LAB[r]) for r in REG],loc='lower center',bbox_to_anchor=(.5,.075),ncol=3,frameon=False,columnspacing=1.2)
 fig.legend(handles=[Line2D([],[],ls='',color='#7C8389',marker='o',markersize=np.sqrt(12+90*t),label=f'{int(t*100)}%') for t in [.1,.2,.3,.4]],title='Marker area: target test fraction',title_fontsize=7,loc='lower center',bbox_to_anchor=(.5,.005),ncol=4,frameon=False,columnspacing=1)
 save(fig,'supp_distance',[['a','b'],['c','d'],['e','f']],[['a','c','e'],['b','d','f']],{'uncertainty':'Mean and sample seed SD of the same five saved fits per point; not independent-domain confidence intervals.','trajectory':'Lines follow increasing test-target fraction and can backtrack in distance; no smoothing or interpolation. Marker area also encodes target fraction.','source_data':'results/figures_2026_11_08/gradient_all_runs.csv; gradient_display.csv'})
if __name__=='__main__':gradient();distance();print('Exported both gradient figures with fresh audits')
