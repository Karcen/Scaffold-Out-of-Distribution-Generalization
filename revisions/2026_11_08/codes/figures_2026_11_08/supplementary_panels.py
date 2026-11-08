"""Redraw four supplementary figures from saved records; no models are fitted."""
import os
from pathlib import Path
import hashlib, json, sys, subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'results/review_2026_10_14'
OUT=ROOT/'work/figure_redesign_2026_11_08'
DATA=ROOT/'results/figures_2026_11_08'
QA=ROOT/'provenance/figures_2026_11_08'
SKILL=Path(os.environ['NATURE_FIGURE_SKILL'])
sys.path.insert(0,str(SKILL/'scripts'))
from audit_panel_alignment import require_matplotlib_panel_alignment
for p in [OUT, DATA, QA]: p.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':7.5,'axes.labelsize':7.5,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':.6,'lines.linewidth':1.1,'lines.markersize':3.8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
COL={'scaffold':'#7C8389','descriptor':'#207F88','morgan':'#CE795D'}
REG=list(COL)
RN={'scaffold':'Scaffold','descriptor':'Descriptor','morgan':'Morgan'}
TASK=['BBB','HIA','CYP2D6','DILI','PPBR','Solubility']
METRIC={e:'auprc' if e=='CYP2D6' else 'mae' if e in ['PPBR','Solubility'] else 'auroc' for e in TASK}
ML={'auroc':'AUROC','auprc':'Average precision','mae':'MAE'}
REP={'physchem':'#527C9D','morgan':'#C97965'}
trace=[]

def load(rel):
    p=R/rel
    d=pd.read_csv(p,float_precision='round_trip')
    trace.append({'source':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':len(d),'columns':list(d.columns)})
    return d

def source(frame,name):
    frame.to_csv(DATA/(name+'.csv'),index=False)

def letter(ax,l):
    ax.set_gid(l)
    ax.text(0,1,l,transform=ax.transAxes+ScaledTranslation(-24/72,11/72,ax.figure.dpi_scale_trans),fontsize=9,fontweight='bold',ha='left',va='bottom')

def grid(ax,axis='x'):
    ax.grid(axis=axis,color='#E8EBED',lw=.5,zorder=0)
    ax.set_axisbelow(True)

def save(fig,name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=QA/(name+'.alignment.json'),overlay_svg=QA/(name+'.alignment.svg'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True,strict=True)
    fig.savefig(OUT/(name+'.pdf'),format='pdf')
    fig.savefig(OUT/(name+'.svg'),format='svg')
    fig.savefig(OUT/(name+'.png'),format='png',dpi=300)
    plt.close(fig)
    commands=[['audit_pdf_text.py',str(OUT/(name+'.pdf')),'--min-pt','7','--json'],['audit_figure_collisions.py',str(OUT/(name+'.pdf')),'--json-out',str(QA/(name+'.collision.json')),'--overlay-pdf',str(QA/(name+'.collision.pdf'))]]
    for command in commands:
        res=subprocess.run([sys.executable,str(SKILL/'scripts'/command[0]),*command[1:]],capture_output=True,text=True)
        suffix='text' if command[0]=='audit_pdf_text.py' else 'collision'
        (QA/(name+'.'+suffix+'.log')).write_text(res.stdout+res.stderr)
        if suffix=='text': (QA/(name+'.text.json')).write_text(res.stdout)
        print(name,suffix,res.returncode)

def support():
    name='review_atom_pair_support'
    d=load('composition/independent_support_molecules.csv.gz')
    d.to_csv(DATA/(name+'_records.csv.gz'),index=False,compression='gzip')
    fig,axes=plt.subplots(2,3,figsize=(6.6929133858,136/25.4))
    fig.subplots_adjust(left=.10,right=.98,bottom=.15,top=.85,wspace=.47,hspace=.67)
    summary=[]
    for k,(ax,e) in enumerate(zip(axes.flat,TASK)):
        for reg,ls in [('scaffold','--'),('descriptor','-'),('morgan',':')]:
            q=d[(d.endpoint==e)&(d.regime==reg)]
            for seed,z in q.groupby('seed'):
                v=np.sort(z.atom_pair_max_training_tanimoto.to_numpy())
                ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',color=COL[reg],ls=ls,lw=.5,alpha=.18)
            v=np.sort(q.atom_pair_max_training_tanimoto.to_numpy())
            ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',color=COL[reg],ls=ls,lw=1.5)
            summary.append({'endpoint':e,'regime':reg,'pooled_records':len(q),'seeds':q.seed.nunique()})
        ax.set_title(e,loc='left',pad=9)
        ax.set_xlim(0,1);ax.set_ylim(0,1.02)
        ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1]);grid(ax,'both')
        if k%3==0: ax.set_ylabel('Cumulative fraction')
        if k>=3: ax.set_xlabel('Maximum training Tanimoto')
        letter(ax,'abcdef'[k])
    fig.legend(handles=[Line2D([],[],color=COL[r],ls={'scaffold':'--','descriptor':'-','morgan':':'}[r],label={'scaffold':'Scaffold control','descriptor':'Descriptor frontier','morgan':'Morgan frontier'}[r]) for r in REG],loc='lower center',ncol=3,bbox_to_anchor=(.5,.025),frameon=False,columnspacing=1.5)
    fig.text(.10,.98,'Independent atom-pair support',weight='bold',fontsize=9,va='top')
    fig.text(.10,.938,'Heavy lines: pooled records    Light lines: individual allocation seeds',fontsize=7.5,va='top',color='#555555')
    source(pd.DataFrame(summary),name+'_summary');save(fig,name)

def curation():
    name='review_curation'
    s=load('sensitivities/runs.csv')
    b=load('benchmark_uncertainty/model_intervals.csv')
    b=b[(b.scheme=='cluster')&(b.subset=='all')]
    fig,axes=plt.subplots(2,3,figsize=(6.6929133858,150/25.4))
    fig.subplots_adjust(left=.17,right=.98,bottom=.17,top=.90,wspace=.45,hspace=.61)
    records=[];selected=[]
    for k,(ax,e) in enumerate(zip(axes.flat,TASK)):
        for j,(rep,model) in enumerate([('physchem','linear'),('morgan','mlp')]):
            modelid='physchem:ridge' if model=='linear' and e in ['PPBR','Solubility'] else 'physchem:logistic' if model=='linear' else 'morgan:mlp'
            q=s[(s.endpoint==e)&(s.variant=='standardised_input')&(s.representation==rep)&s.model.isin(['logistic','ridge'] if model=='linear' else ['mlp'])]
            selected.append(q)
            for i,reg in enumerate(REG):
                old=b[(b.endpoint==e)&(b.model==modelid)&(b.metric==METRIC[e])&(b.regime==reg)].iloc[0]['mean']
                z=q[q.regime==reg]
                val=z[METRIC[e]].mean();y=i*2.6+j*.75
                ax.plot([old,val],[y,y],color=COL[reg],lw=1.5,zorder=2)
                ax.scatter(z[METRIC[e]],y+(z.seed-z.seed.mean())*.047,s=7,color=COL[reg],alpha=.35,zorder=3,linewidths=0)
                marker='o' if j==0 else '^'
                ax.plot(old,y,marker,mfc='white',mec=COL[reg],ms=4.5,zorder=4)
                ax.plot(val,y,marker,color=COL[reg],ms=4.5,zorder=5)
                records.append({'endpoint':e,'regime':reg,'model':modelid,'metric':METRIC[e],'primary_mean':old,'standardised_mean':val,'standardised_records':len(z)})
        ax.set_yticks([.375,2.975,5.575],[RN[r] for r in REG] if k%3==0 else [])
        ax.set_ylim(6.45,-.8)
        for y in [1.7,4.3]:ax.axhline(y,color='#E3E6E8',lw=.5)
        ax.set_xlabel(ML[METRIC[e]]+('\n(percentage points)' if e=='PPBR' else '\n(log molar)' if e=='Solubility' else ''))
        ax.set_title(e,loc='left',pad=9);letter(ax,'abcdef'[k]);grid(ax)
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=5)
    fig.text(.17,.98,'Curation changes the complete evaluation pipeline',weight='bold',fontsize=9,va='top')
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#555555',label='Physchem linear'),Line2D([],[],marker='^',ls='',color='#555555',label='Morgan MLP'),Line2D([],[],marker='o',ls='',mfc='white',mec='#555555',label='Primary mean'),Line2D([],[],marker='o',ls='',color='#555555',label='Standardised mean')],ncol=2,loc='lower center',bbox_to_anchor=(.5,.025),columnspacing=2.3,frameon=False)
    source(pd.DataFrame(records),name+'_summary');source(pd.concat(selected),name+'_standardised_records');save(fig,name)

def geometry():
    name='review_geometry_sensitivity'
    s=load('sensitivities/runs.csv');s=s[s.variant!='standardised_input']
    variants=['morgan_svd16','morgan_svd64','descriptor_k5','descriptor_k20','morgan_k5','morgan_k20','descriptor_radius25','descriptor_radius75','morgan_radius25','morgan_radius75','descriptor_no_size','atom_pair_svd32']
    labels=['Morgan SVD 16','Morgan SVD 64','Descriptor k = 5','Descriptor k = 20','Morgan k = 5','Morgan k = 20','Descriptor w = 0.25','Descriptor w = 0.75','Morgan w = 0.25','Morgan w = 0.75','Remove size','Atom-pair SVD 32']
    fig,axes=plt.subplots(2,3,figsize=(6.6929133858,180/25.4))
    fig.subplots_adjust(left=.255,right=.98,bottom=.12,top=.91,wspace=.42,hspace=.40)
    summaries=[]
    for k,(ax,e) in enumerate(zip(axes.flat,TASK)):
        for lo,hi in [(-.5,1.5),(5.5,9.5)]:ax.axhspan(lo,hi,color='#F3F5F6',zorder=0)
        for y in [1.5,5.5,9.5]:ax.axhline(y,color='#D6DCE0',lw=.6)
        for rep in ['physchem','morgan']:
            for i,v in enumerate(variants):
                q=s[(s.endpoint==e)&(s.variant==v)&(s.representation==rep)]
                assert len(q)==5,(e,v,rep,len(q))
                val=q[METRIC[e]].mean();y=i+(-.14 if rep=='physchem' else .14)
                ax.scatter(q[METRIC[e]],y+(q.seed-q.seed.mean())*.02,s=5,color=REP[rep],alpha=.27,linewidths=0,zorder=2)
                ax.plot(val,y,'o' if rep=='physchem' else '^',color=REP[rep],ms=3.6,zorder=3)
                summaries.append({'endpoint':e,'variant':v,'representation':rep,'metric':METRIC[e],'mean':val,'n':len(q)})
        ax.set_yticks(range(len(variants)),labels if k%3==0 else [])
        ax.set_ylim(len(variants)-.45,-.6)
        ax.set_title(e,loc='left',pad=9)
        ax.set_xlabel(ML[METRIC[e]]+('\n(percentage points)' if e=='PPBR' else '\n(log molar)' if e=='Solubility' else ''))
        ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=6);grid(ax)
        letter(ax,'abcdef'[k])
    fig.text(.255,.98,'Sensitivity to the allocation geometry',weight='bold',fontsize=9,va='top')
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=REP['physchem'],label='Physchem linear'),Line2D([],[],marker='^',ls='',color=REP['morgan'],label='Morgan linear')],loc='lower center',ncol=2,bbox_to_anchor=(.6,.025),frameon=False)
    source(s,name+'_records');source(pd.DataFrame(summaries),name+'_summary');save(fig,name)

def bbb():
    name='review_bbb_diagnostics'
    a=load('knn_matched_1024/descriptor_intervals.csv')
    g=load('knn_matched_1024/mlp_group_burden_mean.csv').sort_values('fraction_total_error',ascending=False).reset_index(drop=True)
    b=load('benchmark_uncertainty/model_intervals.csv')
    b=b[(b.endpoint=='BBB')&(b.subset=='cyclic')&(b.scheme=='cluster')&(b.metric=='auroc')&b.model.isin(['physchem:logistic','morgan:mlp'])]
    fig,axes=plt.subplots(2,2,figsize=(6.6929133858,165/25.4))
    fig.subplots_adjust(left=.25,right=.97,bottom=.12,top=.90,wspace=.44,hspace=.62)
    ax=axes[0,0]
    models=['morgan_mlp','knn5_weighted','knn5_unweighted','knn1_weighted','knn1_unweighted']
    labels=['Morgan MLP','Weighted kNN','Unweighted kNN','Single weighted','Single unweighted']
    for j,model in enumerate(models):
        for k,scheme in enumerate(['compound','cluster']):
            row=a[(a.scheme==scheme)&(a.model==model)].iloc[0];val=row.mean_auroc
            color=COL['morgan'] if model=='morgan_mlp' else COL['descriptor']
            ax.errorbar(val,j+(k-.5)*.25,xerr=[[max(0,val-row.lower)],[max(0,row.upper-val)]],fmt='o' if k==0 else 's',mfc='white' if k==0 else color,mec=color,ecolor=color,ms=3.8,capsize=2,zorder=3)
    ax.set_yticks(range(5),labels);ax.set_ylim(4.7,-.6);ax.set_xlim(.2,.95);ax.set_xticks([.25,.5,.75]);ax.set_xlabel('AUROC');ax.set_title('Matched Morgan inputs',loc='left',pad=10);ax.axvline(.5,color='#AAB2B7',ls='--',lw=.7);grid(ax)
    ax.legend(handles=[Line2D([],[],marker='o',ls='',mfc='white',mec='#555555',label='Compound'),Line2D([],[],marker='s',ls='',color='#555555',label='Group')],loc='upper center',bbox_to_anchor=(.5,-.25),ncol=2,fontsize=7,frameon=False,handlelength=1,borderpad=0,labelspacing=.3)
    ax=axes[0,1]
    for i,reg in enumerate(REG):
        for j,model in enumerate(['physchem:logistic','morgan:mlp']):
            row=b[(b.model==model)&(b.regime==reg)].iloc[0];val=row['mean'];y=i+(j-.5)*.26
            ax.errorbar(val,y,xerr=[[max(0,val-row.lower)],[max(0,row.upper-val)]],fmt='o' if j==0 else '^',color=COL[reg],capsize=2,ms=3.8)
    ax.set_yticks(range(3),['Scaf.','Desc.','Morgan']);ax.set_ylim(2.6,-.6);ax.set_xlim(.2,1.02);ax.set_xticks([.25,.5,.75,1]);ax.set_xlabel('Cyclic-only AUROC');ax.set_title('Fixed models, cyclic subset',loc='left',pad=10);ax.axvline(.5,color='#AAB2B7',ls='--',lw=.7);grid(ax)
    ax.legend(handles=[Line2D([],[],marker='o',ls='',color='#555555',label='Physchem LR'),Line2D([],[],marker='^',ls='',color='#555555',label='Morgan MLP')],loc='upper center',bbox_to_anchor=(.5,-.25),ncol=2,frameon=False,fontsize=7,handlelength=1,borderpad=0,labelspacing=.3)
    ax=axes[1,0];top=g.iloc[:5]
    yl=['Acyclic cell 2','Acyclic cell 1','Cyclic group 3','Cyclic group 4','Cyclic group 5']
    ax.barh(range(5),top.fraction_total_error.to_numpy()*100,color=[COL['morgan'],COL['morgan'],'#9BA9AF','#9BA9AF','#9BA9AF'],height=.58)
    ax.set_yticks(range(5),yl);ax.set_ylim(4.7,-.6);ax.set_xlabel('AUROC loss credit (%)');ax.set_title('Largest group burdens',loc='left',pad=10);grid(ax)
    ax=axes[1,1];cum=np.cumsum(g.fraction_total_error.to_numpy())*100
    ax.step(np.arange(1,len(g)+1),cum,where='post',color='#777777',lw=1.4)
    acyclic=g.group_id.str.startswith('__ACYCLIC').to_numpy();ax.scatter(np.arange(1,len(g)+1)[acyclic],cum[acyclic],s=20,color=COL['morgan'],zorder=3)
    ax.set_xlim(0,len(g));ax.set_ylim(0,103);ax.set_xticks([0,100,200]);ax.set_yticks([0,50,100]);ax.set_xlabel('Groups ordered by burden');ax.set_ylabel('Cumulative loss credit (%)');ax.set_title('Error concentration',loc='left',pad=10);grid(ax,'both')
    ax.legend(handles=[Line2D([],[],color='#777777',label='All groups'),Line2D([],[],marker='o',ls='',color=COL['morgan'],label='Acyclic cells')],loc='lower right',frameon=False,fontsize=7,handlelength=1.5)
    for ax,l in zip(axes.flat,'abcd'):letter(ax,l)
    fig.text(.25,.98,'BBB: model failure and retained label information',weight='bold',fontsize=9,va='top')
    source(a,name+'_matched_intervals');source(b,name+'_cyclic_intervals');source(g.assign(cumulative_fraction=np.cumsum(g.fraction_total_error)),name+'_group_burdens');save(fig,name)

CONTRACT={
 'common':{'backend':'Python / matplotlib','reuse':'Build anew from existing saved-result definitions; source measurements unchanged','export':{'width_mm':170,'max_height_mm':180,'min_font_pt':7,'formats':['PDF','SVG','PNG'],'png_dpi':300,'svg_text':'editable','pdf_fonttype':42},'uncertainty':'No new fitting, resampling, tests or CI estimates. Existing bootstrap limits retained where used. Seed records are descriptive allocation/fitting records, not independent chemical domains.','integrity':'No images or generated/demo values. All selections and aggregation recorded below.'},
 'review_atom_pair_support':{'question':'Does geometry-dependent training support persist in an independent representation?','claim':'Independent atom-pair similarities expose endpoint-specific differences between the two frontiers and scaffold control.','archetype':'quantitative grid','panels':{l:{'endpoint':e,'role':'Cross-endpoint orthogonal validation','display':'Pooled-record ECDF with every seed ECDF visible as a light trace'} for l,e in zip('abcdef',TASK)},'mapping':'atom_pair_max_training_tanimoto -> x; sorted record rank / record count -> cumulative fraction; regime -> hue and linestyle','selection':'All81628 saved rows, no exclusion, all six endpoints and all three regimes. Repeated test molecules remain repeated records; seed traces and pooled traces are descriptive, without intervals.'},
 'review_curation':{'question':'How do primary and fully standardised input pipelines compare?','claim':'The curation sensitivity changes scores across endpoint, model and allocation combinations rather than isolating one processing step.','archetype':'quantitative grid','panels':{l:{'endpoint':e,'role':'Complete-pipeline stress test','display':'Paired primary and standardised means; individual standardised seed records'} for l,e in zip('abcdef',TASK)},'mapping':'primary model_intervals cluster/all primary-metric mean -> open marker; standardised_input runs grouped by endpoint/representation/regime -> filled mean; every standardised seed -> small dot;physchem linear circle, Morgan MLP triangle','selection':'All180 standardised records for these two models, six endpoints,three regimes,five seeds. Primary interval table is used only for matching mean; omission of primary intervals is explicit because this descriptive panel compares mean pipelines and does not estimate paired processing effects.'},
 'review_geometry_sensitivity':{'question':'Which geometry choices alter frontier evaluation?','claim':'Alternative geometry settings produce endpoint- and model-dependent primary-metric profiles.','archetype':'quantitative grid','panels':{l:{'endpoint':e,'role':'Grouped allocation perturbation','display':'All12 variants with two linear models; every saved seed and arithmetic mean'} for l,e in zip('abcdef',TASK)},'mapping':'variant -> ordered categorical y; primary metric -> x; representation -> colour+marker; shaded bands group SVD/radius families; no interpolation between categorical variants','selection':'All720 non-standardised geometry records,12variants,two models,six endpoints,five seeds. All seed dots retained, no inferential intervals calculated.'},
 'review_bbb_diagnostics':{'question':'Does the BBB MLP failure imply absence of local label information, and which groups concentrate error?','claim':'Matched nearest-neighbour rules retain ranking information while Morgan-MLP loss is concentrated in a subset of groups; cyclic restriction does not remove the model comparison.','archetype':'quantitative grid','panels':{'a':{'role':'Matched-feature control','display':'All five saved models × compound/group bootstrap intervals'},'b':{'role':'Failure-boundary test','display':'Existing cyclic-subset group intervals for two models across three regimes'},'c':{'role':'Localise error','display':'Same five highest saved mean error-burden groups as prior figure'},'d':{'role':'Bound representative selection','display':'Cumulative curve of all206 groups sorted by saved mean burden; all acyclic cells highlighted'}},'mapping':'Saved interval mean/lower/upper used unchanged. Pair-error share×100 -> panelc; sorted cumulative sum of all fractions×100 -> paneld. No new uncertainty estimated.','selection':'Matched intervals all10rows retained (prior figure showed8); cyclic intervals select BBB/cyclic/cluster/AUROC and two statedmodels (6rows); top5burdens in c; all206groups in d. Models and split manifests remain fixed.'}
}
CAPTIONS={
 'review_atom_pair_support':'Independent atom-pair support across endpoint collections. Panels a–f show the empirical cumulative distribution of each test record’s maximum atom-pair Tanimoto similarity to its own training partition. Heavy curves pool all saved allocation-seed records; light curves show each seed separately. Repeated test molecules contribute repeated records. A leftward curve indicates lower training similarity in this representation, which is separate from both primary allocation geometries. These descriptive curves are not independent-domain uncertainty or a label-shift test.',
 'review_curation':'Primary and standardised-input evaluation outcomes. Panels a–f compare the complete input pipelines across endpoint collections. Open markers show primary means; filled markers show fully standardised-input means, and small points show each standardised-input seed record. Circles denote physicochemical linear models and triangles Morgan MLPs. Colours identify the scaffold control, descriptor frontier and Morgan frontier. Lines connect complete pipelines that jointly change duplicate aggregation, features, grouping and allocations. Regression retains its endpoint units; displayed seed variation does not isolate a causal effect of curation.',
 'review_geometry_sensitivity':'Geometry sensitivity across endpoint collections. Panels a–f retain every geometry variant for physicochemical and Morgan linear models. Large markers show arithmetic means; light points show the individual seed records. Rows group changes to Morgan SVD dimension, neighbour count k, radius weight w, size-descriptor removal and atom-pair coordinates. Primary molecules, grouping identities and linear recipes stay fixed while allocations change. No line interpolates between the categorical variants. Regression retains its endpoint units. Full variant definitions and configurations accompany the result registry.',
 'review_bbb_diagnostics':'BBB label information and group-level ranking errors. a, Matched Morgan-input comparisons show every nearest-neighbour rule and the MLP under compound (open circles) and whole-group (filled squares) resampling; whiskers are the existing conditional percentile intervals. b, Fixed-model cyclic-subset scores retain the same exclusion rule across all primary regimes, with group intervals; circles denote physicochemical LR and triangles Morgan MLP. Dashed lines indicate chance AUROC. c, The five groups with the largest mean error burdens. Each inverted positive–negative pair assigns half its loss credit to each molecule before group aggregation. d, Cumulative loss credit over all groups ordered by burden; coloured points identify acyclic cells. The curves describe conditional ranking errors, without refitting models or allocating new chemical domains. Full group identities and neighbour labels accompany the diagnostic tables.'
}
if __name__=='__main__':
    (QA/'supplementary_contracts.json').write_text(json.dumps(CONTRACT,indent=2)+'\n')
    (QA/'supplementary_caption_proposals.json').write_text(json.dumps(CAPTIONS,indent=2,ensure_ascii=False)+'\n')
    for fun in [support,curation,geometry,bbb]: fun()
    (QA/'supplementary_source_trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    preflight=subprocess.run([sys.executable,str(SKILL/'scripts/validate_figure.py'),__file__,'--json'],capture_output=True,text=True)
    (QA/'supplementary_source_preflight.json').write_text(preflight.stdout)
