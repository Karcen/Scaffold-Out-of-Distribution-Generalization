"""Joint cyclic-only compound/group resampling of saved DILI predictions."""
from pathlib import Path
import hashlib, itertools, json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[2]
EXP=ROOT/'results/revision/experiments'
OUT=ROOT/'results/review_2026_10_14/cyclic_grid'
B=2000; SEED=20261014
REGIMES=['random','scaffold','descriptor','morgan']
MODELS=[f'{rep}:{model}' for rep in ['physchem','maccs','morgan','combined'] for model in ['logistic','mlp']]
SOURCES={}

def source(path):
 path=Path(path);SOURCES[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest();return path
def csv(path):return pd.read_csv(source(path))
def save(name,rows):
 frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
 frame.to_csv(OUT/name,index=False,compression={'method':'gzip','mtime':0} if name.endswith('.gz') else None)
def weighted_auc(y,p,w):
 order=np.argsort(-p,kind='mergesort');yp=y[order];ps=p[order]
 starts=np.r_[0,np.flatnonzero(np.diff(ps)!=0)+1]
 pos=np.add.reduceat(w[:,order]*yp,starts,axis=1)
 neg=np.add.reduceat(w[:,order]*(1-yp),starts,axis=1)
 cp=pos.sum(axis=1);cn=neg.sum(axis=1)
 numerator=np.sum(pos*(cn[:,None]-np.cumsum(neg,axis=1)+.5*neg),axis=1)
 score=np.divide(numerator,cp*cn,out=np.full(len(w),np.nan),where=(cp>0)&(cn>0))
 return score,((cp==0)&(cn==0)),(((cp==0)|(cn==0))&((cp+cn)>0))
def percentile(arr):
 a=arr[np.isfinite(arr)];return (np.quantile(a,[.025,.975]).tolist() if len(a) else [np.nan,np.nan]),len(a)
def simultaneous(matrix,point):
 """A complete exploratory family; maximum standardised centred deviation."""
 valid=np.isfinite(matrix).all(axis=1);a=matrix[valid]
 if len(a)<2:return np.full(len(point),np.nan),np.nan,int(valid.sum())
 sd=a.std(axis=0,ddof=1);nonzero=sd>1e-12
 assert np.all(np.abs(a[:,~nonzero]-point[~nonzero])<1e-10)
 maxima=np.max(np.abs((a[:,nonzero]-point[nonzero])/sd[nonzero]),axis=1) if nonzero.any() else np.zeros(len(a))
 critical=float(np.quantile(maxima,.95))
 return critical*sd,critical,int(valid.sum())

def same_display_check(runs):
 rows=[];raw=[]
 h=runs[(runs.endpoint=='HIA')&(runs.regime=='morgan')&(runs.test_target==.2)&(runs.representation=='morgan')&(runs.model=='mlp')&(runs.recipe=='common')]
 assert len(h)==5
 hia=[]
 for r in h.itertuples():
  p=csv(EXP/'HIA/runs'/r.run_id/'predictions.csv.gz');v=roc_auc_score(p.y_true,p.prediction)
  assert abs(v-r.auroc)<1e-6;hia.append(v);raw.append({'cell':'HIA Morgan-frontier MLP','seed':r.seed,'metric':'auroc','recomputed':v,'saved':r.auroc})
 gm=csv(ROOT/'results/scaffold/results/gnn_metrics.csv')
 selected=gm[(gm.task=='CYP2D6')&(gm.split=='frontier')&(gm.method=='ERM')&(gm.partition=='test')&(gm.metric=='auprc')]
 assert len(selected)==5
 cyp=[]
 for r in selected.itertuples():
  p=csv(ROOT/'results/scaffold/results/gnn_runs'/r.run_id/'predictions.csv.gz');p=p[p.partition=='test']
  v=average_precision_score(p.y_true,p.prediction);assert abs(v-r.value)<1e-6;cyp.append(v)
  raw.append({'cell':'CYP2D6 descriptor-frontier GNN','seed':r.seed,'metric':'auprc','recomputed':v,'saved':r.value})
 for cell,metric,vals in [('HIA Morgan-frontier MLP','auroc',hia),('CYP2D6 descriptor-frontier GNN','auprc',cyp)]:
  rows.append({'cell':cell,'metric':metric,'mean':np.mean(vals),'sample_seed_sd':np.std(vals,ddof=1),'display_mean':f'{np.mean(vals):.3f}','display_sd':f'{np.std(vals,ddof=1):.3f}','fits':len(vals)})
 save('same_display_check.csv',rows);save('same_display_per_seed.csv',raw)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 runs=csv(EXP/'runs.csv')
 dataset=csv(EXP/'DILI/dataset.csv.gz')
 features=np.load(source(EXP/'DILI/features.npz'));groups=features['groups']
 assert len(dataset)==len(groups) and np.array_equal(dataset.row_index,np.arange(len(dataset)))
 selected=runs[(runs.endpoint=='DILI')&(runs.recipe=='common')&(runs.test_target==.2)&(runs.regime.isin(REGIMES))]
 assert len(selected)==200
 prepared=[]
 for r in selected.itertuples():
  p=csv(EXP/'DILI/runs'/r.run_id/'predictions.csv.gz')
  manifest=csv(EXP/'DILI/splits'/f'{r.regime}_test20_seed{r.seed}.csv.gz')
  assert np.array_equal(manifest.group.to_numpy(str),groups.astype(str))
  assert set(p.row_index)==set(manifest.loc[manifest.partition=='test','row_index'])
  assert not p.row_index.duplicated().any()
  assert np.array_equal(dataset.set_index('row_index').loc[p.row_index,'y'],p.y_true)
  point=roc_auc_score(p.y_true,p.prediction);assert abs(point-r.auroc)<1e-6
  p=p[~np.char.startswith(groups[p.row_index.to_numpy(int)].astype(str),'__ACYCLIC')].copy()
  assert len(p)>0 and p.y_true.nunique()==2
  point=roc_auc_score(p.y_true,p.prediction)
  family='architecture' if r.representation=='architecture' else 'grid'
  model=r.model if family=='architecture' else f'{r.representation}:{r.model}'
  prepared.append({'row':r,'family':family,'model':model,'prediction':p,'point':point})
 ids=np.unique(np.concatenate([p['prediction'].row_index.to_numpy(int) for p in prepared]))
 index={int(v):i for i,v in enumerate(ids)};y=dataset.set_index('row_index').loc[ids,'y'].to_numpy(int)
 group_ids,compound_group_ix=np.unique(groups[ids].astype(str),return_inverse=True)
 save('resampling_units.csv',[{'row_index':int(v),'y':int(label),'group_id':str(group),'group_index':int(gi)} for v,label,group,gi in zip(ids,y,groups[ids],compound_group_ix)])
 rng=np.random.default_rng(SEED);compound=np.zeros((B,len(ids)),dtype=np.uint16)
 for label in [0,1]:
  ix=np.flatnonzero(y==label);compound[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=B)
 cluster_counts=rng.multinomial(len(group_ids),np.full(len(group_ids),1/len(group_ids)),size=B).astype(np.uint16)
 np.savez_compressed(OUT/'resampling_weights.npz',compound_ids=ids,compound_weights=compound,group_ids=group_ids,cluster_group_counts=cluster_counts)
 # Scoring validation includes ties and a zero-weight class.
 ty=np.array([0,1,0,1]);tp=np.array([.1,.7,.7,.4]);tw=np.array([[1,2,3,1],[1,0,2,0]],float)
 ts,_,single=weighted_auc(ty,tp,tw)
 assert abs(ts[0]-roc_auc_score(ty,tp,sample_weight=tw[0]))<1e-12 and np.isnan(ts[1]) and single[1]
 points={};membership={}
 for item in prepared:
  key=(item['family'],item['row'].regime,item['model'])
  membership.setdefault(key,[]).append(item);points.setdefault(key,[]).append(item['point'])
 points={key:np.mean(value) for key,value in points.items()}
 assert all(len(v)==5 for v in membership.values())
 interval_rows=[];top_rows=[];tau_rows=[];pair_rows=[];invalid_rows=[];family_rows=[]
 for scheme,w in [('compound',compound),('cluster',cluster_counts[:,compound_group_ix])]:
  fit_draws={}
  for item in prepared:
   p=item['prediction'];ix=np.array([index[int(v)] for v in p.row_index])
   arr,empty,single=weighted_auc(p.y_true.to_numpy(int),p.prediction.to_numpy(float),w[:,ix])
   fit_draws[item['row'].run_id]=arr
   invalid_rows.append({'scheme':scheme,'family':item['family'],'regime':item['row'].regime,'model':item['model'],'seed':item['row'].seed,'run_id':item['row'].run_id,'valid_draws':int(np.isfinite(arr).sum()),'empty_draws':int(empty.sum()),'single_class_draws':int(single.sum()),'test_n':len(p),'test_groups':len(np.unique(groups[p.row_index.to_numpy(int)]))})
  draws={key:np.mean([fit_draws[item['row'].run_id] for item in members],axis=0) for key,members in membership.items()}
  save(f'{scheme}_auroc_draws.csv.gz',pd.DataFrame({'replicate':np.arange(B),**{':'.join(key):a for key,a in draws.items()}}))
  for key,a in draws.items():
   ci,n=percentile(a);interval_rows.append({'scheme':scheme,'family':key[0],'regime':key[1],'model':key[2],'mean_auroc':points[key],'lower':ci[0],'upper':ci[1],'valid_draws':n,'invalid_draws':B-n})
  grid_arrays={reg:np.column_stack([draws[('grid',reg,model)] for model in MODELS]) for reg in REGIMES}
  for reg,a in grid_arrays.items():
   ok=np.isfinite(a).all(axis=1);aa=a[ok];best=aa.max(axis=1);tied=np.abs(aa-best[:,None])<=1e-12;credit=tied/tied.sum(axis=1)[:,None]
   ranks=np.array([rankdata(-row,method='average') for row in aa])
   for i,model in enumerate(MODELS):top_rows.append({'scheme':scheme,'regime':reg,'model':model,'point_auroc':points[('grid',reg,model)],'point_rank':rankdata(-np.array([points[('grid',reg,m)] for m in MODELS]),method='average')[i],'top1_frequency':credit[:,i].mean(),'mean_bootstrap_rank':ranks[:,i].mean(),'valid_draws':int(ok.sum()),'invalid_draws':int((~ok).sum()),'tied_top_draws':int((tied.sum(axis=1)>1).sum())})
  for first,second in itertools.combinations(REGIMES,2):
   a,b=grid_arrays[first],grid_arrays[second];ok=np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1)
   tau=np.array([kendalltau(x,z,variant='b').statistic for x,z in zip(a[ok],b[ok])]);ci,n=percentile(tau)
   tau_rows.append({'scheme':scheme,'first_regime':first,'second_regime':second,'point_tau_b':kendalltau([points[('grid',first,m)] for m in MODELS],[points[('grid',second,m)] for m in MODELS],variant='b').statistic,'lower':ci[0],'upper':ci[1],'valid_draws':n,'invalid_draws':B-n})
   save(f'{scheme}_tau_{first}_{second}_draws.csv.gz',pd.DataFrame({'replicate':np.flatnonzero(ok),'tau_b':tau}))
  for family,models in [('grid',MODELS),('architecture',['allmlp','moltrinet'])]:
   local={};all_columns=[];all_points=[]
   for reg in REGIMES:
    pairs=list(itertools.combinations(models,2));matrix=np.column_stack([draws[(family,reg,a)]-draws[(family,reg,b)] for a,b in pairs]);point=np.array([points[(family,reg,a)]-points[(family,reg,b)] for a,b in pairs]);local[reg]=(pairs,matrix,point);all_columns.append(matrix);all_points.extend(point)
   global_radius,global_crit,global_n=simultaneous(np.column_stack(all_columns),np.array(all_points));offset=0
   family_rows.append({'scheme':scheme,'family':family,'scope':'all_four_regimes','contrasts':len(all_points),'critical_max_standardised_deviation':global_crit,'valid_draws':global_n,'invalid_draws':B-global_n})
   for reg,(pairs,matrix,point) in local.items():
    radius,crit,n=simultaneous(matrix,point);family_rows.append({'scheme':scheme,'family':family,'scope':reg,'contrasts':len(pairs),'critical_max_standardised_deviation':crit,'valid_draws':n,'invalid_draws':B-n})
    for j,(first,second) in enumerate(pairs):
     ci,pair_n=percentile(matrix[:,j]);pair_rows.append({'scheme':scheme,'family':family,'regime':reg,'first_model':first,'second_model':second,'delta_first_minus_second':point[j],'percentile_lower':ci[0],'percentile_upper':ci[1],'pair_valid_draws':pair_n,'regime_simultaneous_lower':point[j]-radius[j],'regime_simultaneous_upper':point[j]+radius[j],'regime_family_contrasts':len(pairs),'regime_family_valid_draws':n,'global_simultaneous_lower':point[j]-global_radius[offset+j],'global_simultaneous_upper':point[j]+global_radius[offset+j],'global_family_contrasts':len(all_points),'global_family_valid_draws':global_n})
    offset+=len(pairs)
  print('Completed',scheme,'bootstrap',flush=True)
 save('model_intervals.csv',interval_rows);save('top1_frequency.csv',top_rows);save('kendall_tau.csv',tau_rows);save('pairwise_and_simultaneous_intervals.csv',pair_rows);save('invalid_draws_by_fit.csv',invalid_rows);save('multiplicity_families.csv',family_rows)
 config={'evaluation_subset':'cyclic-only test compounds; fixed group IDs beginning __ACYCLIC excluded; original training sets/models retained; union resampling restricted to retained compounds/groups','bootstrap_replicates':B,'seed':SEED,'confidence_level':.95,'estimand':'mean of five per-seed AUROCs conditional on fitted models, labels and realised partitions','union_test_compounds':len(ids),'union_test_groups':len(group_ids),'compound_resampling':'class-stratified multinomial union-ID weights shared across all models, seeds and four regimes','cluster_resampling':'unstratified multinomial fixed group-ID weights over observed test union, shared across all models, seeds and four regimes; all compounds in a group share multiplicity','invalid_policy':'empty/single-class fit AUROC undefined; model mean requires all five fits valid; every family/rank analysis conditions on its complete finite draws and reports count','top1_tie_tolerance':1e-12,'top1_ties':'fractional 1/k credit across tied maxima','rank_correlation':'Kendall tau-b of the same eight models in each regime, percentile intervals','multiplicity':'full exploratory 28-contrast family within each grid regime and 112-contrast family across all four regimes; bootstrap max of standardised centred deviations, 95% quantile; architecture family contains four contrasts across regimes','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_sha256':SOURCES,'fit_ids':selected.run_id.tolist(),'interpretation':'conditional sensitivity to observed test compounds/clusters; no training or partition resampling, independent-domain inference, causal geometry effect or post-selection confirmatory testing'}
 (OUT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 (OUT/'sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='sha256.json' and not p.name.startswith('._')},indent=2)+'\n')
 print('Output:',OUT,flush=True)

if __name__=='__main__':main()
