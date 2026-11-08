"""Conditional joint uncertainty for completed classical/MLP/GNN benchmarks."""
from pathlib import Path
import hashlib,itertools,json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'results/revision/experiments'
BASE=ROOT/'results/review_2026_10_14/baselines'
OUT=ROOT/'results/review_2026_10_14/benchmark_uncertainty'
PROTOCOL=ROOT/'docs/review_2026_10_14/BENCHMARK_UNCERTAINTY_PROTOCOL.md'
B=2000;SEED=20261014;TASKS=['BBB','HIA','CYP2D6','DILI','PPBR','Solubility']
REGIMES=['scaffold','descriptor','morgan'];SOURCES={}
def source(path):
 path=Path(path);SOURCES[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest();return path
def read(path):return pd.read_csv(source(path))
def save(name,rows):
 a=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
 a.to_csv(OUT/name,index=False,compression={'method':'gzip','mtime':0} if name.endswith('.gz') else None)
def intervals(arr):
 valid=np.isfinite(arr);return np.quantile(arr[valid],[.025,.975]).tolist() if valid.any() else [np.nan,np.nan],int(valid.sum())
def scores(y,p,w,binary,median):
 total=w.sum(axis=1);empty=total==0
 if binary:
  order=np.argsort(-p,kind='mergesort');starts=np.r_[0,np.flatnonzero(np.diff(p[order])!=0)+1];label=y[order]
  pos=np.add.reduceat(w[:,order]*label,starts,axis=1);neg=np.add.reduceat(w[:,order]*(1-label),starts,axis=1)
  cp=np.cumsum(pos,axis=1);cn=np.cumsum(neg,axis=1);pt=cp[:,-1];nt=cn[:,-1]
  auroc=np.divide(np.sum(pos*(nt[:,None]-cn+.5*neg),axis=1),pt*nt,out=np.full(len(w),np.nan),where=(pt>0)&(nt>0))
  precision=np.divide(cp,cp+cn,out=np.zeros_like(cp),where=(cp+cn)>0)
  ap=np.divide(np.sum(pos*precision,axis=1),pt,out=np.full(len(w),np.nan),where=pt>0)
  return {'auroc':auroc,'auprc':ap},empty,(pt==0)&~empty,(nt==0)&~empty,np.zeros(len(w),bool)
 error=np.abs(y-p);baseline_error=np.abs(y-median)
 mae=np.divide(w@error,total,out=np.full(len(w),np.nan),where=~empty)
 baseline=np.divide(w@baseline_error,total,out=np.full(len(w),np.nan),where=~empty)
 ratio=np.divide(mae,baseline,out=np.full(len(w),np.nan),where=baseline>0)
 return {'mae':mae,'baseline_mae':baseline,'mae_skill':1-ratio},empty,np.zeros(len(w),bool),np.zeros(len(w),bool),(baseline==0)&~empty

def load_endpoint(endpoint,base_runs,old_runs,gnn_metrics):
 data=read(OLD/endpoint/'dataset.csv.gz');assert np.array_equal(data.row_index,np.arange(len(data)))
 groups=np.load(source(OLD/endpoint/'features.npz'))['groups'].astype(str)
 binary=endpoint not in ['PPBR','Solubility'];new=base_runs[base_runs.endpoint==endpoint]
 assert len(new)==(60 if binary else 75) and new.status.eq('complete').all(),(endpoint,len(new))
 manifests={};fits=[]
 for regime in REGIMES:
  for seed in range(5):
   manifest=read(OLD/endpoint/'splits'/f'{regime}_test20_seed{seed}.csv.gz')
   assert np.array_equal(manifest.group.to_numpy(str),groups)
   manifests[(regime,seed)]=manifest
 for rows,folder,prefix in [(new,BASE,'classical'),(old_runs[(old_runs.endpoint==endpoint)&(old_runs.recipe=='common')&(old_runs.test_target==.2)&(old_runs.regime.isin(REGIMES))&(old_runs.representation=='morgan')&(old_runs.model=='mlp')],OLD,'benchmark')]:
  for row in rows.itertuples():
   pred=read(folder/endpoint/'runs'/row.run_id/'predictions.csv.gz')
   config_path=source(folder/endpoint/'runs'/row.run_id/'config.json')
   config=json.loads(config_path.read_text())
   split_path=source(ROOT/config['split_manifest']);assert SOURCES[str(split_path.relative_to(ROOT))]==config['split_sha256']
   dataset_path=source(folder/endpoint/'dataset.csv.gz');assert SOURCES[str(dataset_path.relative_to(ROOT))]==config['dataset_sha256']
   snapshot=BASE/'code_snapshots'/f"{config['script_sha256']}.py" if prefix=='classical' else ROOT/'provenance/revision/code_snapshots'/f"{config['script_sha256']}.py"
   snapshot=source(snapshot);assert SOURCES[str(snapshot.relative_to(ROOT))]==config['script_sha256']
   if prefix=='classical':
    protocol_snapshot=source(ROOT/config['protocol_snapshot']) if 'protocol_snapshot' in config else source(BASE/'protocol_snapshots'/f"{config['protocol_sha256']}.md")
    assert SOURCES[str(protocol_snapshot.relative_to(ROOT))]==config['protocol_sha256']
    feature_path=source(folder/endpoint/'features.npz');assert SOURCES[str(feature_path.relative_to(ROOT))]==config['feature_cache_sha256']
    assert hashlib.sha256(config_path.read_bytes()).hexdigest()==row.config_sha256
    assert hashlib.sha256((folder/endpoint/'runs'/row.run_id/'predictions.csv.gz').read_bytes()).hexdigest()==row.predictions_sha256
   manifest=manifests[(row.regime,row.seed)];assert set(pred.row_index)==set(manifest.loc[manifest.partition=='test','row_index'])
   assert not pred.row_index.duplicated().any() and np.allclose(data.y.iloc[pred.row_index],pred.y_true,atol=2e-6,rtol=0)
   median=float(data.y[manifest.partition=='train'].median()) if not binary else np.nan
   model=f'{row.representation}:{row.model}'
   if prefix=='benchmark':model='morgan:mlp'
   fits.append({'run_id':row.run_id,'regime':row.regime,'seed':row.seed,'model':model,'row_ids':pred.row_index.to_numpy(int),'y':data.y.iloc[pred.row_index].to_numpy(float),'p':pred.prediction.to_numpy(float),'median':median,'saved_primary':getattr(row,row.primary_metric),'primary_metric':row.primary_metric,'source_family':prefix})
 # Preserve GNN as its own recipe. Exact partition/identity checks justify the
 # canonical row-ID and fixed-group mapping; no Morgan GNN fit is imputed.
 for regime,old_regime in [('scaffold','scaffold_clustered_matched'),('descriptor','frontier')]:
  candidates=[]
  for seed in range(5):
   run_id=f'{endpoint}__{old_regime}__seed{seed}__ERM__gine3x128'
   path=ROOT/'results/scaffold/results/gnn_runs'/run_id/'predictions.csv.gz'
   if not path.exists():continue
   pred=read(path);source(path.with_name('config.json'));manifest=manifests[(regime,seed)]
   for partition in ['val','test']:
    rows=pred[pred.partition==partition];assert set(rows.row_index)==set(manifest.loc[manifest.partition==partition,'row_index'])
    assert np.array_equal(rows.smiles.to_numpy(str),data.smiles.iloc[rows.row_index].to_numpy(str))
    assert np.allclose(rows.y_true,data.y.iloc[rows.row_index],atol=2e-6,rtol=0)
   test=pred[pred.partition=='test'];metric='mae' if not binary else 'auprc' if endpoint=='CYP2D6' else 'auroc'
   saved=gnn_metrics[(gnn_metrics.run_id==run_id)&(gnn_metrics.partition=='test')&(gnn_metrics.metric==metric)]
   assert len(saved)==1
   candidates.append({'run_id':run_id,'regime':regime,'seed':seed,'model':'graph:gnn','row_ids':test.row_index.to_numpy(int),'y':data.y.iloc[test.row_index].to_numpy(float),'p':test.prediction.to_numpy(float),'median':float(data.y[manifest.partition=='train'].median()) if not binary else np.nan,'saved_primary':saved.value.iloc[0],'primary_metric':metric,'source_family':'gnn'})
  assert len(candidates) in [0,5],(endpoint,regime,len(candidates));fits.extend(candidates)
 return data,groups,binary,fits

def main():
 OUT.mkdir(parents=True,exist_ok=True);source(PROTOCOL)
 base_runs=read(BASE/'runs.csv');old_runs=read(OLD/'runs.csv');gnn_metrics=read(ROOT/'results/scaffold/results/gnn_metrics.csv')
 assert len(base_runs)==390 and set(base_runs.endpoint)==set(TASKS),'Baseline run registry incomplete'
 # Independent weighted scoring checks, including tied scores and all positives.
 y=np.array([0.,1.,0.,1.]);p=np.array([.2,.8,.8,.5]);w=np.array([[1,2,3,1],[0,2,0,1]],float)
 actual,*_=scores(y,p,w,True,np.nan)
 assert abs(actual['auroc'][0]-roc_auc_score(y,p,sample_weight=w[0]))<1e-12
 assert np.allclose(actual['auprc'],[average_precision_score(y,p,sample_weight=z) for z in w])
 assert np.isnan(actual['auroc'][1]) and actual['auprc'][1]==1
 summary=[];frontier=[];pairs=[];invalid=[];composition=[];validation=[]
 for endpoint in TASKS:
  data,groups,binary,fits=load_endpoint(endpoint,base_runs,old_runs,gnn_metrics)
  full_fits=fits
  for subset in (['all','cyclic'] if endpoint in ['BBB','DILI'] else ['all']):
   fits=[]
   for original in full_fits:
    fit=original.copy()
    if subset=='cyclic':
     keep=~np.char.startswith(groups[fit['row_ids']],'__ACYCLIC')
     for key in ['row_ids','y','p']:fit[key]=fit[key][keep]
     assert len(fit['row_ids'])>0
    fits.append(fit)
   ids=np.unique(np.concatenate([fit['row_ids'] for fit in fits]));index={int(v):i for i,v in enumerate(ids)}
   labels=data.y.iloc[ids].to_numpy(float);group_ids,group_ix=np.unique(groups[ids],return_inverse=True)
   rng=np.random.default_rng(SEED);compound=np.zeros((B,len(ids)),np.uint16)
   if binary:
    for label in [0,1]:
     ix=np.flatnonzero(labels==label);compound[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=B)
   else:compound[:]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=B)
   cluster_counts=rng.multinomial(len(group_ids),np.full(len(group_ids),1/len(group_ids)),size=B).astype(np.uint16)
   np.savez_compressed(OUT/f'{endpoint}_{subset}_resampling_weights.npz',row_ids=ids,compound_weights=compound,group_ids=group_ids,cluster_group_counts=cluster_counts)
   save(f'{endpoint}_{subset}_resampling_units.csv',[{'row_index':int(row),'y':float(label),'group_id':groups[row]} for row,label in zip(ids,labels)])
   memberships={};point={}
   for fit in fits:
    ix=np.array([index[int(v)] for v in fit['row_ids']]);fit['union_ix']=ix
    observed,*_=scores(fit['y'],fit['p'],np.ones((1,len(ix))),binary,fit['median'])
    if subset=='all':assert abs(observed[fit['primary_metric']][0]-fit['saved_primary'])<2e-6,(fit['run_id'],observed,fit['saved_primary'])
    validation.append({'endpoint':endpoint,'subset':subset,'run_id':fit['run_id'],'primary_metric':fit['primary_metric'],'recomputed':observed[fit['primary_metric']][0],'saved':fit['saved_primary'] if subset=='all' else np.nan,'absolute_difference':abs(observed[fit['primary_metric']][0]-fit['saved_primary']) if subset=='all' else np.nan,'source_family':fit['source_family'],'identity_partition_group_mapping_passed':True})
    for metric,values in observed.items():
     key=(fit['regime'],fit['model'],metric);memberships.setdefault(key,[]).append(fit);point.setdefault(key,[]).append(values[0])
    composition.append({'endpoint':endpoint,'subset':subset,'run_id':fit['run_id'],'regime':fit['regime'],'model':fit['model'],'seed':fit['seed'],'test_n':len(ix),'test_groups':len(np.unique(groups[fit['row_ids']])),'test_y_mean':fit['y'].mean(),'test_y_sd':fit['y'].std(ddof=1),'training_median':fit['median']})
   assert all(len(members)==5 for members in memberships.values())
   for scheme in ['compound','cluster']:
    per_fit={}
    for fit in fits:
     metrics={metric:np.empty(B) for metric in (['auroc','auprc'] if binary else ['mae','baseline_mae','mae_skill'])}
     counters=np.zeros(4,int)
     for begin in range(0,B,100):
      end=min(begin+100,B);ix=fit['union_ix']
      weights=compound[begin:end,ix] if scheme=='compound' else cluster_counts[begin:end,group_ix[ix]]
      arr,empty,no_pos,no_neg,zero_baseline=scores(fit['y'],fit['p'],weights,binary,fit['median'])
      for metric,a in arr.items():metrics[metric][begin:end]=a
      counters+=np.array([empty.sum(),no_pos.sum(),no_neg.sum(),zero_baseline.sum()])
     per_fit[fit['run_id']]=metrics
     for metric,arr in metrics.items():invalid.append({'endpoint':endpoint,'subset':subset,'scheme':scheme,'run_id':fit['run_id'],'regime':fit['regime'],'model':fit['model'],'seed':fit['seed'],'metric':metric,'valid_draws':int(np.isfinite(arr).sum()),'invalid_draws':int((~np.isfinite(arr)).sum()),'empty_draws':int(counters[0]),'missing_positive_draws':int(counters[1]),'missing_negative_draws':int(counters[2]),'zero_baseline_mae_draws':int(counters[3])})
    draws={key:np.mean([per_fit[fit['run_id']][key[2]] for fit in members],axis=0) for key,members in memberships.items()}
    save(f'{endpoint}_{subset}_{scheme}_draws.csv.gz',pd.DataFrame({'replicate':np.arange(B),**{':'.join(key):arr for key,arr in draws.items()}}))
    for key,arr in draws.items():
     ci,n=intervals(arr);values=np.array(point[key]);summary.append({'endpoint':endpoint,'subset':subset,'scheme':scheme,'regime':key[0],'model':key[1],'metric':key[2],'mean':values.mean(),'seed_sd':values.std(ddof=1),'lower':ci[0],'upper':ci[1],'valid_draws':n,'invalid_draws':B-n})
     if key[0] in ['descriptor','morgan'] and ('scaffold',key[1],key[2]) in draws:
      control=('scaffold',key[1],key[2]);delta=arr-draws[control];ci,n=intervals(delta)
      frontier.append({'endpoint':endpoint,'subset':subset,'scheme':scheme,'frontier':key[0],'model':key[1],'metric':key[2],'delta_frontier_minus_scaffold':values.mean()-np.mean(point[control]),'lower':ci[0],'upper':ci[1],'valid_draws':n,'invalid_draws':B-n,'multiplicity_adjusted':False})
    for regime in REGIMES:
     for metric in (['auroc','auprc'] if binary else ['mae','mae_skill']):
      keys=[key for key in draws if key[0]==regime and key[2]==metric]
      family_size=len(keys)*(len(keys)-1)//2
      for first,second in itertools.combinations(keys,2):
       delta=draws[first]-draws[second];ci,n=intervals(delta)
       pairs.append({'endpoint':endpoint,'subset':subset,'scheme':scheme,'regime':regime,'metric':metric,'first_model':first[1],'second_model':second[1],'delta_first_minus_second':np.mean(point[first])-np.mean(point[second]),'lower':ci[0],'upper':ci[1],'valid_draws':n,'invalid_draws':B-n,'exploratory_family_size':family_size,'multiplicity_adjusted':False})
    print(endpoint,subset,scheme,'done',len(fits),'fits',flush=True)
 save('model_intervals.csv',summary);save('frontier_minus_scaffold.csv',frontier);save('all_model_pairs.csv',pairs);save('invalid_draws_by_fit.csv',invalid);save('fit_composition.csv',composition);save('point_and_identity_validation.csv',validation)
 config={'bootstrap_replicates':B,'seed':SEED,'confidence_level':.95,'estimand':'mean of five per-seed metrics conditional on fitted models and observed test partitions; MAE skill is mean per-seed 1-model_MAE/training_median_MAE','resampling':'weights shared by union compound/group identifier across every model and regime within endpoint; classification compound weights class-stratified; regression compound and all fixed-group weights unstratified','binary_degeneracy':'AUROC requires both classes; AUPRC requires positives and equals 1 for positive-only draws; empty/absent-class counts retained','intervals':'percentile 2.5/97.5, complete finite five-seed means; all frontier-minus-scaffold and within-regime model pairs exploratory and unadjusted; no p-values','regression_units':'original endpoint units kept separate; no cross-endpoint average','cyclic_subset':'BBB and DILI only; exclude test rows in fixed grouping IDs starting __ACYCLIC; preserve complete original training sets and medians; bootstrap each subset union separately','gnn_mapping':'included descriptor/scaffold only after every test/val row ID, SMILES and label matches canonical data and manifests','source_sha256':SOURCES,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (OUT/'config.json').write_text(json.dumps(config,indent=2)+'\n');(OUT/'sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='sha256.json' and not p.name.startswith('._')},indent=2)+'\n')

if __name__=='__main__':main()
