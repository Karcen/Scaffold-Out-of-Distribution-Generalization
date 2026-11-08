"""Conditional Chemprop intervals on the unchanged benchmark bootstrap weights."""
from pathlib import Path
import hashlib, importlib.util, json, os, sys
# The sibling statistics.py is an analysis script, not Python's stdlib module.
SCRIPT_DIRECTORY=Path(__file__).resolve().parent
sys.path[:]=[p for p in sys.path if Path(p or os.getcwd()).resolve()!=SCRIPT_DIRECTORY]
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'results/revision/experiments'
CP=ROOT/'results/review_2026_10_14/chemprop'
BENCH=ROOT/'results/review_2026_10_14/benchmark_uncertainty'
OUT=ROOT/'results/review_2026_10_14/chemprop_uncertainty'
PROTOCOL=ROOT/'docs/review_2026_10_14/CHEMPROP_UNCERTAINTY_PROTOCOL.md'
TASKS=['BBB','HIA','CYP2D6','DILI','PPBR','Solubility']
REGIMES=['scaffold','descriptor','morgan'];B=2000;MODEL='chemprop:dmpnn'
SOURCES={}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source(p):
 p=Path(p);assert p.is_file(),p
 SOURCES[str(p.relative_to(ROOT))]=sha(p);return p
def read(p):return pd.read_csv(source(p),float_precision='round_trip')
def load(p):return json.loads(source(p).read_text())
def save(name,rows):
 frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
 frame.to_csv(OUT/name,index=False,compression={'method':'gzip','mtime':0} if name.endswith('.gz') else None)
def checked(p,expected):assert sha(source(p))==expected,(p,expected)

def main():
 OUT.mkdir(parents=True,exist_ok=True);source(PROTOCOL)
 helper=source(ROOT/'codes/review_2026_10_14/benchmark_uncertainty.py')
 spec=importlib.util.spec_from_file_location('benchmark_score_helper',helper)
 scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
 sys.path.insert(0,str(ROOT/'codes/scaffold'))
 from src.metrics import evaluate
 source(ROOT/'codes/scaffold/src/metrics.py')
 runs=read(CP/'runs.csv')
 assert len(runs)==90 and runs.run_id.is_unique and runs.status.eq('complete').all()
 actual=set(runs[['endpoint','regime','seed']].itertuples(index=False,name=None))
 assert actual=={(t,r,s) for t in TASKS for r in REGIMES for s in range(5)}
 bench_config=load(BENCH/'config.json');bench_hashes=load(BENCH/'sha256.json')
 assert bench_config['bootstrap_replicates']==B and bench_config['seed']==20261014
 checked(helper,bench_config['script_sha256'])
 summaries=read(BENCH/'model_intervals.csv')
 output=[];frontier=[];pairs=[];invalid=[];validation=[];compositions=[]
 for endpoint in TASKS:
  data=read(OLD/endpoint/'dataset.csv.gz');groups=np.load(source(OLD/endpoint/'features.npz'))['groups'].astype(str)
  assert np.array_equal(data.row_index,np.arange(len(data))) and data.connectivity_key.is_unique
  assert len(groups)==len(data)
  binary=endpoint not in ['PPBR','Solubility'];fits=[]
  parsed=load(CP/endpoint/'parse_audit.json');assert parsed['n']==len(data) and parsed['invalid']==[]
  graph=load(CP/endpoint/'graph_config.json');graph_rows=read(CP/endpoint/'graph_audit.csv')
  assert graph['n']==len(data) and graph['dataset_sha256']==sha(OLD/endpoint/'dataset.csv.gz')
  assert np.array_equal(graph_rows.row_index,data.row_index)
  assert graph['cuik_featurization'] is False and graph['recanonicalisation_or_deduplication'] is False
  for row in runs[runs.endpoint==endpoint].itertuples():
   folder=CP/endpoint/'runs'/row.run_id
   config=load(folder/'config.json');summary=load(folder/'summary.json')
   assert config['run_id']==row.run_id and summary['status']=='complete'
   for key in ['endpoint','regime','seed','model','representation','recipe','primary_metric']:
    assert config[key]==summary[key]==getattr(row,key)
   assert config['model']=='chemprop_dmpnn' and config['recipe']=='chemprop_2_3_1_cpu_fixed'
   for key in ['graph_arrays_sha256','dataset_sha256','n','rdkit','chemprop','graph_shapes','extra_descriptors','cuik_featurization','recanonicalisation_or_deduplication']:
    assert config['graph_metadata'][key]==graph[key],(row.run_id,key)
   checked(ROOT/config['dataset'],config['dataset_sha256'])
   checked(ROOT/config['source_dataset'],config['dataset_sha256'])
   checked(ROOT/config['split_manifest'],config['split_sha256'])
   checked(ROOT/config['source_split_manifest'],config['split_sha256'])
   checked(CP/'code_snapshots'/f"{config['script_sha256']}.py",config['script_sha256'])
   checked(CP/'protocol_snapshots'/f"{config['protocol_sha256']}.md",config['protocol_sha256'])
   checked(CP/'environment.json',config['environment_sha256'])
   checked(ROOT/'codes/scaffold/src/metrics.py',config['metrics_source_sha256'])
   checked(folder/'config.json',summary['config_sha256'])
   checked(folder/'predictions.csv.gz',summary['predictions_sha256'])
   checked(folder/'history.csv',summary['history_sha256'])
   checked(ROOT/summary['best_checkpoint'],summary['best_checkpoint_sha256'])
   manifest=read(ROOT/config['split_manifest']);pred=read(folder/'predictions.csv.gz')
   assert np.array_equal(manifest.row_index,data.row_index)
   assert np.array_equal(manifest.group.to_numpy(str),groups)
   assert manifest.groupby('group').partition.nunique().max()==1
   ids=pred.row_index.to_numpy(int);test=np.flatnonzero(manifest.partition=='test')
   assert np.array_equal(ids,test) and not pred.row_index.duplicated().any()
   for key in ['connectivity_key','smiles']:
    assert np.array_equal(pred[key].to_numpy(str),data[key].iloc[ids].to_numpy(str))
   assert np.array_equal(pred.group.to_numpy(str),groups[ids])
   assert np.allclose(pred.y_true,data.y.iloc[ids],atol=1e-10,rtol=0)
   # Recompute from the exact retained per-fit target values. Re-parsing the
   # canonical source can change floating-point ties at machine precision.
   # Identity remains checked above against the unchanged source dataset.
   metrics=evaluate(config['task_kind'],pred.y_true.to_numpy(float),pred.prediction.to_numpy(float))
   errors={key:abs(value-summary[key]) for key,value in metrics.items() if np.isfinite(value)}
   assert all(v<1e-6 for v in errors.values()),(row.run_id,errors)
   history=read(folder/'history.csv');assert summary['selected_epoch'] in set(history.epoch)
   assert len(history)==summary['epochs']
   train=np.flatnonzero(manifest.partition=='train')
   median=float(data.y.iloc[train].median()) if not binary else np.nan
   validation.append({'run_id':row.run_id,'endpoint':endpoint,'regime':row.regime,'seed':row.seed,'primary_metric':row.primary_metric,'saved_primary':summary[row.primary_metric],'recomputed_primary':metrics[row.primary_metric],'max_metric_difference':max(errors.values(),default=0.),'max_saved_target_source_difference':float(np.max(np.abs(pred.y_true.to_numpy(float)-data.y.iloc[ids].to_numpy(float)))),'dataset_split_identity_groups_snapshots_checkpoint_passed':True})
   fits.append(dict(run_id=row.run_id,regime=row.regime,seed=row.seed,row_ids=ids,y=pred.y_true.to_numpy(float),p=pred.prediction.to_numpy(float),median=median))
  for subset in (['all','cyclic'] if endpoint in ['BBB','DILI'] else ['all']):
   ws=source(BENCH/f'{endpoint}_{subset}_resampling_weights.npz')
   checked(ws,bench_hashes[str(ws.relative_to(ROOT))]);weights=np.load(ws)
   union=weights['row_ids'];compound=weights['compound_weights'];group_ids=weights['group_ids'].astype(str);cluster=weights['cluster_group_counts']
   assert compound.shape==(B,len(union)) and cluster.shape==(B,len(group_ids))
   ixmap={int(row):i for i,row in enumerate(union)};gmap={g:i for i,g in enumerate(group_ids)}
   group_ix=np.array([gmap[g] for g in groups[union]])
   subset_fits=[];members={};points={}
   for fit in fits:
    fit=fit.copy()
    if subset=='cyclic':
     keep=~np.char.startswith(groups[fit['row_ids']],'__ACYCLIC')
     for key in ['row_ids','y','p']:fit[key]=fit[key][keep]
    assert len(fit['row_ids']) and set(fit['row_ids'])<=set(union)
    fit['union_ix']=np.array([ixmap[int(row)] for row in fit['row_ids']])
    point,*_=scoring.scores(fit['y'],fit['p'],np.ones((1,len(fit['y']))),binary,fit['median'])
    members.setdefault(fit['regime'],[]).append(fit)
    for metric,value in point.items():points.setdefault((fit['regime'],metric),[]).append(value[0])
    compositions.append({'endpoint':endpoint,'subset':subset,'run_id':fit['run_id'],'regime':fit['regime'],'seed':fit['seed'],'model':MODEL,'test_n':len(fit['y']),'test_groups':len(np.unique(groups[fit['row_ids']])),'training_median':fit['median']})
    subset_fits.append(fit)
   assert all(len(v)==5 for v in members.values())
   for scheme in ['compound','cluster']:
    per_fit={}
    for fit in subset_fits:
     metric_names=['auroc','auprc'] if binary else ['mae','baseline_mae','mae_skill']
     result={key:np.empty(B) for key in metric_names};counts=np.zeros(4,int)
     for begin in range(0,B,100):
      end=min(begin+100,B);ix=fit['union_ix']
      w=compound[begin:end,ix] if scheme=='compound' else cluster[begin:end,group_ix[ix]]
      values,*flags=scoring.scores(fit['y'],fit['p'],w,binary,fit['median'])
      for key,value in values.items():result[key][begin:end]=value
      counts+=np.array([f.sum() for f in flags])
     per_fit[fit['run_id']]=result
     for metric,value in result.items():
      invalid.append({'endpoint':endpoint,'subset':subset,'scheme':scheme,'run_id':fit['run_id'],'regime':fit['regime'],'model':MODEL,'seed':fit['seed'],'metric':metric,'valid_draws':int(np.isfinite(value).sum()),'invalid_draws':int((~np.isfinite(value)).sum()),'empty_draws':int(counts[0]),'missing_positive_draws':int(counts[1]),'missing_negative_draws':int(counts[2]),'zero_baseline_mae_draws':int(counts[3])})
    draws={(regime,metric):np.mean([per_fit[f['run_id']][metric] for f in members[regime]],axis=0) for regime,metric in points}
    save(f'{endpoint}_{subset}_{scheme}_draws.csv.gz',pd.DataFrame({'replicate':np.arange(B),**{f'{r}:{MODEL}:{m}':a for (r,m),a in draws.items()}}))
    old_drawpath=source(BENCH/f'{endpoint}_{subset}_{scheme}_draws.csv.gz');checked(old_drawpath,bench_hashes[str(old_drawpath.relative_to(ROOT))]);old_draws=read(old_drawpath)
    assert np.array_equal(old_draws.replicate,np.arange(B))
    old_points=summaries[(summaries.endpoint==endpoint)&(summaries.subset==subset)&(summaries.scheme==scheme)]
    for (regime,metric),arr in draws.items():
     ci,n=scoring.intervals(arr);point=np.array(points[(regime,metric)])
     output.append({'endpoint':endpoint,'subset':subset,'scheme':scheme,'regime':regime,'model':MODEL,'metric':metric,'mean':point.mean(),'seed_sd':point.std(ddof=1),'lower':ci[0],'upper':ci[1],'valid_draws':n,'invalid_draws':B-n})
     if regime in ['descriptor','morgan']:
      delta=arr-draws[('scaffold',metric)];ci,n=scoring.intervals(delta)
      frontier.append({'endpoint':endpoint,'subset':subset,'scheme':scheme,'frontier':regime,'model':MODEL,'metric':metric,'delta_frontier_minus_scaffold':point.mean()-np.mean(points[('scaffold',metric)]),'lower':ci[0],'upper':ci[1],'valid_draws':n,'invalid_draws':B-n,'multiplicity_adjusted':False})
     if metric=='baseline_mae':continue
     candidates=old_points[(old_points.regime==regime)&(old_points.metric==metric)]
     for row in candidates.itertuples():
      delta=arr-old_draws[f'{regime}:{row.model}:{metric}'].to_numpy(float);ci,n=scoring.intervals(delta)
      pairs.append({'endpoint':endpoint,'subset':subset,'scheme':scheme,'regime':regime,'metric':metric,'first_model':MODEL,'second_model':row.model,'delta_first_minus_second':point.mean()-row.mean,'lower':ci[0],'upper':ci[1],'valid_draws':n,'invalid_draws':B-n,'complete_added_exploratory_family_size':len(candidates),'multiplicity_adjusted':False})
    print(endpoint,subset,scheme,'done',flush=True)
 save('model_intervals.csv',output);save('frontier_minus_scaffold.csv',frontier);save('chemprop_minus_existing_models.csv',pairs);save('invalid_draws_by_fit.csv',invalid);save('point_and_identity_validation.csv',validation);save('fit_composition.csv',compositions)
 config={'bootstrap_replicates':B,'seed':20261014,'confidence_level':.95,'completed_source_fits':90,'model':MODEL,'weights':'Exact stored original benchmark endpoint/subset weights; no regeneration or new units','estimand':'Mean of five per-fit metrics, conditional on fitted models and realised populations; regression skill averaged within fits','csv_numeric_parser':'round_trip; exact retained per-fit y_true values reproduce saved score ties, with source-label differences independently retained in validation','intervals':'Primary unstratified whole-group percentile; secondary class-stratified compound; complete added exploratory within-regime contrasts unadjusted; no p-values','source_sha256':SOURCES,'script_sha256':sha(__file__)}
 (OUT/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 (OUT/'README.md').write_text('# Conditional Chemprop uncertainty\n\nSeparate postprocessing of 90 fixed Chemprop fits. Stored original benchmark weights are reused exactly; no model is refitted and original benchmark results are unchanged. Group intervals are primary and compound intervals are sensitivity results. All scores average per-fit metrics over five seeds. Regression skill uses the complete training-partition median separately in each fit. Invalid weighted draws are retained. Chemprop-minus-existing-model comparisons contain every available within-regime comparator and are exploratory, unadjusted contrasts.\n\n`config.json` pins source hashes and the protocol. `point_and_identity_validation.csv` reports independent input/metric/checkpoint validation. Each endpoint/subset/scheme draw file preserves the complete 2,000 score vectors. `sha256.json` identifies all outputs.\n')
 (OUT/'sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='sha256.json' and not p.name.startswith('._')},indent=2)+'\n')

if __name__=='__main__':main()
