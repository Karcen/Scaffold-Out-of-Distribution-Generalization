from pathlib import Path
import sys,json,hashlib,concurrent.futures
import numpy as np,pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")
root=Path(__file__).resolve().parents[3];out=root/'results/review_2026_10_14/allocation_controls'
sys.path.insert(0,str(root/'codes/scaffold'));from src.metrics import evaluate
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source_code_sha=sha(out/'code_snapshot.py');helper_sha=sha(out/'baseline_helper_snapshot.py')
data={name:pd.read_csv(root/'results/revision/experiments'/name/'dataset.csv.gz') for name in ['BBB','HIA','CYP2D6','PPBR','Solubility','DILI']}
input_hashes={name:sha(root/'results/revision/experiments'/name/'dataset.csv.gz') for name in data}
group_arrays={}
for name in data:
 with np.load(root/'results/revision/experiments'/name/'features.npz',allow_pickle=False) as cache:group_arrays[name]=cache['groups'].astype(str)
split_cache={}
for name,df in data.items():
 for p in (out/name/'splits').glob('*.csv.gz'):
  if p.name.startswith('.'):continue
  m=pd.read_csv(p);meta=json.loads(p.with_name(p.name.replace('.csv.gz','.json')).read_text())
  assert m.connectivity_key.tolist()==df.connectivity_key.tolist()
  assert np.array_equal(m.y.to_numpy(),df.y.to_numpy())
  assert m.group.tolist()==group_arrays[name].tolist()
  assert m.groupby('group').partition.nunique().max()==1
  assert meta['manifest_sha256']==sha(p)
  assert meta['script_sha256']==source_code_sha
  if meta['mode']=='exact_random':
   assert (m.partition=='test').sum()==meta['reference_test_count']
   assert (m.partition=='val').sum()==meta['reference_val_count']
   assert meta['label_blind'] and meta['no_label_stratification'] and not meta['uniform_over_feasible_partitions']
  split_cache[str(p)]={k:np.flatnonzero(m.partition.to_numpy()==k) for k in ['train','val','test']}
def validate(path):
 folder=path.parent;s=json.loads(path.read_text());c=json.loads((folder/'config.json').read_text())
 assert s['status']=='complete'
 assert s['script_sha256']==c['script_sha256']==source_code_sha
 assert c['dependency_hashes']['codes/review_2026_10_14/baselines.py']==helper_sha
 assert s['config_sha256']==sha(folder/'config.json')
 assert c['split_manifest_sha256']==sha(root/c['split_manifest'])
 assert c['dataset_sha256']==input_hashes[s['endpoint']]
 assert c['dependency_hashes']['codes/scaffold/src/metrics.py']==sha(root/'codes/scaffold/src/metrics.py')
 pred=pd.read_csv(folder/'predictions.csv.gz');idx=split_cache[str(root/c['split_manifest'])]['test'];df=data[s['endpoint']]
 assert pred.connectivity_key.tolist()==df.iloc[idx].connectivity_key.tolist()
 assert np.array_equal(pred.y_true.to_numpy(),df.iloc[idx].y.to_numpy())
 assert np.isfinite(pred.prediction).all()
 if not np.isfinite(s[s['primary_metric']]):
  assert c['primary_metric']=='auroc' and len(np.unique(pred.y_true))<2
 metrics=evaluate(c['task_kind'],pred.y_true.to_numpy(),pred.prediction.to_numpy())
 max_diff=max([abs(metrics[k]-s[k]) for k in metrics if np.isfinite(metrics[k])]+[0])
 assert max_diff<1e-6
 return {'run_id':s['run_id'],'status':'passed','max_metric_difference':max_diff,'reused_deterministic_predictions':not s['fitted_again']}
paths=sorted(out.glob('*/runs/*/summary.json'))
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(validate,paths))
pd.DataFrame(rows).to_csv(out/'validation_runs.csv',index=False)
runs=pd.read_csv(out/'runs.csv');summaries=[]
safe_mean=lambda a:float(a.mean()) if len(a) else float('nan')
safe_min=lambda a:float(a.min()) if len(a) else float('nan')
safe_max=lambda a:float(a.max()) if len(a) else float('nan')
for (endpoint,geometry,rep),g in runs.groupby(['endpoint','geometry','representation']):
 metric=g.primary_metric.iloc[0];null=g[g['mode']=='exact_random'];high=g[g['mode']=='frontier_reference'];rev=g[g['mode']=='reverse_frontier'];ref=float(high[metric].iloc[0]);v=null[metric].dropna().to_numpy();rv=rev[metric].dropna().to_numpy()
 row={'endpoint':endpoint,'geometry':geometry,'representation':rep,'primary_metric':metric,'random_n':len(v),'frontier_reference':ref,'random_mean':float(v.mean()),'random_sd':float(v.std(ddof=1)),'random_median':float(np.median(v)),'random_q025':float(np.quantile(v,.025)),'random_q975':float(np.quantile(v,.975)),'frontier_minus_random_mean':ref-float(v.mean()),'empirical_difficulty_fraction':float(np.mean(v>=ref)) if metric!='mae' else float(np.mean(v<=ref)),'reverse_allocation_records':len(rev),'reverse_valid_primary_records':len(rv),'reverse_unique_partitions':int(pd.read_csv(out/endpoint/'allocation_manifest.csv').query('mode=="reverse_frontier" and geometry==@geometry').assignment_sha256.nunique()),'reverse_mean_over_allocation_records':safe_mean(rv),'reverse_min':safe_min(rv),'reverse_max':safe_max(rv),'reference_test_n':int(high.test_n.iloc[0]),'reference_val_n':int(high.val_n.iloc[0]),'reverse_test_n_min':int(rev.test_n.min()),'reverse_test_n_max':int(rev.test_n.max())}
 if null.test_positive_fraction.notna().any():row.update(random_prevalence_q025=float(null.test_positive_fraction.quantile(.025)),random_prevalence_q975=float(null.test_positive_fraction.quantile(.975)),frontier_prevalence=float(high.test_positive_fraction.iloc[0]))
 summaries.append(row)
pd.DataFrame(summaries).to_csv(out/'allocation_summary.csv',index=False)
meta_files=[json.loads(p.read_text()) for p in out.glob('*/splits/*exact_random*.json') if not p.name.startswith('.')]
report={'status':'all_passed','split_manifests':len(split_cache),'run_records':len(rows),'distinct_fits':int(runs.fitted_again.sum()),'reused_deterministic_run_records':int((~runs.fitted_again).sum()),'random_allocations':len(meta_files),'random_per_endpoint_geometry':50,'random_joint_size_matches_verified':True,'max_sequential_attempts':max(m['attempts'] for m in meta_files),'allocation_failures':len(list(out.glob('*/allocation_failures/*.json'))),'max_metric_recomputation_difference':max(r['max_metric_difference'] for r in rows),'source_code_sha256':source_code_sha,'executed_common_helper_sha256':helper_sha,'convergence_warning_runs':int(runs.warnings.fillna('').str.contains('failed to converge|TOTAL NO. OF ITERATIONS|STOP: TOTAL NO.',case=False,regex=True).sum()),'undefined_primary_metric_run_records':int(sum(pd.isna(row[row['primary_metric']]) for _,row in runs.iterrows())),'validation_script_sha256':sha(__file__),'undefined_primary_cases':runs[[not np.isfinite(row[row['primary_metric']]) for _,row in runs.iterrows()]][['endpoint','geometry','mode','representation']].value_counts().reset_index(name='records').to_dict('records'),'distribution_note':'random quantiles describe nonuniform label-blind constrained allocation sensitivity, not bootstrap confidence intervals or independent deployment-domain inference; reverse fields describe nominal whole-group allocations.'}
(out/'validation_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
print(pd.DataFrame(summaries)[['endpoint','geometry','representation','frontier_reference','random_mean','empirical_difficulty_fraction','reverse_mean_over_allocation_records']].round(4).to_string(index=False))
