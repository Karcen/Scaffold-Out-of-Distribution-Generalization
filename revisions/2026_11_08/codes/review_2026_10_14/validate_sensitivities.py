"""Independent file, grouping and metric validation of completed sensitivities."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/'results/review_2026_10_14/sensitivities'
OUT=ROOT/'results/review_2026_10_14/audit';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'codes/scaffold'))
from src.metrics import evaluate
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 rows=[];dataset_cache={};split_cache={};snapshots={};max_metric_difference=0.
 for summary_path in sorted(SOURCE.glob('*/*/runs/*/summary.json')):
  summary=json.loads(summary_path.read_text());assert summary['status']=='complete'
  folder=summary_path.parent;config=json.loads((folder/'config.json').read_text())
  assert summary['run_id']==config['run_id']==folder.name
  dataset_path=ROOT/config['dataset'];feature_path=dataset_path.with_name('features.npz')
  if str(dataset_path) not in dataset_cache:
   data=pd.read_csv(dataset_path,float_precision='round_trip');features=np.load(feature_path)
   assert np.array_equal(data.row_index,np.arange(len(data))) and data.connectivity_key.is_unique
   assert all(len(features[key])==len(data) for key in features.files)
   assert features['morgan'].shape==(len(data),1024) and features['physchem'].shape==(len(data),9) and features['maccs'].shape==(len(data),167)
   dataset_cache[str(dataset_path)]=(data,features,sha(dataset_path),sha(feature_path))
  data,features,dh,fh=dataset_cache[str(dataset_path)]
  assert dh==config['dataset_sha256'] and fh==config['features_sha256']
  split_path=ROOT/config['split_manifest']
  if str(split_path) not in split_cache:
   manifest=pd.read_csv(split_path,float_precision='round_trip');assert np.array_equal(manifest.row_index,data.row_index)
   assert np.array_equal(manifest.connectivity_key.to_numpy(str),data.connectivity_key.to_numpy(str))
   assert np.allclose(manifest.y,data.y,atol=1e-10,rtol=0)
   assert np.array_equal(manifest.group.to_numpy(str),features['groups'].astype(str))
   assert set(manifest.partition)=={'train','val','test'} and manifest.groupby('group').partition.nunique().max()==1
   split_cache[str(split_path)]=(manifest,sha(split_path))
  manifest,sh=split_cache[str(split_path)];assert sh==config['split_sha256']
  script_path=SOURCE/'code_snapshots'/f"sensitivity_{config['script_sha256']}.py"
  helper_path=ROOT/config['helper_snapshot'];protocol_path=SOURCE/'code_snapshots'/f"protocol_{config['protocol_sha256']}.md"
  for path,expected in [(script_path,config['script_sha256']),(helper_path,config['helper_sha256']),(protocol_path,config['protocol_sha256'])]:
   assert path.is_file() and sha(path)==expected,(path,expected)
   snapshots[str(path.relative_to(ROOT))]=expected
  predictions=pd.read_csv(folder/'predictions.csv.gz',float_precision='round_trip')
  assert not predictions.row_index.duplicated().any()
  ids=predictions.row_index.to_numpy(int);test=np.flatnonzero(manifest.partition=='test')
  assert set(ids)==set(test) and np.allclose(predictions.y_true,data.y.iloc[ids],atol=1e-10,rtol=0)
  for column in ['connectivity_key','smiles']:assert np.array_equal(predictions[column].to_numpy(str),data[column].iloc[ids].to_numpy(str))
  assert np.array_equal(predictions.group.to_numpy(str),manifest.group.iloc[ids].to_numpy(str))
  metrics=evaluate(config['task_kind'],predictions.y_true.to_numpy(float),predictions.prediction.to_numpy(float))
  errors={metric:abs(value-summary[metric]) for metric,value in metrics.items() if np.isfinite(value)}
  assert all(error<1e-6 for error in errors.values()),(folder,errors)
  max_metric_difference=max(max_metric_difference,max(errors.values(),default=0.))
  counts={partition:int((manifest.partition==partition).sum()) for partition in ['train','val','test']}
  assert all(counts[partition]==summary[partition+'_n'] for partition in counts)
  if config['task_kind']=='binary':assert abs(data.y.iloc[test].mean()-summary['test_positive_fraction'])<1e-12
  else:
   train=np.flatnonzero(manifest.partition=='train');baseline=float(np.abs(data.y.iloc[test]-np.median(data.y.iloc[train])).mean())
   assert abs(baseline-summary['training_median_mae'])<1e-10
   assert abs(1-metrics['mae']/baseline-summary['mae_skill'])<1e-6
  if config['model']=='mlp':
   history=pd.read_csv(folder/'history.csv',float_precision='round_trip');assert len(history)>0
   # train_model uses zero-based epoch indices; zero is a valid selection.
   assert isinstance(summary['best_epoch'],int) and summary['best_epoch'] in set(history.epoch)
  rows.append({'run_id':config['run_id'],'endpoint':config['endpoint'],'variant':config['variant'],'regime':config['regime'],'representation':config['representation'],'model':config['model'],'seed':config['seed'],'primary_metric':config['primary_metric'],'saved_primary':summary[config['primary_metric']],'recomputed_primary':metrics[config['primary_metric']],'max_metric_difference':max(errors.values(),default=0.),'dataset':config['dataset'],'split_manifest':config['split_manifest'],'script_snapshot':str(script_path.relative_to(ROOT)),'helper_snapshot':config['helper_snapshot'],'passed':True})
 frame=pd.DataFrame(rows)
 registry=pd.read_csv(SOURCE/'runs.csv',float_precision='round_trip')
 assert registry.status.eq('complete').all() and registry.run_id.is_unique
 assert set(frame.run_id)==set(registry.run_id) and len(frame)==1170
 variants={'descriptor_k5':'descriptor','descriptor_k20':'descriptor','descriptor_radius25':'descriptor','descriptor_radius75':'descriptor','descriptor_no_size':'descriptor','morgan_svd16':'morgan','morgan_svd64':'morgan','morgan_k5':'morgan','morgan_k20':'morgan','morgan_radius25':'morgan','morgan_radius75':'morgan','atom_pair_svd32':'atom_pair'}
 expected=set()
 for endpoint in ['BBB','HIA','CYP2D6','DILI','PPBR','Solubility']:
  linear='ridge' if endpoint in ['PPBR','Solubility'] else 'logistic'
  for seed in range(5):
   for variant,regime in variants.items():
    for representation in ['physchem','morgan']:expected.add((endpoint,variant,regime,representation,linear,seed))
   for regime in ['scaffold','descriptor','morgan']:
    for representation,model in [('physchem',linear),('maccs',linear),('morgan',linear),('morgan','rf'),('morgan','mlp')]:expected.add((endpoint,'standardised_input',regime,representation,model,seed))
 actual=set(frame[['endpoint','variant','regime','representation','model','seed']].itertuples(index=False,name=None))
 assert actual==expected and len(expected)==1170,(actual-expected,expected-actual)
 frame.to_csv(OUT/'sensitivity_fit_validation.csv',index=False)
 report={'completed_fits_validated':len(rows),'expected_fits':len(expected),'geometry_fits':int((frame.variant!='standardised_input').sum()),'curation_fits':int((frame.variant=='standardised_input').sum()),'registry_and_exact_recipe_grid_complete':True,'dataset_files':len(dataset_cache),'split_manifests':len(split_cache),'source_snapshots':snapshots,'max_metric_difference':max_metric_difference,'all_completed_fits_passed':True,'coverage':'All 720 frozen geometry and 450 standardised-input fits; registry IDs and endpoint/variant/regime/representation/model/seed recipe tuples exactly match the full specified grid','csv_numeric_parser':'float_precision=round_trip preserves target ties written by the original run; default parser can perturb Spearman ties without changing substantive target values','validator_sha256':sha(__file__),'registry_sha256':sha(SOURCE/'runs.csv'),'scoring_source_sha256':sha(ROOT/'codes/scaffold/src/metrics.py')}
 (OUT/'sensitivity_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({key:value for key,value in report.items() if key!='source_snapshots'},indent=2))
if __name__=='__main__':main()
