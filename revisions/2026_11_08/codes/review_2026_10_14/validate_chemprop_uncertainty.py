"""Read-only statistical checks of Chemprop draw-derived interval outputs."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/review_2026_10_14/chemprop_uncertainty'
BENCH=ROOT/'results/review_2026_10_14/benchmark_uncertainty'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return pd.read_csv(p,float_precision='round_trip')

def main():
 summary=read(OUT/'model_intervals.csv');front=read(OUT/'frontier_minus_scaffold.csv')
 pairs=read(OUT/'chemprop_minus_existing_models.csv');invalid=read(OUT/'invalid_draws_by_fit.csv');validation=read(OUT/'point_and_identity_validation.csv')
 assert len(summary)==108 and len(front)==72 and len(pairs)==568 and len(validation)==90
 cache={}
 def draws(endpoint,subset,scheme,old=False):
  key=(endpoint,subset,scheme,old)
  if key not in cache:cache[key]=read((BENCH if old else OUT)/f'{endpoint}_{subset}_{scheme}_draws.csv.gz')
  assert np.array_equal(cache[key].replicate,np.arange(2000))
  return cache[key]
 def check(values,lower,upper,valid):
  finite=np.isfinite(values);assert finite.sum()==valid
  ci=np.quantile(values[finite],[.025,.975]);assert np.max(np.abs(ci-[lower,upper]))<1e-12
 for row in summary.itertuples():
  values=draws(row.endpoint,row.subset,row.scheme)[f'{row.regime}:{row.model}:{row.metric}'].to_numpy(float)
  check(values,row.lower,row.upper,row.valid_draws)
 for row in front.itertuples():
  data=draws(row.endpoint,row.subset,row.scheme)
  values=data[f'{row.frontier}:{row.model}:{row.metric}'].to_numpy(float)-data[f'scaffold:{row.model}:{row.metric}'].to_numpy(float)
  check(values,row.lower,row.upper,row.valid_draws)
 for row in pairs.itertuples():
  first=draws(row.endpoint,row.subset,row.scheme)[f'{row.regime}:{row.first_model}:{row.metric}'].to_numpy(float)
  second=draws(row.endpoint,row.subset,row.scheme,old=True)[f'{row.regime}:{row.second_model}:{row.metric}'].to_numpy(float)
  check(first-second,row.lower,row.upper,row.valid_draws)
 manifest=json.loads((OUT/'sha256.json').read_text())
 assert all(sha(ROOT/path)==value for path,value in manifest.items())
 config=json.loads((OUT/'config.json').read_text())
 assert all(sha(ROOT/path)==value for path,value in config['source_sha256'].items())
 assert sha(ROOT/'codes/review_2026_10_14/chemprop_uncertainty.py')==config['script_sha256']
 report={'source_fits':len(validation),'model_interval_rows':len(summary),'frontier_contrast_rows':len(front),'complete_new_exploratory_pair_rows':len(pairs),'per_fit_metric_scheme_invalid_records':len(invalid),'summary_intervals_recomputed_from_exact_draws':len(summary),'frontier_intervals_recomputed_from_exact_draw_differences':len(front),'new_pair_intervals_recomputed_from_exact_shared_draws':len(pairs),'all_draw_interval_checks_passed':True,'invalid_metric_draws':int(invalid.invalid_draws.sum()),'empty_draws':int(invalid.empty_draws.sum()),'missing_positive_draws':int(invalid.missing_positive_draws.sum()),'missing_negative_draws':int(invalid.missing_negative_draws.sum()),'max_metric_difference':float(validation.max_metric_difference.max()),'max_saved_target_source_difference':float(validation.max_saved_target_source_difference.max()),'output_hashes_verified':len(manifest),'input_hashes_verified':len(config['source_sha256']),'validator_sha256':sha(__file__)}
 (OUT/'output_validation.json').write_text(json.dumps(report,indent=2)+'\n')
 (OUT/'sha256.json').write_text(json.dumps({str(path.relative_to(ROOT)):sha(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name!='sha256.json' and not path.name.startswith('._')},indent=2)+'\n')
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
