"""Prespecified BBB ECFP4 neighbours and observed AUROC error accounting."""
from pathlib import Path
import hashlib, importlib.util, json
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'results/revision/experiments/BBB'
OUT=ROOT/'results/review_2026_10_14/knn_matched_1024';B=2000;SEED=20261014;SOURCES={}
def source(p):
 p=Path(p);SOURCES[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return pd.read_csv(source(p))
def save(name,rows):
 d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
 d.to_csv(OUT/name,index=False,compression={'method':'gzip','mtime':0} if name.endswith('.gz') else None)

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 d=read(EXP/'dataset.csv.gz');assert np.array_equal(d.row_index,np.arange(len(d)))
 features=np.load(source(EXP/'features.npz'));groups=features['groups'].astype(str)
 generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024,includeChirality=False)
 fps=[]
 for smiles in d.smiles:
  molecule=Chem.MolFromSmiles(smiles);assert molecule is not None;fps.append(generator.GetFingerprint(molecule))
 prediction_rows=[];metric_rows=[];neighbour_rows=[];burden_rows=[];pair_rows=[];descriptor_predictions=[]
 generated=np.stack([np.frombuffer(fp.ToBitString().encode('ascii'),dtype=np.uint8)-ord('0') for fp in fps])
 assert generated.shape==features['morgan'].shape and np.array_equal(generated,features['morgan'])
 matching={'matched_legacy_morgan_bit_matrix':True,'shape':list(generated.shape),'source_feature_dtype':str(features['morgan'].dtype),'generated_in_source_dtype_sha256':hashlib.sha256(generated.astype(features['morgan'].dtype).tobytes()).hexdigest(),'legacy_matrix_sha256':hashlib.sha256(features['morgan'].tobytes()).hexdigest()}
 assert matching['generated_in_source_dtype_sha256']==matching['legacy_matrix_sha256']
 (OUT/'fingerprint_matrix_validation.json').write_text(json.dumps(matching,indent=2)+'\n')
 for regime in ['scaffold','descriptor','morgan']:
  for seed in range(5):
   manifest=read(EXP/'splits'/f'{regime}_test20_seed{seed}.csv.gz')
   assert np.array_equal(manifest.group.to_numpy(str),groups)
   train=np.sort(manifest.loc[manifest.partition=='train','row_index'].to_numpy(int))
   test=manifest.loc[manifest.partition=='test','row_index'].to_numpy(int)
   ytrain=d.y.to_numpy(int)[train];ytest=d.y.to_numpy(int)[test];prevalence=ytrain.mean()
   similarities=np.array([DataStructs.BulkTanimotoSimilarity(fps[i],[fps[j] for j in train]) for i in test])
   order=np.argsort(-similarities,axis=1,kind='stable')[:,:5]
   neighbour_ids=train[order];neighbour_sim=np.take_along_axis(similarities,order,axis=1);neighbour_y=d.y.to_numpy(int)[neighbour_ids]
   predictions={}
   for k in [5,1]:
    sm=neighbour_sim[:,:k];labels=neighbour_y[:,:k];total=sm.sum(axis=1)
    predictions[f'knn{k}_unweighted']=labels.mean(axis=1)
    predictions[f'knn{k}_weighted']=np.divide((sm*labels).sum(axis=1),total,out=np.full(len(test),prevalence),where=total>0)
    for weighted in [False,True]:
     model=f'knn{k}_'+('weighted' if weighted else 'unweighted');p=predictions[model]
     metric_rows.append({'regime':regime,'seed':seed,'model':model,'k':k,'similarity_weighted':weighted,'test_n':len(test),'train_n':len(train),'train_prevalence':prevalence,'test_prevalence':ytest.mean(),'zero_similarity_sum_n':int((total==0).sum()),'auroc':roc_auc_score(ytest,p),'auprc':average_precision_score(ytest,p)})
     prediction_rows.extend({'regime':regime,'seed':seed,'model':model,'row_index':int(i),'smiles':d.smiles.iloc[i],'group_id':groups[i],'y_true':int(y),'prediction':float(v),'train_prevalence':prevalence,'similarity_sum':float(z)} for i,y,v,z in zip(test,ytest,p,total))
   neighbour_rows.extend({'regime':regime,'seed':seed,'test_row_index':int(test[i]),'neighbour_rank':j+1,'train_row_index':int(neighbour_ids[i,j]),'tanimoto':float(neighbour_sim[i,j]),'train_label':int(neighbour_y[i,j]),'train_smiles':d.smiles.iloc[neighbour_ids[i,j]]} for i in range(len(test)) for j in range(5))
   if regime=='descriptor':
    p=read(EXP/'runs'/f'BBB__descriptor__test20__morgan__mlp__seed{seed}__common'/'predictions.csv.gz').set_index('row_index').loc[test]
    assert np.array_equal(ytest,p.y_true.to_numpy(int));score=p.prediction.to_numpy(float)
    predictions['morgan_mlp']=score
    descriptor_predictions.append({'seed':seed,'ids':test,'y':ytest,'predictions':predictions})
    positive=np.flatnonzero(ytest==1);negative=np.flatnonzero(ytest==0)
    difference=score[positive,None]-score[negative][None,:]
    error=(difference<0).astype(float)+.5*(difference==0)
    denominator=len(positive)*len(negative);credit=np.zeros(len(test))
    credit[positive]=.5*error.sum(axis=1);credit[negative]=.5*error.sum(axis=0)
    auroc=roc_auc_score(ytest,score);assert abs(credit.sum()/denominator-(1-auroc))<1e-12
    pair_rows.append({'seed':seed,'positive_n':len(positive),'negative_n':len(negative),'positive_negative_pairs':denominator,'inversions_n':int((difference<0).sum()),'ties_n':int((difference==0).sum()),'error_credit':error.sum(),'auroc':auroc,'normalised_burden_sum':credit.sum()/denominator})
    for i,row_id in enumerate(test):
     burden_rows.append({'seed':seed,'row_index':int(row_id),'smiles':d.smiles.iloc[row_id],'group_id':groups[row_id],'y_true':int(ytest[i]),'mlp_prediction':score[i],'pair_error_burden':credit[i],'normalised_pair_error_burden':credit[i]/denominator,'fraction_total_error':credit[i]/credit.sum(),'nearest_train_row_index':int(neighbour_ids[i,0]),'nearest_tanimoto':neighbour_sim[i,0],'nearest_train_label':int(neighbour_y[i,0]),'knn5_unweighted':predictions['knn5_unweighted'][i],'knn5_weighted':predictions['knn5_weighted'][i]})
   print(regime,seed,flush=True)
 save('per_molecule_predictions.csv.gz',prediction_rows);save('selected_neighbours.csv.gz',neighbour_rows);save('metrics_by_seed.csv',metric_rows)
 metrics=pd.DataFrame(metric_rows);summary=metrics.groupby(['regime','model'])[['auroc','auprc']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];save('metric_summary.csv',summary.reset_index())
 burden=pd.DataFrame(burden_rows);save('mlp_compound_burden_by_seed.csv.gz',burden);save('mlp_pair_accounting.csv',pair_rows)
 compound=burden.groupby(['row_index','smiles','group_id','y_true'])[['mlp_prediction','pair_error_burden','normalised_pair_error_burden','fraction_total_error','nearest_tanimoto','nearest_train_label','knn5_unweighted','knn5_weighted']].mean().reset_index().sort_values('normalised_pair_error_burden',ascending=False)
 save('mlp_compound_burden_mean.csv',compound)
 group=burden.groupby(['seed','group_id']).agg(test_n=('row_index','size'),positive_n=('y_true','sum'),pair_error_burden=('pair_error_burden','sum'),normalised_pair_error_burden=('normalised_pair_error_burden','sum'),fraction_total_error=('fraction_total_error','sum')).reset_index();save('mlp_group_burden_by_seed.csv',group)
 gg=group.groupby('group_id').mean(numeric_only=True).drop(columns='seed').reset_index().sort_values('normalised_pair_error_burden',ascending=False);save('mlp_group_burden_mean.csv',gg)
 # Conditional uncertainty for the shared descriptor-frontier comparison.
 script=source(ROOT/'codes/review_2026_10_14/statistics.py');spec=importlib.util.spec_from_file_location('review_statistic_helpers',script);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 ids=np.unique(np.concatenate([p['ids'] for p in descriptor_predictions]));index={int(v):i for i,v in enumerate(ids)};labels=d.y.to_numpy(int)[ids]
 group_ids,group_index=np.unique(groups[ids],return_inverse=True);rng=np.random.default_rng(SEED)
 compound_weights=np.zeros((B,len(ids)),dtype=np.uint16)
 for label in [0,1]:
  ix=np.flatnonzero(labels==label);compound_weights[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=B)
 cluster_counts=rng.multinomial(len(group_ids),np.full(len(group_ids),1/len(group_ids)),size=B).astype(np.uint16)
 np.savez_compressed(OUT/'descriptor_resampling_weights.npz',compound_ids=ids,compound_weights=compound_weights,group_ids=group_ids,cluster_group_counts=cluster_counts)
 intervals=[];invalid=[];contrasts=[]
 for scheme,weights in [('compound',compound_weights),('cluster',cluster_counts[:,group_index])]:
  draws={}
  for fit in descriptor_predictions:
   ix=np.array([index[int(v)] for v in fit['ids']])
   for model,p in fit['predictions'].items():
    arr,empty,single=helper.weighted_auc(fit['y'],p,weights[:,ix]);draws.setdefault(model,[]).append(arr)
    invalid.append({'scheme':scheme,'seed':fit['seed'],'model':model,'empty_draws':int(empty.sum()),'single_class_draws':int(single.sum())})
  means={model:np.mean(arrays,axis=0) for model,arrays in draws.items()}
  save(f'descriptor_{scheme}_auroc_draws.csv.gz',pd.DataFrame({'replicate':np.arange(B),**means}))
  for model,arr in means.items():
   valid=np.isfinite(arr);lo,hi=np.quantile(arr[valid],[.025,.975]);point=np.mean([roc_auc_score(fit['y'],fit['predictions'][model]) for fit in descriptor_predictions])
   intervals.append({'scheme':scheme,'regime':'descriptor','model':model,'mean_auroc':point,'lower':lo,'upper':hi,'valid_draws':int(valid.sum()),'invalid_draws':int((~valid).sum())})
  models=[model for model in means if model!='morgan_mlp']
  matrix=np.column_stack([means[model]-means['morgan_mlp'] for model in models])
  mlp_point=np.mean([roc_auc_score(fit['y'],fit['predictions']['morgan_mlp']) for fit in descriptor_predictions])
  point_delta=np.array([np.mean([roc_auc_score(fit['y'],fit['predictions'][model]) for fit in descriptor_predictions])-mlp_point for model in models])
  radius,critical,family_n=helper.simultaneous(matrix,point_delta)
  for j,model in enumerate(models):
   valid=np.isfinite(matrix[:,j]);lo,hi=np.quantile(matrix[valid,j],[.025,.975])
   contrasts.append({'scheme':scheme,'regime':'descriptor','first_model':model,'second_model':'morgan_mlp','delta_first_minus_second':point_delta[j],'percentile_lower':lo,'percentile_upper':hi,'valid_draws':int(valid.sum()),'family_simultaneous_lower':point_delta[j]-radius[j],'family_simultaneous_upper':point_delta[j]+radius[j],'exploratory_family_contrasts':len(models),'family_valid_draws':family_n,'critical_max_standardised_deviation':critical})
 save('descriptor_intervals.csv',intervals);save('descriptor_invalid_draws.csv',invalid);save('descriptor_knn_minus_mlp.csv',contrasts)
 config={'matched_legacy_morgan_bit_matrix':matching,'fingerprint':'RDKit Morgan bit vector, radius=2, fpSize=1024, includeChirality=False, default bond/atom invariants, ECFP4','rdkit_version':rdBase.rdkitVersion,'regimes':['scaffold','descriptor','morgan'],'seeds':list(range(5)),'k':[5,1],'selection':'prespecified diagnostic; no validation/test label tuning','tied_similarity':'ascending stable training row ID','weighted_prediction':'sum(similarity*training label)/sum(similarity); zero denominator uses that fit training prevalence','unweighted_prediction':'mean labels of selected neighbours','pair_accounting':'error=1 for inversion and 0.5 for tie; split error equally across positive and negative endpoints; group sum of normalised burdens equals 1-AUROC; descriptive accounting, not causal contribution','bootstrap_replicates':B,'bootstrap_seed':SEED,'interval_estimand':'mean five-seed AUROC conditional on fitted predictors and observed descriptor test compounds/groups; shared weights, no refitting','group_resampling':'whole fixed grouping units; unstratified group multinomial multiplicities','paired_family':'all four neighbour methods minus Morgan MLP, exploratory; 95% maximum standardised centred-deviation simultaneous intervals','source_sha256':SOURCES,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (OUT/'config.json').write_text(json.dumps(config,indent=2)+'\n');(OUT/'sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='sha256.json' and not p.name.startswith('._')},indent=2)+'\n')

if __name__=='__main__':main()
