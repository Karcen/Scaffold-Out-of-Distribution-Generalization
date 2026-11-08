"""Frozen post-processing for reviewer composition/overlap/support requests."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,average_precision_score,mean_absolute_error
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'codes/scaffold'))
from src.chem import audit_molecule
EXP=ROOT/'results/revision/experiments';OUT=ROOT/'results/review_2026_10_14/composition';OUT.mkdir(parents=True,exist_ok=True)
ENDPOINTS=['BBB','HIA','CYP2D6','DILI','PPBR','Solubility'];SEEDS=range(5);SOURCES=set()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):SOURCES.add(Path(p));return pd.read_csv(p)
def writejson(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
 writejson(OUT/'protocol.json',{'date':'2026-10-14','seeds':list(SEEDS),'test_target':.2,'models':'completed common Morgan MLP; no refitting','regression_baseline':'training-set median, mean absolute test error','regression_skill':'1 - model_MAE / training_median_MAE','similarity':'maximum training Tanimoto; ECFP4 radius2 2048bit and atom-pair 2048bit; atom-pair does not enter either allocation geometry','chemistry_flags':'RDKit fragment count>1; total nonzero formal charge; declared original METALS set; no Murcko ring scaffold','script_sha256':sha(__file__)})
 composition=[];overlap=[];support=[];properties=[]
 for ep in ENDPOINTS:
  data=read(EXP/ep/'dataset.csv.gz');attrs=[]
  for s in data.smiles:
   a=audit_molecule(s);assert a.valid;attrs.append({'fragments':a.fragments,'multifragment':a.fragments>1,'charged':a.formal_charge!=0,'metal':a.contains_metal,'acyclic':a.scaffold=='__ACYCLIC__'})
  attrs=pd.DataFrame(attrs);props=pd.concat([data[['row_index','connectivity_key','smiles','y']],attrs],axis=1);props['endpoint']=ep;properties.append(props)
  desc=read(EXP/ep/'descriptor_group_scores.csv').set_index('group');morg=read(EXP/ep/'morgan_group_scores.csv').set_index('group');assert set(desc.index)==set(morg.index)
  corr=spearmanr(desc.score,morg.loc[desc.index,'score']).statistic
  mols=[Chem.MolFromSmiles(s) for s in data.smiles]
  generators={'ecfp4':rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048),'atom_pair':rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)}
  fps={key:[g.GetFingerprint(m) for m in mols] for key,g in generators.items()}
  cached={}
  for seed in SEEDS:
   manifests={regime:read(EXP/ep/'splits'/f'{regime}_test20_seed{seed}.csv.gz') for regime in ['scaffold','descriptor','morgan']}
   tests={k:set(m.loc[m.partition.eq('test'),'row_index'].astype(int)) for k,m in manifests.items()}
   intersection=len(tests['descriptor']&tests['morgan']);union=len(tests['descriptor']|tests['morgan'])
   overlap.append({'endpoint':ep,'seed':seed,'descriptor_test_n':len(tests['descriptor']),'morgan_test_n':len(tests['morgan']),'intersection_n':intersection,'union_n':union,'jaccard':intersection/union,'group_score_spearman':corr,'groups':len(desc)})
   for regime,m in manifests.items():
    assert m.row_index.tolist()==data.row_index.tolist();tr=np.flatnonzero(m.partition.eq('train'));te=np.flatnonzero(m.partition.eq('test'));va=np.flatnonzero(m.partition.eq('val'))
    p=read(EXP/ep/'runs'/f'{ep}__{regime}__test20__morgan__mlp__seed{seed}__common'/'predictions.csv.gz').set_index('row_index').loc[te];assert np.allclose(p.y_true,data.y.iloc[te])
    rec={'endpoint':ep,'regime':regime,'seed':seed,'target_test_fraction':.2,'train_n':len(tr),'val_n':len(va),'test_n':len(te),'realised_test_fraction':len(te)/len(data),'test_groups':m.group.iloc[te].nunique(),'largest_test_group_n':int(m.group.iloc[te].value_counts().max())}
    y=data.y.iloc[te].to_numpy(float);ytr=data.y.iloc[tr].to_numpy(float)
    rec.update({'target_mean':float(y.mean()),'target_sd':float(y.std(ddof=1)),'target_min':float(y.min()),'target_max':float(y.max()),'target_q25':float(np.quantile(y,.25)),'target_q75':float(np.quantile(y,.75))})
    if ep in ['PPBR','Solubility']:
     med=float(np.median(ytr));base=float(np.abs(y-med).mean());mae=mean_absolute_error(y,p.prediction)
     rec.update({'positive_fraction':None,'training_median':med,'median_baseline_mae':base,'model_mae':mae,'mae_skill':1-mae/base if base>0 else None,'auroc':None,'auprc':None})
    else:
     rec.update({'positive_fraction':float(y.mean()),'training_median':None,'median_baseline_mae':None,'model_mae':None,'mae_skill':None,'auroc':roc_auc_score(y,p.prediction),'auprc':average_precision_score(y,p.prediction)})
    for flag in ['multifragment','charged','metal','acyclic']:
     rec[flag+'_n']=int(attrs[flag].iloc[te].sum());rec[flag+'_fraction']=float(attrs[flag].iloc[te].mean())
    composition.append(rec)
    # Cache exact membership. Geometry coordinates are never used for similarities.
    cachekey=(tuple(tr),tuple(te));simdata=cached.get(cachekey)
    if simdata is None:
     simdata={name:np.array([max(DataStructs.BulkTanimotoSimilarity(fp[i],[fp[j] for j in tr])) for i in te]) for name,fp in fps.items()};cached[cachekey]=simdata
    for i,row in enumerate(te):
     support.append({'endpoint':ep,'regime':regime,'seed':seed,'row_index':int(row),'connectivity_key':data.connectivity_key.iloc[row],'y_true':float(data.y.iloc[row]),'prediction':float(p.prediction.iloc[i]),**{name+'_max_training_tanimoto':float(values[i]) for name,values in simdata.items()}})
  print(ep,'composition/support complete',flush=True)
 pd.DataFrame(composition).to_csv(OUT/'split_composition.csv',index=False);pd.DataFrame(overlap).to_csv(OUT/'frontier_overlap.csv',index=False)
 pd.concat(properties).to_csv(OUT/'molecule_properties.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 s=pd.DataFrame(support);s.to_csv(OUT/'independent_support_molecules.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 groupcols=['endpoint','regime'];c=pd.DataFrame(composition);cols=[x for x in c if x not in groupcols+['seed']];summary=c.groupby(groupcols)[cols].agg(['mean','std','min','max']);summary.columns=['_'.join(x) for x in summary.columns];summary.reset_index().to_csv(OUT/'split_composition_summary.csv',index=False)
 ss=s.groupby(['endpoint','regime','seed'])[['ecfp4_max_training_tanimoto','atom_pair_max_training_tanimoto']].agg(['mean','median']);ss.columns=['_'.join(x) for x in ss.columns];ss.reset_index().to_csv(OUT/'support_summary.csv',index=False)
 writejson(OUT/'input_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(SOURCES)})
 writejson(OUT/'sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='sha256.json' and not p.name.startswith('._')})
if __name__=='__main__':main()
