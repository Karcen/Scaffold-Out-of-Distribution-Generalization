"""Prespecified geometry ablations and raw-input chemical-curation sensitivity."""
from __future__ import annotations
import os
for v in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[v]='2'
from pathlib import Path
import argparse,dataclasses,hashlib,importlib.util,json,shutil,sys,time,warnings
import numpy as np,pandas as pd,torch
from rdkit import Chem,rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'results/revision/experiments';OUT=ROOT/'results/review_2026_10_14/sensitivities';PROTO=ROOT/'docs/review_2026_10_14/SENSITIVITY_PROTOCOL.md'
sys.path.insert(0,str(ROOT/'codes/scaffold'));sys.path.insert(0,str(ROOT/'codes/repro'))
from src.chem import descriptor_vector,murcko_scaffold,connectivity_key,atom_pair_fingerprint,audit_molecule
from src.data import TASKS
from src.splits import _group_indices,_cluster_acyclic_scaffolds
from src.models import TrainConfig,train_model
from src.objectives import ObjectiveConfig
from src.metrics import evaluate
from grid_common import build_features
from run_submission_revision import norm,allocate
SEEDS=range(5);GEOMETRY_SEED=1729
VARIANTS=[('morgan_svd16','morgan',16,10,.5,None),('morgan_svd64','morgan',64,10,.5,None),('descriptor_k5','descriptor',32,5,.5,None),('descriptor_k20','descriptor',32,20,.5,None),('morgan_k5','morgan',32,5,.5,None),('morgan_k20','morgan',32,20,.5,None),('descriptor_radius25','descriptor',32,10,.25,None),('descriptor_radius75','descriptor',32,10,.75,None),('morgan_radius25','morgan',32,10,.25,None),('morgan_radius75','morgan',32,10,.75,None),('descriptor_no_size','descriptor',32,10,.5,[0,9]),('atom_pair_svd32','atom_pair',32,10,.5,None)]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rel(p):return str(Path(p).relative_to(ROOT))
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
 def_unused=None

def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(clean(v),indent=2,allow_nan=False));tmp.replace(p)
def frozen_helper():
 path=OUT/'code_snapshots/baselines.py';path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():shutil.copyfile(ROOT/'codes/review_2026_10_14/baselines.py',path)
 spec=importlib.util.spec_from_file_location('frozen_review_baselines',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod,path

def score(z,groups,k,weight):
 gd=_group_indices(groups);keys=list(gd);centers=np.stack([np.median(z[gd[g]],axis=0) for g in keys]);r=np.linalg.norm(centers-np.median(centers,axis=0),axis=1)
 num=min(k+1,len(keys));di=NearestNeighbors(n_neighbors=num,algorithm='ball_tree',n_jobs=1).fit(centers).kneighbors(centers)[0];local=di[:,1:].mean(axis=1)
 def rank(x):
  order=np.argsort(x,kind='mergesort');out=np.empty(len(x));out[order]=np.arange(len(x));return out/max(len(x)-1,1)
 return keys,gd,weight*rank(r)+(1-weight)*rank(local),r,local

def manifest(path,df,groups,assignment,keys,q):
 assert len(assignment)==len(df);m=df[['row_index','connectivity_key','y']].copy();m['group']=groups;m['partition']=assignment;m['frontier_score']=[dict(zip(keys,q))[g] for g in groups]
 assert m.groupby('group').partition.nunique().max()==1
 assert set(m.partition)=={'train','val','test'}
 if path.exists():assert pd.read_csv(path).partition.tolist()==m.partition.tolist()
 else:m.to_csv(path,index=False,compression={'method':'gzip','mtime':0})
 return m

def fit(spec,df,a,path,variant,regime,seed,representation,family,base,helperpath):
 model='mlp' if family=='mlp' else 'rf' if family=='rf' else 'logistic' if spec.kind=='binary' else 'ridge';rid=f'{spec.name}__{variant}__{regime}__{representation}__{model}__seed{seed}'
 folder=path.parent.parent/'runs'/rid;done=folder/'summary.json'
 if done.exists():
  assert json.loads(done.read_text())['status']=='complete'
  stored=json.loads((folder/'config.json').read_text())
  assert stored['split_sha256']==sha(path) and stored['dataset_sha256']==sha(path.parent.parent/'dataset.csv.gz')
  assert stored['features_sha256']==sha(path.parent.parent/'features.npz') and stored['helper_sha256']==sha(helperpath)
  return
 folder.mkdir(parents=True,exist_ok=True);m=pd.read_csv(path);parts={k:np.flatnonzero(m.partition.eq(k)) for k in ['train','val','test']};tr,va,te=[parts[k] for k in ['train','val','test']];y=df.y.to_numpy();x,scaling=base.transform(a,representation,tr)
 config={'run_id':rid,'endpoint':spec.name,'variant':variant,'regime':regime,'seed':seed,'allocation_seed':seed,'fitting_seed':seed,'test_target':.2,'representation':representation,'model':model,'task_kind':spec.kind,'primary_metric':spec.primary_metric,'dataset':rel(path.parent.parent/'dataset.csv.gz'),'dataset_sha256':sha(path.parent.parent/'dataset.csv.gz'),'features_sha256':sha(path.parent.parent/'features.npz'),'split_manifest':rel(path),'split_sha256':sha(path),'script_sha256':sha(__file__),'protocol_sha256':sha(PROTO),'helper_snapshot':rel(helperpath),'helper_sha256':sha(helperpath),'input_transformation':scaling,'recipe':'benchmark_mlp' if family=='mlp' else 'common_linear' if family=='linear' else 'fixed_random_forest','validation_used':family=='mlp'}
 if family=='mlp':config['training']=dataclasses.asdict(TrainConfig(seed=seed));config['objective']=dataclasses.asdict(ObjectiveConfig(method='ERM'))
 else:config['training']=base.estimator(spec.kind,family,seed).get_params()
 dump(folder/'config.json',config);t=time.monotonic()
 with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=2):
  warnings.simplefilter('always')
  if family=='mlp':
   result=train_model(x,y,tr,va,te,spec.kind,spec.primary_metric,np.zeros((1,len(y)),int),np.zeros((1,1)),np.ones((1,1))*len(tr),ObjectiveConfig(method='ERM'),TrainConfig(seed=seed));pred=result.test_prediction;pd.DataFrame(result.history).to_csv(folder/'history.csv',index=False);best=result.best_epoch
  else:
   training_target,scale,target_rule=base.target_transform(spec.kind,family,y,tr);config['target_transformation']=target_rule;dump(folder/'config.json',config);pred=base.fit_prediction(spec.kind,base.estimator(spec.kind,family,seed),x,training_target,tr,te,scale);best=None
  metrics=evaluate(spec.kind,y[te],pred)
 assert np.isfinite(pred).all();p=df.iloc[te][['row_index','connectivity_key','smiles']].copy();p['group']=m.group.iloc[te].to_numpy();p['y_true']=y[te];p['prediction']=pred;p.to_csv(folder/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 baseline=float(np.abs(y[te]-np.median(y[tr])).mean()) if spec.kind=='regression' else None
 dump(done,{k:config[k] for k in ['run_id','endpoint','variant','regime','seed','representation','model','task_kind','primary_metric','recipe']}|{'status':'complete',**metrics,'train_n':len(tr),'val_n':len(va),'test_n':len(te),'test_positive_fraction':float(y[te].mean()) if spec.kind=='binary' else None,'training_median_mae':baseline,'mae_skill':1-metrics['mae']/baseline if baseline and spec.kind=='regression' else None,'best_epoch':best,'warnings':[str(w.message) for w in caught],'seconds':time.monotonic()-t})
 print('COMPLETE',rid,metrics[spec.primary_metric],flush=True)

def curation_data(spec):
 folder=OUT/'curation'/spec.name;folder.mkdir(parents=True,exist_ok=True);p=folder/'dataset.csv.gz';features=folder/'features.npz'
 if p.exists() and features.exists():return pd.read_csv(p),{k:v for k,v in np.load(features).items()}
 source=ROOT/'data/raw/scaffold/data/external'/spec.path;raw=pd.read_csv(source,sep='\t')
 if spec.name=='PPBR':raw=raw[raw.Species.eq('Homo sapiens')].copy()
 raw=raw.rename(columns={'Drug':'smiles','Y':'y','Drug_ID':'source_id'});raw['y']=pd.to_numeric(raw.y,errors='coerce');raw=raw[raw.smiles.notna()&raw.y.notna()].copy()
 chooser=rdMolStandardize.LargestFragmentChooser(True);uncharger=rdMolStandardize.Uncharger(True,False,False);taut=rdMolStandardize.TautomerEnumerator();rows=[];mapping=[]
 for i,row in raw.iterrows():
  mol=Chem.MolFromSmiles(row.smiles);reason=None
  if mol is None:reason='invalid_input'
  try:
   if mol is not None:
    std=rdMolStandardize.Cleanup(mol);std=chooser.choose(std);std=uncharger.uncharge(std);std=taut.Canonicalize(std);Chem.SanitizeMol(std);s=Chem.MolToSmiles(std,canonical=True,isomericSmiles=True);key=connectivity_key(s);assert key!='__INVALID__'
    rows.append({'source_id':str(row.source_id),'smiles':s,'connectivity_key':key,'y':row.y});mapping.append({'raw_row':int(i),'source_id':str(row.source_id),'original_smiles':row.smiles,'standardised_smiles':s,'standardised_key':key,'status':'retained'})
   else:mapping.append({'raw_row':int(i),'source_id':str(row.source_id),'status':reason})
  except Exception as e:mapping.append({'raw_row':int(i),'source_id':str(row.source_id),'status':'standardisation_failed','error':repr(e)})
 frame=pd.DataFrame(rows);final=[];ties=0
 for key,g in frame.groupby('connectivity_key',sort=False):
  if spec.kind=='binary' and g.y.mean()==.5:ties+=1;continue
  y=int(g.y.mean()>.5) if spec.kind=='binary' else float(g.y.median());final.append({'source_id':'|'.join(g.source_id),'smiles':g.smiles.iloc[0],'connectivity_key':key,'y':y,'replicate_n':len(g)})
 df=pd.DataFrame(final);df.insert(0,'row_index',np.arange(len(df)));df.to_csv(p,index=False,compression={'method':'gzip','mtime':0});pd.DataFrame(mapping).to_csv(folder/'raw_standardisation_mapping.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 feats=build_features(df.smiles.tolist());rawdesc=np.stack([descriptor_vector(s) for s in df.smiles]);z,_=norm(rawdesc);groups,acyc=_cluster_acyclic_scaffolds(np.array([murcko_scaffold(s) for s in df.smiles],object),z)
 svd=TruncatedSVD(n_components=min(32,len(df)-1),random_state=GEOMETRY_SEED);mz,_=norm(svd.fit_transform(feats['morgan'].astype(float)));a={**feats,'raw_descriptors':rawdesc,'desc_geometry':z,'morgan_geometry':mz,'groups':groups.astype(str)};np.savez_compressed(features,**a)
 dump(folder/'dataset_config.json',{'source':rel(source),'source_sha256':sha(source),'raw_rows_after_endpoint_filter':len(raw),'successful_standardisations':len(rows),'failed_or_invalid':len(raw)-len(rows),'standardised_connectivity_groups':len(frame.groupby('connectivity_key')),'binary_tie_groups_removed':ties,'final_n':len(df),'dataset_sha256':sha(p),'standardisation':['RDKit Cleanup','largest fragment preferOrganic=True','Uncharger canonicalOrder=True force=False protonationOnly=False','canonical tautomer default TautomerEnumerator','stereochemistry-free connectivity deduplication'],'neutralisation':'attempt; residual formal charges retained if Uncharger cannot remove them','labels':'raw records majority with binary ties removed; regression median after standardisation','geometry_seed':GEOMETRY_SEED,'acyclic_group_count':acyc,'protocol_sha256':sha(PROTO),'script_sha256':sha(__file__)})
 return df,a

def curation(spec,base,helper):
 df,a=curation_data(spec);folder=OUT/'curation'/spec.name;(folder/'splits').mkdir(exist_ok=True)
 for regime in ['scaffold','descriptor','morgan']:
  z=a['desc_geometry'] if regime in ['scaffold','descriptor'] else a['morgan_geometry'];keys,gd,q,r,l=score(z,a['groups'],10,.5)
  for seed in SEEDS:
   assign=allocate(gd,keys,q,seed,.2,regime=='scaffold');path=folder/'splits'/f'{regime}_test20_seed{seed}.csv.gz';manifest(path,df,a['groups'],assign,keys,q)
   for rep,family in [('physchem','linear'),('maccs','linear'),('morgan','linear'),('morgan','rf'),('morgan','mlp')]:fit(spec,df,a,path,'standardised_input',regime,seed,rep,family,base,helper)


def geometry(spec,base,helper):
 df=pd.read_csv(OLD/spec.name/'dataset.csv.gz');old={k:v for k,v in np.load(OLD/spec.name/'features.npz').items()};folder=OUT/'geometry'/spec.name;folder.mkdir(parents=True,exist_ok=True);(folder/'splits').mkdir(exist_ok=True)
 if not (folder/'dataset.csv.gz').exists():shutil.copyfile(OLD/spec.name/'dataset.csv.gz',folder/'dataset.csv.gz')
 a={**old};
 if 'physchem' not in a:a.update(build_features(df.smiles.tolist()))
 if not (folder/'features.npz').exists():np.savez_compressed(folder/'features.npz',**a)
 stats=[]
 for name,geo,dim,k,w,drop in VARIANTS:
  if geo=='descriptor':
   z=old['desc_geometry'] if drop is None else norm(np.delete(old['raw_descriptors'],drop,axis=1))[0]
  elif geo=='morgan' and dim==32:z=old['morgan_geometry']
  else:
   bits=old['morgan'] if geo=='morgan' else np.stack([atom_pair_fingerprint(s,n_bits=2048) for s in df.smiles]);svd=TruncatedSVD(n_components=min(dim,len(df)-1),random_state=GEOMETRY_SEED);z,_=norm(svd.fit_transform(bits.astype(float)))
  keys,gd,q,r,l=score(z,a['groups'],k,w);pd.DataFrame({'group':keys,'n':[len(gd[x]) for x in keys],'radius':r,'local_distance':l,'score':q}).to_csv(folder/(name+'_scores.csv'),index=False)
  for seed in SEEDS:
   assignment=allocate(gd,keys,q,seed,.2,False);path=folder/'splits'/f'{name}_seed{seed}.csv.gz';m=manifest(path,df,a['groups'],assignment,keys,q);reference=pd.read_csv(OLD/spec.name/'splits'/f'{"descriptor" if geo=="descriptor" else "morgan"}_test20_seed{seed}.csv.gz');test=set(m.row_index[m.partition.eq('test')]);rt=set(reference.row_index[reference.partition.eq('test')]);stats.append({'endpoint':spec.name,'variant':name,'seed':seed,'geometry':geo,'dimensions':z.shape[1],'k':k,'radius_weight':w,'test_n':len(test),'test_jaccard_with_base':len(test&rt)/len(test|rt)})
   for rep in ['physchem','morgan']:fit(spec,df,a,path,name,geo,seed,rep,'linear',base,helper)
 pd.DataFrame(stats).to_csv(folder/'variant_metadata.csv',index=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['geometry','curation','all'],default='all');p.add_argument('--tasks',nargs='+',default=[s.name for s in TASKS]);args=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(1);base,helper=frozen_helper();shutil.copyfile(__file__,OUT/'code_snapshots'/('sensitivity_'+sha(__file__)+'.py'));shutil.copyfile(PROTO,OUT/'code_snapshots'/('protocol_'+sha(PROTO)+'.md'))
 dump(OUT/'environment.json',{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'rdkit':rdBase.rdkitVersion,'torch':torch.__version__,'cpu_threads':2,'protocol_sha256':sha(PROTO),'script_sha256':sha(__file__),'helper_sha256':sha(helper)})
 for name in args.tasks:
  spec=next(s for s in TASKS if s.name==name)
  if args.mode in ['geometry','all']:geometry(spec,base,helper)
  if args.mode in ['curation','all']:curation(spec,base,helper)
  summaries=[json.loads(p.read_text()) for p in OUT.glob('*/*/runs/*/summary.json')];pd.DataFrame(summaries).to_csv(OUT/'runs.csv',index=False)
 print('SENSITIVITIES COMPLETE',flush=True)
if __name__=='__main__':main()
