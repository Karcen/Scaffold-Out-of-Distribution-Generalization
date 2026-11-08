"""Frozen submission protocol: common groups, descriptor/Morgan geometry and fits.
Run with the research Python; every completed run is resumable and independently saved.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
from pathlib import Path
import argparse, sys, json, hashlib, time, warnings, platform, dataclasses, copy
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sklearn, scipy, rdkit
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'codes/scaffold'))
from src.data import TASKS,load_tdc_task
from src.chem import descriptor_vector,morgan_fingerprint,murcko_scaffold
from src.splits import _cluster_acyclic_scaffolds,_group_indices,_allocate_groups,frontier_split,assert_split_integrity,SplitResult
from src.models import train_model,TrainConfig
from src.objectives import ObjectiveConfig
from src.metrics import evaluate
from grid_common import build_features,stratified_random
from arch_models import MolTriNet,SimpleMLPModel,SMILESTokenizer,n_params
OUT=ROOT/'results/revision/experiments'
SEEDS=range(5);FRACTIONS=(.1,.2,.3,.4);GEOMETRY_SEED=1729

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2,allow_nan=True,default=lambda x: x.item() if hasattr(x,'item') else str(x)))
def rpath(p):return str(Path(p).relative_to(ROOT))
def norm(x,train=None):
 x=np.asarray(x,dtype=float);tr=x if train is None else x[train]
 med=np.nanmedian(np.where(np.isfinite(tr),tr,np.nan),axis=0);med=np.nan_to_num(med)
 x=np.where(np.isfinite(x),x,med);scaler=RobustScaler(quantile_range=(10,90)).fit(x if train is None else x[train])
 return np.clip(scaler.transform(x),-20,20).astype(np.float32),{'median_imputation':med.tolist(),'center':scaler.center_.tolist(),'scale':scaler.scale_.tolist()}

def score_geometry(z,groups):
 gd=_group_indices(groups);keys=list(gd);centers=np.stack([np.median(z[gd[g]],axis=0) for g in keys]);radius=np.linalg.norm(centers-np.median(centers,axis=0),axis=1)
 k=min(11,len(keys));nn=NearestNeighbors(n_neighbors=k,algorithm='ball_tree',n_jobs=1).fit(centers);dist=nn.kneighbors(centers)[0];local=dist[:,1:k].mean(axis=1)
 def rank(v):
  order=np.argsort(v,kind='mergesort');r=np.empty(len(v));r[order]=np.arange(len(v));return r/max(len(v)-1,1)
 score=.5*rank(radius)+.5*rank(local)
 return keys,gd,score,radius,local,centers

def allocate(gd,keys,score,seed,fraction,control=False):
 fractions=(.9-fraction,.1,fraction);items=list(gd.items());rng=np.random.default_rng(seed)
 if control:
  rng.shuffle(items);items.sort(key=lambda kv:len(kv[1]),reverse=True);return _allocate_groups(items,sum(map(len,gd.values())),fractions)
 order=np.argsort(-(score+rng.uniform(0,1e-9,len(keys))),kind='mergesort');n=sum(map(len,gd.values()));assignment=np.full(n,'train',object);counts={'test':0,'val':0};targets={'test':fraction*n,'val':.1*n};current='test'
 for j in order:
  if current=='test' and counts['test']>=targets['test']:current='val'
  if current=='val' and counts['val']>=targets['val']:current='train'
  idx=gd[keys[j]];assignment[idx]=current
  if current in counts:counts[current]+=len(idx)
 return assignment

def prepare(spec):
 folder=OUT/spec.name;folder.mkdir(parents=True,exist_ok=True)
 dataset=folder/'dataset.csv.gz';cache=folder/'features.npz'
 if dataset.exists() and cache.exists():
  df=pd.read_csv(dataset);a=np.load(cache,allow_pickle=False);arrays={k:a[k] for k in a.files};return df,arrays
 print('Preparing',spec.name,flush=True);df=load_tdc_task(spec,ROOT/'data/raw/scaffold/data/external');df.insert(0,'row_index',np.arange(len(df)))
 smiles=df.smiles.tolist();rawdesc=np.stack([descriptor_vector(s) for s in smiles]);scaled,descmeta=norm(rawdesc)
 # Keep float64 descriptor geometry exactly as the original splitter; no downcast.
 med=np.nanmedian(rawdesc,axis=0);clean=np.where(np.isfinite(rawdesc),rawdesc,med);scaled=np.clip(RobustScaler(quantile_range=(10,90)).fit_transform(clean),-20,20)
 groups,ng=_cluster_acyclic_scaffolds(np.array([murcko_scaffold(s) for s in smiles],object),scaled)
 morgan=np.stack([morgan_fingerprint(s) for s in smiles]);svd=TruncatedSVD(n_components=min(32,len(df)-1),random_state=GEOMETRY_SEED);z=svd.fit_transform(morgan.astype(np.float64));morganz,morganmeta=norm(z)
 arrays={'morgan':morgan,'desc_geometry':scaled,'morgan_geometry':morganz,'groups':groups.astype(str),'raw_descriptors':rawdesc}
 if spec.name in ['DILI','BBB']:
  features=build_features(smiles);arrays.update({k:features[k] for k in ['physchem','maccs','combined']})
 np.savez_compressed(cache,**arrays);np.savez_compressed(folder/'svd_basis.npz',components=svd.components_,explained_variance=svd.explained_variance_ratio_)
 df.to_csv(dataset,index=False,compression={'method':'gzip','mtime':0})
 dump(folder/'dataset_config.json',{'task':dataclasses.asdict(spec),'source':rpath(ROOT/'data/raw/scaffold/data/external'/spec.path),'source_sha256':sha(ROOT/'data/raw/scaffold/data/external'/spec.path),'dataset_sha256':sha(dataset),'n':len(df),'geometry_seed':GEOMETRY_SEED,'svd_components':svd.n_components,'svd_explained_variance_sum':svd.explained_variance_ratio_.sum(),'acyclic_groups':ng,'descriptor_scaling':descmeta,'morgan_coordinate_scaling':morganmeta,'grouping':'unchanged descriptor-based Murcko/acyclic grouping'})
 return df,arrays

def make_splits(spec,df,a):
 folder=OUT/spec.name/'splits';folder.mkdir(exist_ok=True);geometry={};n=len(df)
 for geo in ['descriptor','morgan']:
  z=a['desc_geometry' if geo=='descriptor' else 'morgan_geometry'];keys,gd,q,r,l,c=score_geometry(z,a['groups']);geometry[geo]=(keys,gd,q)
  pd.DataFrame({'group':keys,'n':[len(gd[k]) for k in keys],'radius':r,'local_distance':l,'score':q}).to_csv(OUT/spec.name/f'{geo}_group_scores.csv',index=False)
 for frac in FRACTIONS:
  for seed in SEEDS:
   for regime in ['scaffold','descriptor','morgan']:
    keys,gd,q=geometry['descriptor' if regime=='scaffold' else regime];assignment=allocate(gd,keys,q,seed,frac,regime=='scaffold')
    mapped=dict(zip(keys,q));score=np.array([mapped[g] for g in a['groups']]);assert_split_integrity(SplitResult(assignment,a['groups'],score,{}))
    man=df[['row_index','connectivity_key','y']].copy();man['group']=a['groups'];man['partition']=assignment;man['frontier_score']=score
    p=folder/f'{regime}_test{round(frac*100):02d}_seed{seed}.csv.gz'
    if not p.exists():man.to_csv(p,index=False,compression={'method':'gzip','mtime':0})
    else:assert pd.read_csv(p).partition.tolist()==assignment.tolist()
    if frac==.2 and regime=='descriptor':
     old=ROOT/f'results/scaffold/results/splits/{spec.name}__frontier__seed{seed}.csv.gz'
     od=pd.read_csv(old);col='partition' if 'partition' in od else 'split';assert assignment.tolist()==od[col].tolist(),f'Original descriptor membership differs: {spec.name} seed{seed}'
   if spec.name=='DILI' and frac==.2:
    tr,va,te=stratified_random(df.y.to_numpy(),seed);assign=np.full(n,'train',object);assign[va]='val';assign[te]='test';man=df[['row_index','connectivity_key','y']].copy();man['group']=a['groups'];man['partition']=assign;man['frontier_score']=np.nan
    p=folder/f'random_test20_seed{seed}.csv.gz'
    if not p.exists():man.to_csv(p,index=False,compression={'method':'gzip','mtime':0})
 return folder

def common_inputs(a,representation,tr):
 if representation in ['morgan','maccs']:return a[representation].astype(np.float32),{'binary_fingerprints':'unchanged'}
 phys,scaling=norm(a['physchem'],tr)
 if representation=='physchem':return phys,scaling
 return np.hstack([a['maccs'],phys]).astype(np.float32),scaling

def save_run(spec,df,a,regime,fraction,seed,representation,model,recipe='common'):
 rid=f'{spec.name}__{regime}__test{round(fraction*100):02d}__{representation}__{model}__seed{seed}__{recipe}'
 folder=OUT/spec.name/'runs'/rid;done=folder/'summary.json'
 if done.exists() and json.loads(done.read_text()).get('status')=='complete':return
 folder.mkdir(parents=True,exist_ok=True);p=OUT/spec.name/'splits'/f'{regime}_test{round(fraction*100):02d}_seed{seed}.csv.gz';man=pd.read_csv(p);parts={k:np.flatnonzero(man.partition.to_numpy()==k) for k in ['train','val','test']};tr,va,te=[parts[k] for k in ['train','val','test']];y=df.y.to_numpy();t0=time.time();warns=[]
 x,scaling=common_inputs(a,representation,tr)
 if recipe=='standard_lr':
  sx=StandardScaler().fit(a[representation][tr]);x=sx.transform(a[representation]).astype(np.float32);scaling={'center':sx.mean_.tolist(),'scale':sx.scale_.tolist(),'kind':'StandardScaler'}
 config={'run_id':rid,'endpoint':spec.name,'regime':regime,'test_target':fraction,'validation_target':.1,'seed':seed,'representation':representation,'model':model,'recipe':recipe,'task_kind':spec.kind,'primary_metric':spec.primary_metric,'split_manifest':rpath(p),'split_sha256':sha(p),'dataset_sha256':sha(OUT/spec.name/'dataset.csv.gz'),'input_transformation':scaling,'script_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'docs/revision/EXPERIMENT_PROTOCOL.md')}
 if model=='mlp':config['training']=dataclasses.asdict(TrainConfig(seed=seed));config['objective']=dataclasses.asdict(ObjectiveConfig(method='ERM'))
 else:config['training']={'C':1.,'solver':'lbfgs','max_iter':2000,'penalty':'l2','class_weight':None}
 dump(folder/'config.json',config)
 try:
  if spec.kind=='binary' and len(np.unique(y[tr]))<2:raise ValueError('Single-class training partition')
  if model=='mlp':
   r=train_model(x,y,tr,va,te,spec.kind,spec.primary_metric,np.zeros((1,len(y)),dtype=int),np.zeros((1,1)),np.ones((1,1))*len(tr),ObjectiveConfig(method='ERM'),TrainConfig(seed=seed));pred=r.test_prediction;metrics=r.test_metrics;best_epoch=r.best_epoch;pd.DataFrame(r.history).to_csv(folder/'history.csv',index=False)
  else:
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always');clf=LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,random_state=seed);clf.fit(x[tr],y[tr]);pred=clf.predict_proba(x[te])[:,1];warns=[str(w.message) for w in caught]
   metrics=evaluate(spec.kind,y[te],pred);best_epoch=None
  predictions=df.iloc[te][['row_index','connectivity_key']].copy();predictions['y_true']=y[te];predictions['prediction']=pred;predictions.to_csv(folder/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
  summary={**{k:config[k] for k in ['run_id','endpoint','regime','test_target','seed','representation','model','recipe','primary_metric']},'status':'complete',**metrics,'best_epoch':best_epoch,'train_n':len(tr),'val_n':len(va),'test_n':len(te),'test_positive_fraction':float(np.mean(y[te])) if spec.kind=='binary' else None,'warnings':warns,'seconds':time.time()-t0}
  dump(done,summary);print(f'{rid} {spec.primary_metric}={metrics.get(spec.primary_metric):.4f} {time.time()-t0:.1f}s',flush=True)
 except Exception as e:
  dump(folder/'failure.json',{'run_id':rid,'status':'failed','error':repr(e)});raise

def run_architecture(spec,df,a):
 y=df.y.to_numpy(dtype=np.int64);tok=SMILESTokenizer();tokens=np.array([tok.encode(s) for s in df.smiles],dtype=np.int64)
 for regime in ['random','scaffold','descriptor','morgan']:
  for seed in SEEDS:
   p=OUT/spec.name/'splits'/f'{regime}_test20_seed{seed}.csv.gz';man=pd.read_csv(p);parts={k:np.flatnonzero(man.partition.to_numpy()==k) for k in ['train','val','test']};tr,va,te=[parts[k] for k in ['train','val','test']];phys,scale=norm(a['physchem'],tr);combined=np.hstack([a['maccs'],phys]).astype(np.float32)
   for kind in ['allmlp','moltrinet']:
    rid=f'DILI__{regime}__test20__architecture__{kind}__seed{seed}__common';folder=OUT/'DILI/runs'/rid;done=folder/'summary.json'
    if done.exists():continue
    folder.mkdir(parents=True,exist_ok=True);torch.manual_seed(seed);np.random.seed(seed);t0=time.time()
    model=SimpleMLPModel(input_dim=176) if kind=='allmlp' else MolTriNet(tok.vocab_size)
    xs=[torch.from_numpy(combined)] if kind=='allmlp' else [torch.from_numpy(tokens),torch.from_numpy(a['maccs'].astype(np.float32)),torch.from_numpy(phys)]
    ds=torch.utils.data.TensorDataset(*[x[tr] for x in xs],torch.from_numpy(y[tr]));loader=torch.utils.data.DataLoader(ds,batch_size=64,shuffle=True,drop_last=False)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);lossf=torch.nn.CrossEntropyLoss(label_smoothing=.1);best=-np.inf;bad=0;beststate=None;history=[];bestepoch=-1
    config={'run_id':rid,'seed':seed,'model':kind,'representation':'architecture','regime':regime,'test_target':.2,'split_manifest':rpath(p),'split_sha256':sha(p),'dataset_sha256':sha(OUT/'DILI/dataset.csv.gz'),'script_sha256':sha(__file__),'input_transformation':scale,'binary_MACCS':'unchanged','tokenisation':'canonical SMILES, max length 200, existing fixed vocabulary','n_params':n_params(model),'training':{'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,'label_smoothing':.1,'batch_size':64,'max_epochs':100,'patience':12,'clip':1,'selection':'validation AUROC','min_delta':1e-6}}
    dump(folder/'config.json',config)
    for epoch in range(100):
     model.train();losses=[]
     for batch in loader:
      if len(batch[-1])==1:continue # BatchNorm cannot learn a singleton batch.
      opt.zero_grad(set_to_none=True);loss=lossf(model(*batch[:-1]),batch[-1]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
     model.eval()
     with torch.no_grad():vp=torch.softmax(model(*[x[va] for x in xs]),1)[:,1].numpy()
     auc=roc_auc_score(y[va],vp);history.append({'epoch':epoch,'loss':np.mean(losses),'val_auroc':auc})
     if auc>best+1e-6:best=auc;bad=0;beststate=copy.deepcopy(model.state_dict());bestepoch=epoch
     else:bad+=1
     if bad>=12:break
    if beststate is None:raise RuntimeError('No valid architecture model selected')
    model.load_state_dict(beststate);model.eval()
    with torch.no_grad():pred=torch.softmax(model(*[x[te] for x in xs]),1)[:,1].numpy()
    pd.DataFrame(history).to_csv(folder/'history.csv',index=False);preds=df.iloc[te][['row_index','connectivity_key']].copy();preds['y_true']=y[te];preds['prediction']=pred;preds.to_csv(folder/'predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    summary={'run_id':rid,'endpoint':'DILI','regime':regime,'test_target':.2,'seed':seed,'representation':'architecture','model':kind,'recipe':'common','primary_metric':'auroc','status':'complete',**evaluate('binary',y[te],pred),'best_epoch':bestepoch,'epochs':len(history),'best_val_auroc':best,'train_n':len(tr),'val_n':len(va),'test_n':len(te),'seconds':time.time()-t0};dump(done,summary);print(f'{rid} AUROC={summary["auroc"]:.4f} {summary["seconds"]:.1f}s',flush=True)

def manifest():
 summaries=[json.loads(p.read_text()) for p in sorted(OUT.glob('*/runs/*/summary.json')) if not p.name.startswith('._')];pd.DataFrame(summaries).to_csv(OUT/'runs.csv',index=False)
 files=[p for p in OUT.rglob('*') if p.is_file() and not p.name.startswith('._')];dump(ROOT/'provenance/revision/experiments_sha256.json',{rpath(p):sha(p) for p in sorted(files)})

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['benchmark','grid','architecture','all'],default='all');parser.add_argument('--tasks',nargs='+',default=['DILI','BBB','HIA','CYP2D6','PPBR','Solubility']);parser.add_argument('--threads',type=int,default=2);args=parser.parse_args();torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);OUT.mkdir(parents=True,exist_ok=True)
 dump(OUT/f'environment_{args.mode}.json',{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'rdkit':rdkit.__version__,'torch':torch.__version__,'threads':args.threads,'command':sys.argv,'script_sha256':sha(__file__)})
 for name in args.tasks:
  spec=next(t for t in TASKS if t.name==name);df,a=prepare(spec);make_splits(spec,df,a)
  if args.mode in ['benchmark','all']:
   for frac in FRACTIONS:
    for regime in ['scaffold','descriptor','morgan']:
     for seed in SEEDS:save_run(spec,df,a,regime,frac,seed,'morgan','mlp')
  if args.mode in ['grid','all'] and name=='DILI':
   for regime in ['random','scaffold','descriptor','morgan']:
    for seed in SEEDS:
     for rep in ['physchem','maccs','morgan','combined']:
      for model in ['logistic','mlp']:save_run(spec,df,a,regime,.2,seed,rep,model)
  if args.mode in ['grid','all'] and name in ['DILI','BBB']:
   for regime in ['scaffold','descriptor','morgan']:
    for seed in SEEDS:
     for rep in ['physchem','morgan']:save_run(spec,df,a,regime,.2,seed,rep,'logistic','standard_lr')
  if args.mode in ['architecture','all'] and name=='DILI':run_architecture(spec,df,a)
  manifest();print('Finished stage',args.mode,name,flush=True)
 manifest()
if __name__=='__main__':main()
