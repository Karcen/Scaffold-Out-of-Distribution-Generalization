"""Compound-resampling uncertainty and support diagnostics for completed runs."""
from pathlib import Path
import sys,json,hashlib,os,argparse
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'results/revision/experiments';OUT=ROOT/'results/revision/analysis';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'codes/scaffold'));from src.chem import murcko_scaffold
B=2000;SEED=20261012

def dump(p,o):Path(p).write_text(json.dumps(o,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
def load_new():
 rows=[json.loads(p.read_text()) for p in EXP.glob('*/runs/*/summary.json')];d=pd.DataFrame(rows);d.to_csv(EXP/'runs.csv',index=False);return d

def weighted_scores(y,p,w):
 """Vectorised weighted ROC AUC / average precision; ties grouped exactly."""
 order=np.argsort(-p,kind='mergesort');s=p[order];yy=y[order];starts=np.r_[0,np.flatnonzero(np.diff(s)!=0)+1]
 wp=np.add.reduceat(w[:,order]*yy,starts,axis=1);wn=np.add.reduceat(w[:,order]*(1-yy),starts,axis=1)
 cp=np.cumsum(wp,axis=1);cn=np.cumsum(wn,axis=1);pt=cp[:,-1];nt=cn[:,-1]
 auc=np.sum(wp*(nt[:,None]-cn+.5*wn),axis=1)/(pt*nt)
 ap=np.sum(wp*np.divide(cp,cp+cn,out=np.zeros_like(cp),where=(cp+cn)>0),axis=1)/pt
 return auc,ap

def validate_weighted():
 from sklearn.metrics import average_precision_score
 y=np.array([0,1,0,1,0,1]);p=np.array([.1,.8,.8,.4,.1,.4]);w=np.array([[1,2,1,1,3,1],[0,1,3,2,1,1]],float);a,b=weighted_scores(y,p,w)
 assert np.allclose(a,[roc_auc_score(y,p,sample_weight=v) for v in w]);assert np.allclose(b,[average_precision_score(y,p,sample_weight=v) for v in w])

def bootstrap_family(name,runs,loader,subset=None):
 # Each regime has a compound universe; identical IDs share a weight across all fits.
 metrics=[];contrasts=[];drawmap={};subsets=['all'] if subset is None else ['all','cyclic']
 for regime,rd in runs.groupby('regime',sort=False):
  pred=[]
  for _,row in rd.iterrows():
   p=loader(row);p=p.rename(columns={'y_prob':'prediction'});p['row_index']=p.row_index.astype(int);pred.append((row,p))
  for sub in subsets:
   selected=[]
   for row,p in pred:
    if sub=='cyclic':p=p[p.row_index.isin(subset)]
    selected.append((row,p))
   labels=pd.concat([p[['row_index','y_true']] for _,p in selected]).drop_duplicates()
   assert labels.groupby('row_index').y_true.nunique().max()==1
   labels=labels.drop_duplicates('row_index').sort_values('row_index');ids=labels.row_index.to_numpy();y=labels.y_true.to_numpy(dtype=int);index={int(v):i for i,v in enumerate(ids)};strata=[np.flatnonzero(y==v) for v in [0,1]]
   if min(map(len,strata))==0:continue
   rng=np.random.default_rng(SEED);rdraw={r.run_id:np.empty((B,2)) for r,_ in selected}
   for begin in range(0,B,100):
    size=min(100,B-begin);weights=np.zeros((size,len(ids)),dtype=float)
    for ix in strata:weights[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=size)
    for row,p in selected:
     ix=np.array([index[int(v)] for v in p.row_index]);aa,ap=weighted_scores(p.y_true.to_numpy(int),p.prediction.to_numpy(float),weights[:,ix]);rdraw[row.run_id][begin:begin+size]=np.c_[aa,ap]
   grouping={}
   for row,p in selected:
    rep=row.get('representation','architecture');key=(rep,row.model,row.get('recipe','historical'));grouping.setdefault(key,[]).append((row,p))
   for key,rs in grouping.items():
    rep,model,recipe=key;arr=np.nanmean([rdraw[r.run_id] for r,p in rs],axis=0);unit=(name,regime,sub,rep,model,recipe);drawmap[unit]=arr
    vals=[]
    for r,p in rs:vals.append(weighted_scores(p.y_true.to_numpy(int),p.prediction.to_numpy(float),np.ones((1,len(p))))[0:2])
    point=np.array([[v[0][0],v[1][0]] for v in vals]);m=point.mean(axis=0)
    for k,metric in enumerate(['auroc','auprc']):
     lo,hi=np.nanquantile(arr[:,k],[.025,.975]);metrics.append({'family':name,'regime':regime,'subset':sub,'representation':rep,'model':model,'recipe':recipe,'metric':metric,'mean':m[k],'seed_sd':point[:,k].std(ddof=1),'lower':lo,'upper':hi,'test_compound_union_n':len(ids),'per_run_test_n_min':min(len(p) for r,p in rs),'per_run_test_n_max':max(len(p) for r,p in rs),'seeds':len(rs),'bootstrap_reps':B,'bootstrap_seed':SEED})
   keys=list(grouping)
   for i,key1 in enumerate(keys):
    for key2 in keys[i+1:]:
     if key1[2]!=key2[2]:continue
     delta=drawmap[(name,regime,sub,*key1)]-drawmap[(name,regime,sub,*key2)]
     for k,metric in enumerate(['auroc','auprc']):
      m1=next(x['mean'] for x in metrics if (x['family'],x['regime'],x['subset'],x['representation'],x['model'],x['recipe'],x['metric'])==(name,regime,sub,*key1,metric));m2=next(x['mean'] for x in metrics if (x['family'],x['regime'],x['subset'],x['representation'],x['model'],x['recipe'],x['metric'])==(name,regime,sub,*key2,metric));lo,hi=np.nanquantile(delta[:,k],[.025,.975]);contrasts.append({'family':name,'regime':regime,'subset':sub,'first':f'{key1[0]}:{key1[1]}','second':f'{key2[0]}:{key2[1]}','recipe':key1[2],'metric':metric,'delta_first_minus_second':m1-m2,'lower':lo,'upper':hi,'multiplicity_adjusted':False})
   print('Bootstrap',name,regime,sub,'runs',len(selected),flush=True)
 pd.DataFrame(metrics).to_csv(OUT/f'bootstrap_{name}.csv',index=False);pd.DataFrame(contrasts).to_csv(OUT/f'contrasts_{name}.csv',index=False)
 return pd.DataFrame(metrics)

def bootstrap_all(d, families):
 validate_weighted();df=pd.read_csv(EXP/'DILI/dataset.csv.gz');cyclic=set(df.loc[df.smiles.map(murcko_scaffold)!='__ACYCLIC__','row_index'])
 for name,folder in [('historical_grid','grid_dili'),('historical_architecture','arch_dili')]:
  if name not in families:continue
  rs=pd.read_csv(ROOT/f'results/repro/{folder}/runs.csv');preds=pd.read_csv(ROOT/f'results/repro/{folder}/predictions.csv.gz');bootstrap_family(name,rs,lambda row:preds[preds.run_id==row.run_id],cyclic)
 nd=d[(d.endpoint=='DILI')&(d.test_target==.2)&(d.recipe=='common')]
 for name,sel in [('unified_grid',nd.representation!='architecture'),('unified_architecture',nd.representation=='architecture')]:
  if name in families:bootstrap_family(name,nd[sel],lambda row:pd.read_csv(EXP/row.endpoint/'runs'/row.run_id/'predictions.csv.gz'),cyclic)
 if 'bbb_controls' in families:
  bbb=d[(d.endpoint=='BBB')&(d.test_target==.2)];bootstrap_family('bbb_controls',bbb,lambda row:pd.read_csv(EXP/row.endpoint/'runs'/row.run_id/'predictions.csv.gz'))
 # Teacher compound bootstrap per regime. Cross-regime contrast kept descriptive
 # because these domains differ; no false matched-test inference is introduced.
 if 'teacher' in families:
  rs=pd.read_csv(ROOT/'results/repro/arch_teacher/runs.csv');preds=pd.read_csv(ROOT/'results/repro/arch_teacher/predictions.csv.gz');bootstrap_family('teacher',rs,lambda row:preds[preds.run_id==row.run_id])


def distances(d):
 rows=[]
 for endpoint,ed in d[d.model.eq('mlp')&d.representation.eq('morgan')&d.recipe.eq('common')].groupby('endpoint'):
  a=np.load(EXP/endpoint/'features.npz');
  for _,r in ed.iterrows():
   p=EXP/endpoint/'splits'/f'{r.regime}_test{round(r.test_target*100):02d}_seed{r.seed}.csv.gz';m=pd.read_csv(p);tr=np.flatnonzero(m.partition=='train');te=np.flatnonzero(m.partition=='test');rec={k:r[k] for k in ['run_id','endpoint','regime','test_target','seed','primary_metric','train_n','test_n']};rec['performance']=r[r.primary_metric]
   for field in ['desc_geometry','morgan_geometry']:
    nn=NearestNeighbors(n_neighbors=1,algorithm='ball_tree',n_jobs=1).fit(a[field][tr]);di=nn.kneighbors(a[field][te])[0][:,0];rec[field+'_nn_mean']=di.mean();rec[field+'_nn_median']=np.median(di)
   rows.append(rec)
  print('Distances',endpoint,len(ed),flush=True)
 pd.DataFrame(rows).to_csv(OUT/'difficulty_distance_runs.csv',index=False)

def diagnostics(d):
 rows=[];individual=[];ranks=[]
 for endpoint in ['BBB','DILI']:
  data=pd.read_csv(EXP/endpoint/'dataset.csv.gz');a=np.load(EXP/endpoint/'features.npz');x=a['morgan'].astype(np.float32);bitcounts=x.sum(axis=1);y=data.y.to_numpy();cyclic=data.smiles.map(murcko_scaffold).to_numpy()!='__ACYCLIC__'
  for regime in ['descriptor','morgan']:
   for seed in range(5):
    m=pd.read_csv(EXP/endpoint/'splits'/f'{regime}_test20_seed{seed}.csv.gz');tr=np.flatnonzero(m.partition=='train');te=np.flatnonzero(m.partition=='test');pred=pd.read_csv(EXP/endpoint/'runs'/f'{endpoint}__{regime}__test20__morgan__mlp__seed{seed}__common'/'predictions.csv.gz').set_index('row_index').loc[te].prediction.to_numpy();inter=x[te]@x[tr].T;den=bitcounts[te,None]+bitcounts[tr][None,:]-inter;sim=np.divide(inter,den,out=np.zeros_like(inter),where=den>0);arg=sim.argmax(axis=1);maxsim=sim[np.arange(len(te)),arg];nn_y=y[tr[arg]];q=m.frontier_score.to_numpy()[te];bins=pd.qcut(q,5,labels=False,duplicates='drop')
    frame=pd.DataFrame({'endpoint':endpoint,'regime':regime,'seed':seed,'row_index':te,'y_true':y[te],'prediction':pred,'frontier_score':q,'quintile':np.asarray(bins)+1,'cyclic':cyclic[te],'max_training_tanimoto':maxsim,'nearest_train_row':tr[arg],'nearest_train_label':nn_y,'nearest_label_agrees':nn_y==y[te]});individual.append(frame)
    for sub,subframe in [('all',frame),('cyclic',frame[frame.cyclic])]:
     for qn,g in subframe.groupby('quintile'):
      row={'endpoint':endpoint,'regime':regime,'seed':seed,'subset':sub,'quintile':qn,'n':len(g),'positives':g.y_true.sum(),'positive_fraction':g.y_true.mean(),'mean_prediction':g.prediction.mean(),'mean_max_tanimoto':g.max_training_tanimoto.mean(),'nn_label_agreement':g.nearest_label_agrees.mean(),'auroc':roc_auc_score(g.y_true,g.prediction) if g.y_true.nunique()==2 else np.nan};rows.append(row)
     pos=subframe[subframe.y_true==1];neg=subframe[subframe.y_true==0]
     if not len(pos) or not len(neg):continue
     dif=pos.prediction.to_numpy()[:,None]-neg.prediction.to_numpy()[None,:];correct=(dif>0)+.5*(dif==0);within=pos.quintile.to_numpy()[:,None]==neg.quintile.to_numpy()[None,:]
     ranks.append({'endpoint':endpoint,'regime':regime,'seed':seed,'subset':sub,'n':len(subframe),'positive_n':len(pos),'auroc':correct.mean(),'within_quintile_auroc':correct[within].mean(),'cross_quintile_auroc':correct[~within].mean(),'within_pair_fraction':within.mean(),'mean_nearest_similarity':subframe.max_training_tanimoto.mean(),'nearest_label_agreement':subframe.nearest_label_agrees.mean()})
 pd.DataFrame(rows).to_csv(OUT/'frontier_quintile_diagnostics.csv',index=False);pd.concat(individual).to_csv(OUT/'frontier_molecule_diagnostics.csv.gz',index=False,compression={'method':'gzip','mtime':0});pd.DataFrame(ranks).to_csv(OUT/'ranking_decomposition.csv',index=False)

def aggregate(d):
 keys=['endpoint','regime','test_target','representation','model','recipe','primary_metric'];summary=d.groupby(keys)[['auroc','auprc','mae']].agg(['mean','std','count']);summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(OUT/'performance_summary.csv',index=False)
 delta=[]
 for endpoint,ed in d[d.representation.eq('morgan')&d.model.eq('mlp')&d.recipe.eq('common')].groupby('endpoint'):
  for frac,fd in ed.groupby('test_target'):
   control=fd[fd.regime=='scaffold'].set_index('seed')
   if len(control)!=5:continue
   for regime in ['descriptor','morgan']:
    f=fd[fd.regime==regime].set_index('seed');metric=f.primary_metric.iloc[0]
    for seed in f.index:
     delta.append({'endpoint':endpoint,'test_target':frac,'regime':regime,'seed':seed,'metric':metric,'scaffold':control.loc[seed,metric],'frontier':f.loc[seed,metric],'delta_frontier_minus_scaffold':f.loc[seed,metric]-control.loc[seed,metric]})
 pd.DataFrame(delta).to_csv(OUT/'paired_split_changes.csv',index=False)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['all','bootstrap','diagnostics','distances'],default='all');parser.add_argument('--families',nargs='+',default=['historical_grid','historical_architecture','unified_grid','unified_architecture','bbb_controls','teacher']);args=parser.parse_args();d=load_new();aggregate(d)
 if args.mode in ['all','bootstrap']:bootstrap_all(d,args.families)
 if args.mode in ['all','diagnostics']:diagnostics(d)
 if args.mode in ['all','distances']:distances(d)
 dump(OUT/'analysis_config.json',{'bootstrap_replicates':B,'bootstrap_seed':SEED,'confidence_level':.95,'estimand':'mean of per-seed metrics conditional on fitted models and realised partitions','resampling':'class-stratified compound IDs, shared weights across seeds/models within each regime','multiplicity':'pairwise intervals are descriptive and unadjusted','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
 dump(ROOT/'provenance/revision/analysis_sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and not p.name.startswith('._')})
if __name__=='__main__':main()
