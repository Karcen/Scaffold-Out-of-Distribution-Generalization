"""Retrieve public TDC inputs and link the unchanged preceding split reference.

This helper does not fit a model or overwrite an existing input. Run explicitly
with --download to retrieve missing raw tables; input hashes are always recorded.
"""
from pathlib import Path
from urllib.request import urlopen
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
FILES={'BBB_Martins.tab':4259566,'HIA_Hou.tab':4259591,'CYP2D6_Veith.tab':4259580,'PPBR_AZ.tab':6413140,'Solubility_AqSolDB.tab':4259610,'DILI.tab':4259585}
def main():
 p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');p.add_argument('--reference-root',type=Path,default=ROOT.parents[1]);a=p.parse_args()
 raw=ROOT/'data/raw/scaffold/data/external';raw.mkdir(parents=True,exist_ok=True);records=[]
 for name,fid in FILES.items():
  dest=raw/name;prior=a.reference_root/'data/external'/name
  if not dest.exists() and prior.is_file():shutil.copyfile(prior,dest)
  if not dest.exists() and a.download:
   with urlopen(f'https://dataverse.harvard.edu/api/access/datafile/{fid}',timeout=120) as response:dest.write_bytes(response.read())
  records.append({'file':str(dest.relative_to(ROOT)),'dataverse_file_id':fid,'present':dest.exists(),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest() if dest.exists() else None})
 old=a.reference_root/'results';link=ROOT/'results/scaffold/results'
 if old.is_dir() and (old/'splits').is_dir() and not link.exists():
  link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(old.resolve(),target_is_directory=True)
 manifest=ROOT/'provenance/public_input_access.json';manifest.parent.mkdir(exist_ok=True,parents=True);manifest.write_text(json.dumps({'inputs':records,'reference_split_link_present':link.exists(),'reference_note':'Existing repository split manifests are a historical consistency check, not newly generated results.'},indent=2)+'\n')
 missing=[r['file'] for r in records if not r['present']]
 if missing:raise SystemExit('Missing inputs; pass --download or provide the unchanged tables: '+', '.join(missing))
 if not link.exists():raise SystemExit('Provide --reference-root pointing to the preceding public repository, including results/splits.')
 print('Public input access paths prepared; verify recorded hashes before fitting.')
if __name__=='__main__':main()
