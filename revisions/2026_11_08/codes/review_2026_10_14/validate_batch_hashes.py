"""Verify stored review-batch SHA-256 entries without changing any manifest."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/review_2026_10_14'
REPORT=ROOT/'provenance/review_2026_10_14/batch_hash_validation.json'
HEX=re.compile(r'^[0-9a-fA-F]{64}$')
def sha(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
 return digest.hexdigest()
def entries(obj):
 if isinstance(obj,list):
  for child in obj:yield from entries(child)
 elif isinstance(obj,dict):
  if 'path' in obj and 'sha256' in obj:
   assert isinstance(obj['path'],str) and HEX.fullmatch(obj['sha256'])
   yield obj['path'],obj['sha256'].lower(),obj.get('bytes')
  else:
   for key,value in obj.items():
    if isinstance(value,str) and HEX.fullmatch(value):
     assert '/' in key or Path(key).suffix,('Digest key is not a file path',key)
     yield key,value.lower(),None
    elif isinstance(value,(list,dict)):yield from entries(value)

def main():
 manifests=sorted(p for p in BASE.rglob('sha256*.json') if not p.name.startswith('._') and p.parent!=BASE and 'sensitivities' not in p.relative_to(BASE).parts)
 summaries=[];missing=[];mismatches=[];byte_mismatches=[];excluded=[];checked_paths=set();total=matched=0
 for manifest in manifests:
  counts={'entries':0,'matched':0,'missing':0,'mismatched':0,'excluded_appledouble':0,'readme_entries':0,'readme_mismatches':0,'science_or_provenance_entries':0,'science_or_provenance_mismatches':0}
  recorded=list(entries(json.loads(manifest.read_text())))
  assert len({path for path,_,_ in recorded})==len(recorded),('Duplicate entries',manifest)
  for relative,expected,expected_bytes in recorded:
   path=ROOT/relative
   if any(part.startswith('._') for part in path.parts):
    counts['excluded_appledouble']+=1;excluded.append({'manifest':str(manifest.relative_to(ROOT)),'path':relative});continue
   role='readme_documentation' if path.name.lower().startswith('readme') else 'science_or_provenance'
   counts['entries']+=1;total+=1;checked_paths.add(relative)
   counts['readme_entries' if role=='readme_documentation' else 'science_or_provenance_entries']+=1
   record={'manifest':str(manifest.relative_to(ROOT)),'path':relative,'role':role,'expected_sha256':expected}
   if not path.is_file():counts['missing']+=1;missing.append(record);continue
   actual=sha(path)
   if actual==expected:counts['matched']+=1;matched+=1
   else:
    counts['mismatched']+=1;counts['readme_mismatches' if role=='readme_documentation' else 'science_or_provenance_mismatches']+=1
    mismatches.append({**record,'actual_sha256':actual})
   if expected_bytes is not None and path.stat().st_size!=expected_bytes:
    byte_mismatches.append({**record,'expected_bytes':expected_bytes,'actual_bytes':path.stat().st_size})
  summaries.append({'manifest':str(manifest.relative_to(ROOT)),'manifest_sha256':sha(manifest),**counts})
 report={'status':'pass' if not missing and not mismatches and not byte_mismatches else 'fail','checked_entries':total,'checked_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Stored SHA-256 entries in existing review batch manifests; AppleDouble excluded; not a completeness inventory. New sensitivities and root-level comprehensive manifests intentionally outside this pass. No manifests, scientific outputs or manuscripts were modified.','manifest_count':len(manifests),'recorded_entries_checked':total,'unique_paths_checked':len(checked_paths),'matched_entries':matched,'missing_entries':len(missing),'mismatched_entries':len(mismatches),'readme_mismatches':sum(x['role']=='readme_documentation' for x in mismatches),'science_or_provenance_mismatches':sum(x['role']=='science_or_provenance' for x in mismatches),'appledouble_entries_excluded':len(excluded),'byte_count_mismatches':len(byte_mismatches),'all_science_or_provenance_hashes_match':not any(x['role']=='science_or_provenance' for x in mismatches) and not any(x['role']=='science_or_provenance' for x in missing),'batches':summaries,'missing':missing,'mismatches':mismatches,'recorded_byte_count_differences':byte_mismatches,'excluded':excluded,'validator_sha256':sha(__file__)}
 REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({key:value for key,value in report.items() if key not in ['batches','excluded','recorded_byte_count_differences']},indent=2))

if __name__=='__main__':main()
