"""Check active revised figure assets, mirrors and immutable saved-result hashes."""
import os
from pathlib import Path
import hashlib,json,sys,subprocess,xml.etree.ElementTree as ET
import pymupdf as fitz
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];QA=ROOT/'provenance/figures_2026_11_08'
SKILL=(Path(os.environ['NATURE_FIGURE_SKILL'])/'scripts')
STEMS=['review_benchmark','review_composition','review_allocation','review_dili_ranking','review_atom_pair_support','review_curation','review_geometry_sensitivity','fig3_gradient','supp_distance','review_bbb_diagnostics']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 records=[]
 for stem in STEMS:
  files={ext:ROOT/'figures/final'/f'{stem}.{ext}' for ext in ['pdf','svg','png']}
  for ext,p in files.items():
   assert p.is_file()
   assert sha(p)==sha(ROOT/'manuscript/figures'/p.name)==sha(ROOT/'work/figure_redesign_2026_11_08'/p.name)
  doc=fitz.open(files['pdf']);assert len(doc)==1
  size=[doc[0].rect.width*25.4/72,doc[0].rect.height*25.4/72];assert abs(size[0]-170)<.01 and size[1]<=185.01,(stem,size)
  align=json.loads((QA/f'{stem}.alignment.json').read_text());assert align['verdict']=='PASS' and align['auditable']
  assert abs(align['layout']['figure']['width_pt']-doc[0].rect.width)<.01
  assert abs(align['layout']['figure']['height_pt']-doc[0].rect.height)<.01
  fonts=[]
  for f in doc[0].get_fonts(full=True):
   name,ext,kind,data=doc.extract_font(f[0]);assert data and kind!='Type3';fonts.append(name)
  text=subprocess.run([sys.executable,str(SKILL/'audit_pdf_text.py'),str(files['pdf']),'--min-pt','7','--json'],capture_output=True,text=True,check=True)
  glyph=json.loads(text.stdout);(QA/f'{stem}.final_glyphs.json').write_text(text.stdout)
  c=subprocess.run([sys.executable,str(SKILL/'audit_figure_collisions.py'),str(files['pdf']),'--json-out',str(QA/f'{stem}.final_collision.json')],capture_output=True,text=True,check=True)
  (QA/f'{stem}.final_collision.txt').write_text(c.stdout)
  assert ET.parse(files['svg']).findall('.//{http://www.w3.org/2000/svg}text')
  with Image.open(files['png']) as im:
   assert min(im.info['dpi'])>=299.9;pixels=im.size;dpi=im.info['dpi']
  records.append({'figure':stem,'dimensions_mm':size,'min_glyph_pt':glyph['minimum_found_pt'],'fonts':fonts,'png_pixels':pixels,'png_dpi':dpi,'sha256':{ext:sha(p) for ext,p in files.items()},'alignment':'PASS','collision':'PASS'})
 sources={}
 for filename in ['benchmark_allocation_source_trace.json','composition_ranking_sources.json','supplementary_source_trace.json']:
  data=json.loads((QA/filename).read_text());rows=data['sources'] if isinstance(data,dict) else data
  for row in rows:
   p=ROOT/row['source'];assert sha(p)==row['sha256'],row['source'];sources[row['source']]=row['sha256']
 contract=json.loads((QA/'gradient_contract.json').read_text());assert sha(ROOT/contract['source'])==contract['source_sha256'];sources[contract['source']]=contract['source_sha256']
 report={'status':'PASS','figures':records,'unchanged_saved_result_sources':sources,'checker_sha256':sha(Path(__file__)),'scope':'Final active assets and source hashes. Scientific/display transformations are independently reviewed separately.'}
 (QA/'final_export_inventory.json').write_text(json.dumps(report,indent=2)+'\n')
 print('PASS:',len(records),'figures,',len(sources),'unchanged saved-result sources')
if __name__=='__main__':main()
