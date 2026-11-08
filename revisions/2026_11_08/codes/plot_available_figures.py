"""Redraw the seven figures supported by included aggregate outputs; no fitting."""
from pathlib import Path
import sys,subprocess,os
ROOT=Path(__file__).resolve().parents[1]
if not os.environ.get('NATURE_FIGURE_SKILL'):
 raise SystemExit('Set NATURE_FIGURE_SKILL to the external nature-figure directory containing scripts/audit_panel_alignment.py and the PDF auditors.')
for folder in ['work/figure_redesign_2026_11_08','provenance/figures_2026_11_08','results/figures_2026_11_08']:(ROOT/folder).mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'codes/figures_2026_11_08'))
import benchmark_allocation as benchmark
from prepare_benchmark_allocation import prepare
prepare();benchmark.benchmark();benchmark.allocation()
import gradients
gradients.gradient();gradients.distance()
import composition_ranking
composition_ranking.ranking()
import supplementary_panels
supplementary_panels.curation();supplementary_panels.geometry()
qa=Path(os.environ['NATURE_FIGURE_SKILL'])/'scripts'
for pdf in (ROOT/'work/figure_redesign_2026_11_08').glob('*.pdf'):
 if pdf.name.startswith('._'):continue
 subprocess.run([sys.executable,str(qa/'audit_pdf_text.py'),str(pdf),'--min-pt','7'],check=True)
 subprocess.run([sys.executable,str(qa/'audit_figure_collisions.py'),str(pdf),'--json-out',str(ROOT/'provenance/figures_2026_11_08'/(pdf.stem+'.public_collision.json'))],check=True)
print('Redrew seven figures supported by aggregate tables. Full composition, support and BBB-group figures additionally require the molecular/group diagnostic outputs documented in the data README.')
