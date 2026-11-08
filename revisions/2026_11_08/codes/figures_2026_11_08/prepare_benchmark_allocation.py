"""Reconstruct figure display tables from saved results without fitting models."""
from pathlib import Path
from hashlib import sha256
from io import StringIO
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'results/review_2026_10_14'
DISPLAY = ROOT / 'results/figures_2026_11_08'
QA = ROOT / 'provenance/figures_2026_11_08'
MODELS = ['physchem:logistic', 'maccs:logistic', 'morgan:logistic', 'morgan:rf',
          'morgan:mlp', 'graph:gnn', 'chemprop:dmpnn']
METRIC = {e: 'mae' if e in ['PPBR', 'Solubility'] else 'auprc' if e == 'CYP2D6'
          else 'auroc' for e in ['BBB', 'HIA', 'CYP2D6', 'DILI', 'PPBR', 'Solubility']}


def prepare():
    trace = []
    def read(relative):
        path = SOURCE / relative
        frame = pd.read_csv(path)
        trace.append({'source': str(path.relative_to(ROOT)), 'rows': len(frame),
                      'sha256': sha256(path.read_bytes()).hexdigest()})
        return frame

    absolute = pd.concat([read(f'{name}/model_intervals.csv') for name in
                          ['benchmark_uncertainty', 'chemprop_uncertainty']])
    absolute = absolute[(absolute.endpoint == 'BBB') & (absolute.subset == 'all') &
                        (absolute.scheme == 'cluster') & (absolute.metric == 'auroc') &
                        absolute.model.isin(MODELS)]
    contrasts = pd.concat([read(f'{name}/frontier_minus_scaffold.csv') for name in
                           ['benchmark_uncertainty', 'chemprop_uncertainty']])
    contrasts = contrasts[(contrasts.subset == 'all') & (contrasts.scheme == 'cluster')]
    contrasts = contrasts[contrasts.apply(lambda row:
        row.metric == METRIC[row.endpoint] and row.model in
        [model.replace('logistic', 'ridge') if row.endpoint in ['PPBR', 'Solubility']
         else model for model in MODELS], axis=1)]
    observations = read('allocation_controls/runs.csv')
    observations['display_value'] = [row[METRIC[row.endpoint]] for _, row in observations.iterrows()]
    summary = read('allocation_controls/allocation_summary.csv')
    assert (len(absolute), len(contrasts), len(observations), len(summary)) == (20, 78, 1344, 24)
    assert observations.loc[observations['mode'] == 'exact_random'].groupby(
        ['endpoint', 'geometry', 'representation']).size().eq(50).all()

    tables = {'figure1_bbb_absolute_display.csv': absolute,
              'figure1_endpoint_contrasts_display.csv': contrasts,
              'figure3_allocation_observations_display.csv': observations,
              'figure3_allocation_summary_display.csv': summary}
    verified = []
    pending = []
    for filename, frame in tables.items():
        buffer = StringIO()
        frame.to_csv(buffer, index=False)
        contents = buffer.getvalue()
        # Compare exactly after the same CSV serialization/read path, so a
        # stale display table cannot silently be overwritten by a redraw.
        expected = pd.read_csv(StringIO(contents))
        target = DISPLAY / filename
        before = sha256(target.read_bytes()).hexdigest() if target.exists() else None
        if target.exists():
            pd.testing.assert_frame_equal(pd.read_csv(target), expected,
                                          check_exact=True, check_dtype=True)
        pending.append((target, contents))
        verified.append({'display': str(target.relative_to(ROOT)), 'rows': len(frame),
                         'existing_frame_equal': before is not None, 'sha256_before': before})
    # Verify every existing table before writing any table.
    DISPLAY.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    for target, contents in pending:
        target.write_text(contents)
    for item, (target, _) in zip(verified, pending):
        item['sha256_after'] = sha256(target.read_bytes()).hexdigest()
        item['existing_bytes_unchanged'] = item['sha256_before'] == item['sha256_after']
    report = {'sources': trace, 'displays': verified,
              'preparation_script_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
              'plot_script_sha256': sha256((Path(__file__).parent/'benchmark_allocation.py').read_bytes()).hexdigest(),
              'filters': 'Figure1 original seven-model registry, all test subset, cluster scheme and endpoint primary metric; Figure3 all saved allocation records and all summary rows.',
              'missing': 'No saved Morgan-frontier graph-control fit; descriptor-reverse HIA AUROC remains undefined. No raw values imputed.'}
    (QA/'benchmark_allocation_extraction_trace.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    prepare()
    print('Reconstructed four display CSVs; existing frame equality checked before writes.')
