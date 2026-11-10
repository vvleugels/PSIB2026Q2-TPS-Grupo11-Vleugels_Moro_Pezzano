"""One readable document with every group result: results/report.md.

Collects what the statistics and figures steps wrote to results/ (tables, hypotheses,
figures and their captions) in reading order, so the results can be read or shared
without the GUI. Written by the figures step (python -m tps.analysis.figures).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tps.analysis.hypotheses import HYPOTHESES
from tps.analysis.regions import region_table

REGION_COLUMNS = ['band', 'region', 'n', 'mean_diff', 'ci95_low', 'ci95_high', 'cohens_dz', 'rank_biserial',
                  'p_wilcoxon', 'p_fdr_wilcoxon', 'p_ttest', 'p_fdr_ttest']
FIGURES = [
    ('topomap_open', 'Topomaps, eyes open (primary)'),
    ('topomap_closed', 'Topomaps, eyes closed (secondary)'),
    ('topomap_relative_open', 'Relative-power topomaps, eyes open'),
    ('topomap_relative_closed', 'Relative-power topomaps, eyes closed'),
    ('psd_regions_open', 'PSD by region, eyes open'),
    ('psd_regions_closed', 'PSD by region, eyes closed'),
    ('paired_open', 'Paired regional band power, eyes open'),
    ('paired_closed', 'Paired regional band power, eyes closed'),
    ('paired_relative_open', 'Paired regional relative power, eyes open'),
    ('paired_relative_closed', 'Paired regional relative power, eyes closed'),
    ('theta_beta', 'Frontal theta/beta ratio'),
]


def cell(value) -> str:
    """Markdown cell text: p-values and effects rounded, missing values blank."""
    if isinstance(value, (bool, np.bool_)):
        return 'yes' if value else 'no'
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ''
        if float(value).is_integer() and abs(value) < 1e6:
            return str(int(value))
        return f'{value:.3g}'
    return str(value).replace('|', '/')


def md_table(table: pd.DataFrame, columns: list[str] | None = None) -> str:
    """A pandas table as a Markdown table (no extra dependency)."""
    if table is None or table.empty:
        return '_No rows._'
    columns = [c for c in (columns or list(table.columns)) if c in table.columns]
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns)]
    lines += ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in table[columns].itertuples(index=False)]
    return '\n'.join(lines)


def read(results_dir: Path, name: str) -> pd.DataFrame | None:
    path = results_dir / name
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def captions(figures_dir: Path) -> dict[str, str]:
    """{figure: caption} from figures/captions.md."""
    path = figures_dir / 'captions.md'
    if not path.is_file():
        return {}
    out = {}
    for block in path.read_text(encoding='utf-8').split('\n## ')[1:]:
        name, _, body = block.partition('\n')
        lines = [line for line in body.strip().splitlines() if not line.startswith(('File:', 'Archivo:'))]
        out[name.strip()] = '\n'.join(lines).strip()
    return out


def build_report(results_dir: Path) -> str:
    """Report text from the files in results_dir (missing parts are noted, not fatal)."""
    figures_dir = results_dir / 'figures'
    parts = [
        '# SD vs NS: group results',
        '',
        f'Generated {datetime.now():%Y-%m-%d %H:%M} from `{results_dir.name}/` by `python -m tps.analysis.figures`.',
        'Eyes open is the primary analysis and eyes closed the secondary; they are never pooled.',
        'Group-level spectral differences only; no clinical interpretation. «Accepted» epochs only',
        'mean «within the amplitude limits», not visual inspection. Details: `docs/pipeline.md`.',
        '',
        'Descriptive figures and differences are available from one complete NS/SD pair. '
        'Inferential paired and cluster tests require at least three pairs (project policy); '
        'missing p-values mean not tested, not evidence of no difference.', '',
        '## Region mapping', '', md_table(region_table()), '',
    ]

    participants = read(results_dir, 'stats_participants.csv')
    parts += ['## Participants', '']
    if participants is not None:
        counts = (participants.groupby(['analysis', 'eyes'])['included']
                  .agg(included='sum', listed='size').reset_index())
        parts += [md_table(counts), '', 'Not included:', '',
                  md_table(participants[~participants['included'].astype(bool)], ['eyes', 'sub', 'reason']), '']
    else:
        parts += ['_Run the statistics step first._', '']

    parts += ['## Epochs used per session', '', md_table(read(results_dir, 'stats_epochs.csv')), '']

    log_power = read(results_dir, 'stats_region_log_power.csv')
    for eyes, label in (('open', 'eyes open (primary)'), ('closed', 'eyes closed (secondary)')):
        parts += [f'## Band power by region (log10 µV²), SD − NS, {label}', '',
                  md_table(None if log_power is None else log_power[log_power['eyes'] == eyes], REGION_COLUMNS), '']
    relative = read(results_dir, 'stats_region_rel_power.csv')
    parts += ['## Relative power (%), SD − NS', '',
              md_table(relative, ['eyes'] + REGION_COLUMNS), '',
              '## Absolute vs relative power', '',
              md_table(read(results_dir, 'stats_relative_vs_absolute.csv'),
                       ['eyes', 'band', 'region', 'mean_diff_absolute', 'p_fdr_wilcoxon_absolute',
                        'mean_diff_relative', 'p_fdr_wilcoxon_relative', 'same_direction']), '',
              '## Sensitivity: recordings with ≥ 20 clean epochs', '',
              md_table(read(results_dir, 'stats_region_log_power_min20.csv'),
                       ['eyes', 'band', 'region', 'n', 'mean_diff', 'cohens_dz', 'p_fdr_wilcoxon']), '',
              '## Frontal theta/beta ratio', '', md_table(read(results_dir, 'stats_frontal_theta_beta.csv')), '']

    order = read(results_dir, 'stats_session_order.csv')
    parts += ['## Session order check (10 smallest p)', '',
              md_table(None if order is None or 'p_mannwhitney' not in order
                       else order.sort_values('p_mannwhitney').head(10)), '']

    clusters = read(results_dir, 'cluster_results.csv')
    parts += ['## Significant electrode clusters', '',
              md_table(None if clusters is None else clusters[clusters['significant'].astype(bool)],
                       ['eyes', 'band', 'sign', 'n_channels', 't_sum', 'p_cluster', 'n_subjects', 'channels']), '']

    parts += ['## Hypotheses', '', 'Defined in `docs/hypotheses.md` (references are TODO until verified).', '']
    parts += [md_table(pd.DataFrame([dict(id=h.id, bands=', '.join(h.bands), regions=', '.join(h.regions),
                                          expected=h.direction or 'none', eyes_focus=h.eyes_focus,
                                          statement=h.statement) for h in HYPOTHESES])), '']
    summary = read(figures_dir, 'hypothesis_summary.csv')
    parts += [md_table(summary, ['hypothesis', 'eyes', 'band', 'region', 'expected', 'observed', 'n', 'mean_diff',
                                 'p_corrected', 'rank_biserial', 'cohens_dz', 'agreement', 'largest_region']), '']

    notes = captions(figures_dir)
    parts += ['## Figures', '']
    for stem, title in FIGURES:
        if (figures_dir / f'{stem}.png').is_file():
            parts += [f'### {title}', '', f'![{title}](figures/{stem}.png)', '', notes.get(stem, ''), '']
    return '\n'.join(parts).rstrip() + '\n'


def write_report(results_dir: Path) -> Path:
    path = Path(results_dir) / 'report.md'
    path.write_text(build_report(Path(results_dir)), encoding='utf-8')
    return path
