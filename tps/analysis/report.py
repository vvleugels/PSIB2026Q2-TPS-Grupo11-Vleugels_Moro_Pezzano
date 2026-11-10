"""One readable document with the group results: results/report.md (in Spanish).

Collects what the batch, statistics and figures steps wrote to results/ in reading order:
data used, the summary table (eyes open), hypotheses, figures, and eyes closed as an
advanced section. Written by the figures step (python -m tps.analysis.figures).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from tps.analysis.hypotheses import HYPOTHESES

BAND_ES = {'delta': 'delta', 'theta': 'theta', 'alpha': 'alfa', 'beta': 'beta', 'theta/beta': 'theta/beta'}
CRITERION_ES = {'manual': 'marcado a mano', 'flat': 'señal plana', 'deviation': 'amplitud anómala',
                'correlation': 'baja correlación', 'high_frequency': 'ruido de alta frecuencia',
                'peak_to_peak': 'pico a pico en > 30 % de las épocas', 'non_finite': 'muestras no finitas'}
RESULT_ES = {'supported': 'apoyada', 'opposite': 'opuesta', 'not significant': 'no significativa',
             'exploratory: significant': 'exploratoria: significativa',
             'exploratory: not significant': 'exploratoria: no significativa', 'not tested': 'sin prueba'}
DIRECTION_ES = {'increase': 'aumento', 'decrease': 'disminución', 'no change': 'sin cambio',
                'none (exploratory)': 'sin predicción', 'n/a': '—'}


def cell(value) -> str:
    """Markdown cell text: numbers rounded, missing values blank, booleans as Sí/No."""
    if isinstance(value, (bool, np.bool_)):
        return 'Sí' if value else 'No'
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ''
        if float(value).is_integer() and abs(value) < 1e6:
            return str(int(value))
        return f'{value:.3g}'
    return str(value).replace('|', '/')


def md_table(table: pd.DataFrame | None, columns: dict[str, str] | None = None) -> str:
    """A pandas table as Markdown; ``columns`` maps CSV column -> heading (and sets the order)."""
    if table is None or table.empty:
        return '_Sin filas._'
    columns = columns or {c: c for c in table.columns}
    columns = {c: h for c, h in columns.items() if c in table.columns}
    lines = ['| ' + ' | '.join(columns.values()) + ' |', '|' + '---|' * len(columns)]
    lines += ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in table[list(columns)].itertuples(index=False)]
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


def summary_view(summary: pd.DataFrame | None, eyes: str) -> pd.DataFrame | None:
    """The summary rows of one eye condition, ready to print (Spanish bands, CI in one column)."""
    if summary is None:
        return None
    part = summary[summary['eyes'] == eyes].copy()
    part['band'] = part['band'].map(lambda b: BAND_ES.get(b, b))
    part['ci'] = [f'{lo:+.1f} a {hi:+.1f}' if np.isfinite(lo) else ''
                  for lo, hi in zip(part['ci95_low_pct'], part['ci95_high_pct'])]
    for column in ('mean_ns_db', 'mean_sd_db'):
        part[column] = [f'{v:.1f}' if np.isfinite(v) else '' for v in part[column]]
    for column in ('change_db', 'change_pct'):
        part[column] = [f'{v:+.1f}' if np.isfinite(v) else '' for v in part[column]]
    part['p'] = [p_text(v) for v in part['p']]
    part['significant'] = part['significant'].astype(bool)
    return part


def p_text(value: float) -> str:
    if not np.isfinite(value):
        return ''
    return '< 0,001' if value < 0.001 else f'{value:.3f}'.replace('.', ',')


SUMMARY_HEADINGS = {'band': 'Banda', 'region': 'Región', 'n': 'n', 'mean_ns_db': 'Media NS (dB)',
                    'mean_sd_db': 'Media SD (dB)', 'change_db': 'Cambio (dB)', 'change_pct': 'Cambio (%)',
                    'ci': 'IC 95 % (%)', 'p': 'p', 'significant': 'Significativo'}


def data_section(results_dir: Path) -> list[str]:
    """Participants per eye condition, excluded recordings and bad channels per criterion."""
    parts = ['## Datos', '']
    features = read(results_dir, 'features_long.csv')
    if features is not None:
        rows = []
        for eyes, label in (('open', 'Ojos abiertos'), ('closed', 'Ojos cerrados')):
            sessions = features[features['eyes'] == eyes].groupby('sub')['condition'].nunique()
            rows.append(dict(eyes=label, paired=int((sessions == 2).sum()), single=int((sessions == 1).sum())))
        parts += [md_table(pd.DataFrame(rows), {'eyes': 'Condición', 'paired': 'Participantes con NS y SD',
                                               'single': 'Con una sola sesión (fuera del análisis)'}), '']
    exclusions = read(results_dir, 'exclusions.csv')
    if exclusions is not None:
        recordings = exclusions[exclusions['level'] == 'recording'].copy()
        recordings['eyes'] = recordings['eyes'].map({'open': 'abiertos', 'closed': 'cerrados'})
        parts += ['Registros excluidos:', '',
                  md_table(recordings, {'sub': 'Participante', 'condition': 'Sesión', 'eyes': 'Ojos',
                                        'reason': 'Motivo'}), '']
        channels = exclusions[exclusions['level'] == 'channel']
        if 'criterion' in channels:
            counts = (channels['criterion'].fillna('peak_to_peak').map(lambda c: CRITERION_ES.get(c, c))
                      .value_counts().rename_axis('criterion').reset_index(name='count'))
            parts += ['Canales malos excluidos, por criterio (canal × registro):', '',
                      md_table(counts, {'criterion': 'Criterio', 'count': 'Canales'}), '']
    return parts


def build_report(results_dir: Path) -> str:
    """Report text from the files in results_dir (missing parts are noted, not fatal)."""
    figures_dir = results_dir / 'figures'
    summary = read(results_dir, 'stats_summary.csv')
    notes = captions(figures_dir)
    parts = [
        '# SD vs NS: resultados del grupo', '',
        f'Generado el {datetime.now():%Y-%m-%d %H:%M} desde `{results_dir.name}/` por '
        '`python -m tps.analysis.figures`. Ojos abiertos es el análisis principal; ojos cerrados, el '
        'secundario (sección «Avanzado»). Nunca se combinan. Solo diferencias espectrales a nivel de grupo, '
        'sin interpretación clínica. Método: `docs/pipeline.md`.', '',
        'Cada fila compara, para los mismos participantes, la potencia absoluta de una banda en una región '
        'tras privación de sueño (SD) frente a sueño normal (NS). **Cambio (%)** = cuánto cambió la potencia '
        '(media geométrica); **Cambio (dB)** = la misma diferencia en decibeles. **p** = prueba t pareada '
        'sobre el log10 de la potencia; **Significativo** = p < 0,05. Con 20 pruebas por condición de ojos '
        'se espera, por azar, alrededor de un resultado «significativo».', '',
    ]
    parts += data_section(results_dir)
    parts += ['## Resumen · ojos abiertos', '', md_table(summary_view(summary, 'open'), SUMMARY_HEADINGS), '']

    parts += ['## Hipótesis', '', 'Definidas en `docs/hypotheses.md` (referencias pendientes de verificar).', '']
    parts += ['\n'.join(f'- **{h.id}**: {h.statement}' for h in HYPOTHESES), '']
    hypotheses = read(figures_dir, 'hypothesis_summary.csv')
    if hypotheses is not None:
        hypotheses = hypotheses.copy()
        hypotheses['eyes'] = hypotheses['eyes'].map({'open': 'abiertos', 'closed': 'cerrados'})
        hypotheses['band'] = hypotheses['band'].map(lambda b: BAND_ES.get(b, b))
        for column in ('expected', 'observed'):
            hypotheses[column] = hypotheses[column].map(lambda v: DIRECTION_ES.get(v, v))
        hypotheses['agreement'] = hypotheses['agreement'].map(lambda v: RESULT_ES.get(v, v))
        hypotheses['change_pct'] = [f'{v:+.1f}' if np.isfinite(v) else '' for v in hypotheses['change_pct']]
        hypotheses['change_db'] = [f'{v:+.1f}' if np.isfinite(v) else '' for v in hypotheses['change_db']]
        hypotheses['p'] = [p_text(v) for v in hypotheses['p']]
    parts += [md_table(hypotheses, {'hypothesis': 'Hipótesis', 'eyes': 'Ojos', 'band': 'Banda',
                                    'region': 'Región', 'expected': 'Esperado', 'observed': 'Observado',
                                    'n': 'n', 'change_pct': 'Cambio (%)', 'change_db': 'Cambio (dB)',
                                    'p': 'p', 'agreement': 'Resultado'}), '']

    def figures(eyes: str) -> list[str]:
        out = []
        for stem, title in ((f'psd_cz_{eyes}', 'Espectro en Cz'), (f'topomaps_{eyes}', 'Topomapas NS y SD')):
            if (figures_dir / f'{stem}.png').is_file():
                out += [f'### {title}', '', f'![{title}](figures/{stem}.png)', '', notes.get(stem, ''), '']
        return out

    parts += ['## Figuras · ojos abiertos', ''] + figures('open')
    parts += ['## Avanzado · ojos cerrados', '',
              'Subgrupo de participantes con registros de ojos cerrados; análisis secundario.', '',
              md_table(summary_view(summary, 'closed'), SUMMARY_HEADINGS), ''] + figures('closed')
    return '\n'.join(parts).rstrip() + '\n'


def write_report(results_dir: Path) -> Path:
    path = Path(results_dir) / 'report.md'
    path.write_text(build_report(Path(results_dir)), encoding='utf-8')
    return path
