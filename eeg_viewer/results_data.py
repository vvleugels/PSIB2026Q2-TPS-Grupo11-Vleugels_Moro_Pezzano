"""Read-only access to the group results in results/ (batch, statistics, figures).

Uses only the csv module, so the GUI can show results without pandas; pandas is only
needed to *compute* the statistics (requirements-analysis.txt).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import re

# Tables the GUI can browse: file stem -> Spanish label (order = selector order).
TABLES = {
    'stats_region_log_power': 'Potencia por región (log10), SD vs NS',
    'stats_region_rel_power': 'Potencia relativa por región, SD vs NS',
    'stats_frontal_theta_beta': 'Theta/beta frontal, SD vs NS',
    'stats_session_order': 'Control del orden de sesión',
    'stats_epochs': 'Epochs usadas por sesión',
    'stats_participants': 'Participantes incluidos',
    'cluster_results': 'Clusters por electrodo',
    'exclusions': 'Exclusiones del lote',
    'recordings_summary': 'Resumen por registro',
    'features_region': 'Potencia por región y participante',
    'tomas_stats_table': 'Estadística regional absoluta (Tomás)',
    'tomas_stats_relative_vs_absolute': 'Relativa vs absoluta por región (Tomás)',
    'tomas_group_summary': 'Resumen del grupo (Tomás)',
    'tomas_epochs_and_exclusions': 'Epochs y exclusiones por sesión (Tomás)',
}
# Tomás's group analysis (scripts/analyze_group.py) writes here; its tables and figures are
# shown in the same tab as ours instead of in a separate read-only tab.
GROUP_ANALYSIS_DIR = Path('group20-channel-2026') / 'analysis'
# Tables that do not live at results/<name>.csv.
TABLE_PATHS = {
    'tomas_stats_table': GROUP_ANALYSIS_DIR / 'stats_table.csv',
    'tomas_stats_relative_vs_absolute': GROUP_ANALYSIS_DIR / 'stats_relative_vs_absolute.csv',
    'tomas_group_summary': GROUP_ANALYSIS_DIR / 'group_summary.csv',
    'tomas_epochs_and_exclusions': GROUP_ANALYSIS_DIR / 'epochs_and_exclusions_by_session_condition.csv',
}
# Readable Spanish column headings; unknown columns keep their own name.
HEADINGS = {
    'sub': 'Participante', 'ses': 'Sesión', 'condition': 'Condición', 'eyes': 'Ojos',
    'analysis': 'Análisis', 'measure': 'Medida', 'band': 'Banda', 'region': 'Región',
    'channel': 'Electrodo', 'channels': 'Electrodos', 'epoch': 'Epoch', 'level': 'Nivel',
    'reason': 'Motivo', 'n': 'n', 'mean_ns': 'Media NS', 'mean_sd': 'Media SD',
    'mean_diff': 'Dif. media SD−NS', 'median_diff': 'Dif. mediana', 'ci95_low': 'IC95 inf.',
    'ci95_high': 'IC95 sup.', 'cohens_dz': 'd_z', 'rank_biserial': 'r biserial',
    'wilcoxon_W': 'W', 'p_wilcoxon': 'p Wilcoxon', 'p_fdr_wilcoxon': 'p FDR Wilcoxon',
    't': 't', 'p_ttest': 'p t', 'p_fdr_ttest': 'p FDR t', 'included': 'Incluido',
    'excluded': 'Excluido', 'n_epochs_used': 'Epochs usadas', 'n_epochs_rejected': 'Epochs rechazadas',
    'n_epochs_complete': 'Epochs completas', 'n_bad_channels': 'Canales malos', 'n_channels': 'Canales',
    'file': 'Archivo', 'log10_power': 'log10 potencia', 'rel_power_pct': 'Potencia rel. (%)',
    'sign': 'Signo', 't_sum': 'Suma t', 'p_cluster': 'p cluster', 'significant': 'Significativo',
    'n_subjects': 'Participantes', 'cluster': 'Cluster', 'p_mannwhitney': 'p Mann-Whitney',
    'p_fdr_mannwhitney': 'p FDR', 'n_ns_first': 'n NS primero', 'n_sd_first': 'n SD primero',
    'hypothesis': 'Hipótesis', 'expected': 'Esperado', 'observed': 'Observado',
    'p_corrected': 'p corregido', 'agreement': 'Acuerdo', 'largest_region': 'Región mayor efecto',
}
FILTER_COLUMNS = ('eyes', 'ses', 'band', 'region', 'sub')
BANDS = ('delta', 'theta', 'alpha', 'beta')
BATCH_EYES = {'eyesopen': 'open', 'eyesclosed': 'closed'}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Rows of a CSV file as dicts of strings; [] if the file does not exist."""
    if not path.is_file():
        return []
    # utf-8-sig also reads files written with a byte-order mark (Tomás's CSVs have one).
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def number(text: str) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def fmt(column: str, text: str) -> str:
    """Display form of a cell: p-values and effects rounded, booleans in Spanish."""
    if text in ('True', 'False'):
        return 'Sí' if text == 'True' else 'No'
    value = number(text)
    if value is None or not re.search(r'\d', text) or column in ('sub', 'ses', 'epoch', 'cluster'):
        return text
    if column.startswith('p_') or '_p_' in column:   # p-values: 3 decimals, tiny ones as < 0.001
        return '< 0.001' if value < 0.001 else f'{value:.3f}'
    if float(value).is_integer() and abs(value) < 1e6:
        return str(int(value))
    return f'{value:.3g}' if abs(value) < 1e-2 or abs(value) >= 1e4 else f'{value:.3f}'


def filter_rows(rows: list[dict], **filters: str | None) -> list[dict]:
    """Keep rows whose column equals each non-empty filter value (missing columns are ignored)."""
    active = {k: v for k, v in filters.items() if v}
    return [r for r in rows if all(k not in r or r[k] == v for k, v in active.items())]


def sort_rows(rows: list[dict], column: str, descending: bool = False) -> list[dict]:
    """Sort numerically when possible (blank values last), otherwise as text."""
    def key(row):
        value = number(row.get(column, ''))
        return (value is None, value if value is not None else 0, row.get(column, ''))
    return sorted(rows, key=key, reverse=descending)


@dataclass
class StepStatus:
    """State of one pipeline step's outputs."""
    done: bool
    updated: datetime | None
    detail: str
    stale: bool = False


@dataclass
class StudySummary:
    """Batch result of one recording, for the sidebar card."""
    found: bool
    excluded: bool = False
    reason: str = ''
    n_used: int | None = None
    n_rejected: int | None = None
    bad_channels: dict[str, str] = field(default_factory=dict)
    rejected_epochs: dict[int, str] = field(default_factory=dict)   # 0-based index -> reason
    powers: dict[str, dict[str, tuple[float, float]]] = field(default_factory=dict)  # ch -> band -> (abs, rel)
    theta_beta: dict[str, float] = field(default_factory=dict)


class Results:
    """Everything the GUI shows about results/; cheap to re-create after a pipeline run."""

    def __init__(self, directory: Path, project: Path | None = None):
        self.directory = Path(directory)
        self.project = Path(project) if project else self.directory.parent
        self.figures_dir = self.directory / 'figures'
        self._cache: dict[str, list[dict]] = {}

    def table_path(self, name: str) -> Path:
        return self.directory / TABLE_PATHS.get(name, Path(f'{name}.csv'))

    def table(self, name: str) -> list[dict[str, str]]:
        if name not in self._cache:
            self._cache[name] = read_csv(self.table_path(name))
        return self._cache[name]

    def available_tables(self) -> list[str]:
        return [name for name in TABLES if self.table_path(name).is_file()]

    # ------------------------------------------------------------------ pipeline state
    def _mtime(self, name: str) -> datetime | None:
        path = self.directory / name
        return datetime.fromtimestamp(path.stat().st_mtime) if path.is_file() else None

    def config_changed(self) -> bool:
        """True when the batch outputs were made with a different config.py than the current one."""
        try:
            used = json.loads((self.directory / 'config_used.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return False
        try:
            from batch.run_batch import config_hash   # project module; absent in some test setups
        except Exception:
            return False
        return used.get('config_hash') != config_hash()

    def status(self) -> dict[str, StepStatus]:
        """Done / missing / outdated state of the batch, statistics and figures steps."""
        summary = self.table('recordings_summary')
        batch_time = self._mtime('features_long.csv')
        excluded = sum(r.get('excluded') == 'True' for r in summary)
        batch = StepStatus(batch_time is not None, batch_time,
                           f'{len(summary)} registros, {excluded} excluidos' if summary else 'Sin resultados',
                           stale=self.config_changed())
        stats_time = self._mtime('stats_region_log_power.csv')
        participants = self.table('stats_participants')
        included = {(r['eyes'], r['sub']) for r in participants if r.get('included') == 'True'}
        eo = sum(1 for e, _ in included if e == 'open')
        ec = sum(1 for e, _ in included if e == 'closed')
        stats = StepStatus(stats_time is not None, stats_time,
                           f'{eo} participantes con ojos abiertos, {ec} con ojos cerrados'
                           if stats_time else 'Sin resultados',
                           stale=bool(stats_time and batch_time and stats_time < batch_time))
        figure_time = self._mtime('figures/captions.md')
        figures = StepStatus(figure_time is not None, figure_time,
                             f'{len(self.figure_list())} figuras' if figure_time else 'Sin figuras',
                             stale=bool(figure_time and stats_time and figure_time < stats_time))
        return dict(batch=batch, stats=stats, figures=figures)

    # ----------------------------------------------------------------------- figures
    def figure_list(self) -> list[Path]:
        """PNG figures in a stable, meaningful order (ours first, eyes open first), then Tomás's."""
        if not self.figures_dir.is_dir():
            return self.group_figures()
        order = ['topomap_open', 'topomap_closed', 'psd_regions_open', 'psd_regions_closed',
                 'paired_open', 'paired_closed', 'theta_beta']
        files = {p.stem: p for p in self.figures_dir.glob('*.png')}
        ours = [files[n] for n in order if n in files] + sorted(p for n, p in files.items() if n not in order)
        return ours + self.group_figures()

    def group_figures(self) -> list[Path]:
        """Tomás's PNGs (topomaps and paired plots, absolute and relative), after ours."""
        folder = self.directory / GROUP_ANALYSIS_DIR
        if not folder.is_dir():
            return []
        eyes = {'open': 0, 'closed': 1}
        bands = {'delta': 0, 'theta': 1, 'alpha': 2, 'beta': 3}

        def key(path: Path):
            parts = path.stem.split('_')
            return (parts[0] != 'topomap', 'relative' in parts,
                    min((eyes[p] for p in parts if p in eyes), default=9),
                    min((bands[p] for p in parts if p in bands), default=9), path.stem)
        return sorted(folder.glob('*.png'), key=key)

    def captions(self) -> dict[str, str]:
        """{figure name: caption text} parsed from figures/captions.md."""
        path = self.figures_dir / 'captions.md'
        if not path.is_file():
            return {}
        text = path.read_text(encoding='utf-8')
        out = {}
        for block in re.split(r'^## ', text, flags=re.M)[1:]:
            name, _, body = block.partition('\n')
            body = re.sub(r'^File: `[^`]*`\s*', '', body.strip())
            out[name.strip()] = body.strip()
        return out

    # ----------------------------------------------------------------- hypotheses
    def hypothesis_rows(self) -> list[dict[str, str]]:
        return read_csv(self.figures_dir / 'hypothesis_summary.csv')

    # --------------------------------------------------------------------- studies
    def study_summary(self, sub: str, ses: str, task: str) -> StudySummary:
        """Batch outcome for one recording (sub-01, ses-1, eyesopen)."""
        eyes = BATCH_EYES.get(task, task)
        match = [r for r in self.table('recordings_summary')
                 if (r['sub'], r['ses'], r['eyes']) == (sub, ses, eyes)]
        if not match:
            return StudySummary(found=False)
        row = match[0]
        result = StudySummary(found=True, excluded=row.get('excluded') == 'True',
                              n_used=int(number(row.get('n_epochs_used')) or 0) if row.get('n_epochs_used') else None,
                              n_rejected=int(number(row.get('n_epochs_rejected')) or 0)
                              if row.get('n_epochs_rejected') else None)
        for r in self.table('exclusions'):
            if (r['sub'], r['ses'], r['eyes']) != (sub, ses, eyes):
                continue
            if r['level'] == 'recording':
                result.reason = r['reason']
            elif r['level'] == 'channel':
                result.bad_channels[r['channel']] = r['reason']
            elif r['level'] == 'epoch' and r['epoch'] and 'incomplete final epoch' not in r['reason']:
                result.rejected_epochs[int(r['epoch']) - 1] = r['reason']
        for r in self._study_features(sub, ses, eyes):
            absolute, relative = number(r['abs_power_uV2']), number(r['rel_power_pct'])
            result.powers.setdefault(r['channel'], {})[r['band']] = (absolute, relative)
        for r in self.table('theta_beta'):
            if (r['sub'], r['ses'], r['eyes']) == (sub, ses, eyes) and number(r['ratio']) is not None:
                result.theta_beta[r['channel']] = number(r['ratio'])
        return result

    def _study_features(self, sub: str, ses: str, eyes: str) -> list[dict]:
        # features_long.csv has ~48k rows: index it once by recording instead of scanning per study.
        if '_features_index' not in self._cache:
            index: dict[tuple, list] = {}
            for r in self.table('features_long'):
                index.setdefault((r['sub'], r['ses'], r['eyes']), []).append(r)
            self._cache['_features_index'] = index   # type: ignore[assignment]
        return self._cache['_features_index'].get((sub, ses, eyes), [])   # type: ignore[union-attr]
