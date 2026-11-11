"""Result figures for the SD vs NS comparison, saved to results/figures/ with captions.

Run from the project folder:  python -m tps.analysis.figures [--results-dir results]

Two figures per eye condition, following the reference paper (Xiang et al., 2024):
* psd_cz_<eyes>: group spectrum at Cz in dB, NS vs SD, mean ± SD (paper's Fig. 4);
* topomaps_<eyes>: band-power maps for NS and SD (paper's Fig. 5).
make_all also writes hypothesis_summary.csv/.md, captions.md and results/report.md.
Figures show group-level spectral differences only; no clinical interpretation is implied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from tps import config
from tps.analysis.hypotheses import evaluate
from tps.core.spectral import montage_channels, standard_info   # shared with the explorer

# One colour per condition everywhere (same teal/rose as the GUI's spectral plot).
COLORS = {'NS': '#0e7c86', 'SD': '#b94b6b'}
LABELS = {'NS': 'NS (sueño normal)', 'SD': 'SD (privación de sueño)'}
EYES_ES = {'open': 'Ojos abiertos', 'closed': 'Ojos cerrados'}
EYES_ROLE = {'open': 'Ojos abiertos (análisis principal)', 'closed': 'Ojos cerrados (análisis secundario)'}
BAND_ES = {'delta': 'Delta', 'theta': 'Theta', 'alpha': 'Alfa', 'beta': 'Beta'}
BAND_LIMITS = dict(zip(config.BANDS, zip(config.SPECTRAL.edges, config.SPECTRAL.edges[1:])))
PSD_CHANNEL = 'Cz'   # the paper's spectrum electrode
FONT = {'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11, 'legend.fontsize': 10,
        'xtick.labelsize': 10, 'ytick.labelsize': 10}
DPI = 150


def _figure(width: float, height: float) -> Figure:
    """Figure without pyplot, so figures can be built in tests/threads without a display."""
    import matplotlib
    matplotlib.rcParams.update(FONT)
    return Figure(figsize=(width, height), layout='constrained')


def _save(fig: Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{name}.png'
    fig.savefig(path, dpi=DPI)
    return path


# --------------------------------------------------------------------------- spectrum
def paired_psd(psd: pd.DataFrame, eyes: str, channel: str = PSD_CHANNEL) -> tuple[np.ndarray, dict]:
    """Frequencies and {condition: participants x frequencies} PSD in dB, paired participants only."""
    meta = ['sub', 'ses', 'condition', 'eyes', 'channel']
    freq_columns = [c for c in psd.columns if c not in meta]
    frequencies = np.array([float(c) for c in freq_columns])
    data = psd[(psd['eyes'] == eyes) & (psd['channel'] == channel)]
    by = {cond: data[data['condition'] == cond].set_index('sub')[freq_columns] for cond in ('NS', 'SD')}
    subjects = sorted(set(by['NS'].index) & set(by['SD'].index))
    values = {cond: 10 * np.log10(np.clip(by[cond].loc[subjects].to_numpy(dtype=float), 1e-12, None))
              for cond in ('NS', 'SD')}
    return frequencies, values


def plot_psd_cz(psd: pd.DataFrame, eyes: str, out_dir: Path) -> Figure | None:
    """Group PSD at Cz in dB for NS and SD: mean (line) ± SD (shading), like the paper's Fig. 4."""
    frequencies, values = paired_psd(psd, eyes)
    n = len(values['NS'])
    if n == 0:
        return None
    fig = _figure(9, 5)
    ax = fig.add_subplot(111)
    for cond in ('NS', 'SD'):
        mean = values[cond].mean(axis=0)
        ax.plot(frequencies, mean, color=COLORS[cond], lw=2, label=LABELS[cond])
        if n >= 2:
            spread = values[cond].std(axis=0, ddof=1)
            ax.fill_between(frequencies, mean - spread, mean + spread, color=COLORS[cond], alpha=.18, lw=0)
    for edge in config.SPECTRAL.edges:
        ax.axvline(edge, color='#9aa7aa', lw=.8, ls=':')
    ax.set_xlim(config.SPECTRAL.edges[0], min(frequencies[-1], 45))
    ax.set_xlabel('Frecuencia (Hz)')
    ax.set_ylabel('Potencia (dB, re 1 µV²/Hz)')
    ax.set_title(f'{EYES_ES[eyes]}: espectro en {PSD_CHANNEL} (n = {n})')
    ax.grid(alpha=.25)
    ax.legend()
    _save(fig, out_dir, f'psd_cz_{eyes}')
    return fig


# --------------------------------------------------------------------------- topomaps
def condition_maps(features: pd.DataFrame, eyes: str) -> tuple[pd.DataFrame, int]:
    """Per band x channel: mean band power in dB for NS and SD over participants with both sessions."""
    data = features[(features['eyes'] == eyes) & (features['abs_power_uV2'] > 0)].copy()
    data['db'] = 10 * np.log10(data['abs_power_uV2'])
    wide = data.pivot_table(index=['sub', 'band', 'channel'], columns='condition', values='db',
                            aggfunc='first').reindex(columns=['NS', 'SD']).dropna()
    n = wide.index.get_level_values('sub').nunique()
    return wide.groupby(['band', 'channel'])[['NS', 'SD']].mean().reset_index(), n


def plot_topomaps(features: pd.DataFrame, eyes: str, out_dir: Path) -> Figure | None:
    """Band power maps (dB) for NS (top row) and SD (bottom row), like the paper's Fig. 5.

    Each band has one colour scale shared by NS and SD, so the two rows can be compared."""
    import mne
    maps, n = condition_maps(features, eyes)
    channels = montage_channels(sorted(set(maps['channel'])))
    if n == 0 or len(channels) < 4:
        return None
    info = standard_info(channels)
    fig = _figure(4 * len(config.BANDS), 7.4)
    axes = fig.subplots(2, len(config.BANDS))
    for column, band in enumerate(config.BANDS):
        part = maps[maps['band'] == band].set_index('channel').reindex(channels)
        low = float(np.nanmin(part[['NS', 'SD']].to_numpy()))
        high = float(np.nanmax(part[['NS', 'SD']].to_numpy()))
        image = None
        for row, cond in enumerate(('NS', 'SD')):
            ax = axes[row, column]
            image, _ = mne.viz.plot_topomap(np.nan_to_num(part[cond].to_numpy(), nan=low), info, axes=ax,
                                            show=False, cmap='viridis', vlim=(low, high), contours=4,
                                            sensors=True)
            if row == 0:
                lo, hi = BAND_LIMITS[band]
                ax.set_title(f'{BAND_ES[band]} ({lo:g}–{hi:g} Hz)')
            if column == 0:
                ax.text(-0.18, 0.5, cond, transform=ax.transAxes, rotation=90, va='center', ha='center',
                        fontsize=13, color=COLORS[cond], fontweight='bold')
        bar = fig.colorbar(image, ax=list(axes[:, column]), location='bottom', shrink=.8, pad=.02)
        bar.set_label('dB (re 1 µV²)')
    fig.suptitle(f'{EYES_ES[eyes]}: potencia por banda, NS (arriba) y SD (abajo) · n = {n}', fontsize=13)
    _save(fig, out_dir, f'topomaps_{eyes}')
    return fig


# --------------------------------------------------------------------------- hypotheses
HYPOTHESIS_COLUMNS = ['hypothesis', 'eyes', 'band', 'region', 'expected', 'observed', 'n',
                      'change_pct', 'change_db', 'p', 'agreement', 'largest_region']


def hypothesis_summary(summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Hypothesis table (CSV + Markdown): expected vs observed direction, change and p."""
    table = evaluate(summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / 'hypothesis_summary.csv', index=False)
    lines = ['| ' + ' | '.join(HYPOTHESIS_COLUMNS) + ' |', '|' + '---|' * len(HYPOTHESIS_COLUMNS)]
    for _, row in table.reindex(columns=HYPOTHESIS_COLUMNS).iterrows():
        cells = [f'{v:.3g}' if isinstance(v, float) else str(v) for v in row]
        lines.append('| ' + ' | '.join(cells) + ' |')
    notes = ('\n\n`change_pct` / `change_db` = cambio SD − NS de la potencia (theta/beta: del cociente). '
             '`p` = prueba t pareada sobre log10. `largest_region` = región con el mayor |cambio %| en esa banda.\n')
    (out_dir / 'hypothesis_summary.md').write_text('\n'.join(lines) + notes, encoding='utf-8')
    return table


# ----------------------------------------------------------------------------- captions
def captions(inputs: dict) -> dict[str, str]:
    """Figure captions in Spanish (shown in the GUI and in the report)."""
    text = {}
    for eyes in ('open', 'closed'):
        text[f'psd_cz_{eyes}'] = (
            f'{EYES_ROLE[eyes]}. Espectro de potencia en {PSD_CHANNEL} para NS (verde azulado) y SD (rosa), '
            'en dB (10·log10 µV²/Hz): línea = media del grupo, sombra = ± 1 desvío estándar entre '
            'participantes. Welch por época (segmentos Hann de 2 s, 50 % de solapamiento, 0,5 Hz de '
            'resolución). Solo participantes con ambas sesiones y Cz válido en las dos. Líneas punteadas: '
            f'límites de las bandas ({", ".join(f"{e:g}" for e in config.SPECTRAL.edges)} Hz). '
            'Equivale a la Fig. 4 de Xiang et al. (2024).')
        text[f'topomaps_{eyes}'] = (
            f'{EYES_ROLE[eyes]}. Potencia absoluta por banda en cada electrodo, en dB (10·log10 µV²), '
            'promediada sobre los participantes con ambas sesiones: NS arriba y SD abajo. Cada banda usa una '
            'misma escala de color para NS y SD, así las dos filas se pueden comparar directamente. '
            'La zona de menor potencia en el centro refleja la cercanía a la referencia FCz (el paper '
            're-referencia al promedio; acá se conserva la referencia original). '
            'Posiciones: montaje estándar 10-05 de MNE. Equivale a la Fig. 5 de Xiang et al. (2024), '
            'que muestra theta, alfa y beta; acá se agrega delta porque la hipótesis H1 la incluye.')
    text['hypothesis_summary'] = (
        'Hipótesis H1–H4 (docs/hypotheses.md) frente a los cambios observados: dirección esperada y '
        'observada de SD − NS, cambio de potencia en % y en dB, y p de la prueba t pareada. «Apoyada» = '
        'signo esperado y p < 0,05. Solo diferencias espectrales a nivel de grupo.')
    processing = inputs.get('processing')
    if processing and 'filter' in processing:
        from tps.core.processing import FilterConfig
        used = processing['filter']
        config_used = FilterConfig(used['enabled'], used['low'], used['high'], used.get('notch'))
        label = 'sin filtrar' if not config_used.active else config_used.label.removeprefix('Filtrada · ')
        detail = (' Procesamiento: ' + label
                  + {'confirmed': ' + ICA confirmada.',
                     'available': ' + ICA confirmada donde existe (el resto, solo filtros).'}.get(
                         processing.get('ica_mode'), '; sin ICA.'))
        text = {key: value + detail for key, value in text.items()}
    return text


def write_captions(text: dict[str, str], out_dir: Path) -> Path:
    lines = ['# Leyendas de las figuras', '',
             'Generadas por `python -m tps.analysis.figures`. Solo diferencias espectrales a nivel de grupo, '
             'sin interpretación clínica.', '']
    for name, caption in text.items():
        file = f'{name}.md / .csv' if name == 'hypothesis_summary' else f'{name}.png'
        lines += [f'## {name}', '', f'Archivo: `{file}`', '', caption, '']
    path = out_dir / 'captions.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


# --------------------------------------------------------------------------------- main
def load_inputs(results_dir: Path) -> dict:
    """Read the batch and statistics outputs; missing optional files become None."""
    def read(name: str) -> pd.DataFrame | None:
        path = results_dir / name
        if not path.is_file():
            return None
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return None
    inputs = dict(features=read('features_long.csv'), summary=read('stats_summary.csv'),
                  psd=read('psd_channels.csv'))
    if inputs['features'] is None:
        raise FileNotFoundError('Run python -m tps.pipeline.batch and python -m tps.analysis.stats first.')
    try:
        inputs['processing'] = json.loads((results_dir / 'config_used.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        inputs['processing'] = None
    return inputs


def make_all(results_dir: Path, out_dir: Path | None = None) -> dict[str, Figure | pd.DataFrame]:
    """Build the figures, the hypothesis table and captions.md; returns them by name."""
    out_dir = out_dir or results_dir / 'figures'
    inputs = load_inputs(results_dir)
    made: dict[str, Figure | pd.DataFrame] = {}
    # Figures of earlier versions would otherwise stay in the folder and in the GUI list.
    for old in out_dir.glob('*.png') if out_dir.is_dir() else ():
        if not old.stem.startswith(('psd_cz_', 'topomaps_')):
            old.unlink()
    for eyes in ('open', 'closed'):
        if not (inputs['features']['eyes'] == eyes).any():
            continue
        if inputs['psd'] is not None and len(inputs['psd']):
            figure = plot_psd_cz(inputs['psd'], eyes, out_dir)
            if figure is not None:
                made[f'psd_cz_{eyes}'] = figure
        figure = plot_topomaps(inputs['features'], eyes, out_dir)
        if figure is not None:
            made[f'topomaps_{eyes}'] = figure
    if inputs['summary'] is not None:
        made['hypothesis_summary'] = hypothesis_summary(inputs['summary'], out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_captions({k: v for k, v in captions(inputs).items() if k in made}, out_dir)
    if out_dir == results_dir / 'figures':   # the report links figures/<name>.png
        from tps.analysis.report import write_report
        write_report(results_dir)
    return made


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Result figures for SD vs NS.')
    parser.add_argument('--results-dir', type=Path, default=config.RESULTS_DIR)
    parser.add_argument('--out-dir', type=Path, default=None, help='default: <results-dir>/figures')
    args = parser.parse_args(argv)
    made = make_all(args.results_dir, args.out_dir)
    print('Saved:', ', '.join(made), '->', args.out_dir or args.results_dir / 'figures')
    return 0


if __name__ == '__main__':
    sys.exit(main())
