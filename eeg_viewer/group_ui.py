"""Read-only group results tab for the already generated analysis directory."""
from __future__ import annotations

import csv
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from PIL import Image


EYE_VALUES = {'Ojos abiertos': 'open', 'Ojos cerrados': 'closed'}
BAND_VALUES = {'Delta': 'delta', 'Theta': 'theta', 'Alfa': 'alpha', 'Beta': 'beta'}
FIGURE_VALUES = {'Pares': 'pairs', 'Topomapa': 'topomap'}
POWER_VALUES = {'Absoluta': 'absolute', 'Relativa': 'relative'}


def _read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def group_view_files(analysis_dir, eye, band, figure_type, power_type):
    root = Path(analysis_dir)
    eye_slug = EYE_VALUES[eye]
    band_slug = BAND_VALUES[band]
    figure_slug = FIGURE_VALUES[figure_type]
    power_slug = POWER_VALUES[power_type]
    prefix = f'{figure_slug}_' if power_slug == 'absolute' else f'{figure_slug}_relative_'
    return {
        'figure': root / f'{prefix}{eye_slug}_{band_slug}.png',
        'stats': root / ('stats_table.csv' if power_slug == 'absolute'
                         else 'stats_relative_vs_absolute.csv'),
        'summary': root / 'group_summary.csv',
        'epochs': root / 'epochs_and_exclusions_by_session_condition.csv',
    }


def load_group_view_data(analysis_dir, eye, band, figure_type, power_type):
    """Load one PNG, its five regional stats, and precomputed group metadata."""
    paths = group_view_files(analysis_dir, eye, band, figure_type, power_type)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        return {'missing': missing, 'paths': paths}
    stats_rows = [row for row in _read_csv(paths['stats'])
                  if row['eyes'] == eye and row['band'] == band]
    if power_type == 'Absoluta':
        table = [dict(region=row['region'], n_pairs=row['n_pairs'],
                      median=row['median_difference_log10_sd_minus_ns'],
                      effect=row['rank_biserial_effect'], q=row['fdr_q_value'])
                 for row in stats_rows]
    else:
        table = [dict(region=row['region'], n_pairs=row['n_pairs_relative'],
                      median=row['median_difference_log10_relative'],
                      effect=row['rank_biserial_relative'], q=row['q_relative'])
                 for row in stats_rows]
    summary_rows = _read_csv(paths['summary'])
    summary = {row['metric']: row['value'] for row in summary_rows}
    epoch_rows = [row for row in _read_csv(paths['epochs']) if row['eyes'] in EYE_VALUES]
    excluded = []
    for row in epoch_rows:
        excluded.append(
            f"{row['condition']} / {row['eyes'].lower()}: "
            f"{row['channels_excluded_total_inferred']} canal(es) ausentes inferidos"
        )
    return {'missing': [], 'paths': paths, 'table': table,
            'summary': summary, 'excluded': excluded}


class GroupUI:
    """Notebook page that displays files from analysis/ without recomputing them."""

    def __init__(self, notebook: ttk.Notebook, analysis_dir):
        self.analysis_dir = Path(analysis_dir)
        self.tab = ttk.Frame(notebook, padding=12)
        notebook.add(self.tab, text='Resultados de grupo')
        self.eye = tk.StringVar(value='Ojos abiertos')
        self.band = tk.StringVar(value='Delta')
        self.figure_type = tk.StringVar(value='Pares')
        self.power_type = tk.StringVar(value='Absoluta')
        self._build()
        self.refresh()

    def _build(self):
        ttk.Label(self.tab, text='Resultados de grupo', style='Title.TLabel').pack(anchor='w', pady=(0, 8))
        controls = ttk.Frame(self.tab)
        controls.pack(fill='x', pady=(0, 8))
        choices = (
            ('Condición ocular', self.eye, tuple(EYE_VALUES)),
            ('Banda', self.band, tuple(BAND_VALUES)),
            ('Figura', self.figure_type, tuple(FIGURE_VALUES)),
            ('Potencia', self.power_type, tuple(POWER_VALUES)),
        )
        for label, variable, values in choices:
            cell = ttk.Frame(controls)
            cell.pack(side='left', padx=(0, 8))
            ttk.Label(cell, text=label).pack(anchor='w')
            combo = ttk.Combobox(cell, textvariable=variable, values=values,
                                 state='readonly', width=16)
            combo.pack(anchor='w')
            combo.bind('<<ComboboxSelected>>', lambda _event: self.refresh())

        self.message = tk.StringVar(value='')
        ttk.Label(self.tab, textvariable=self.message, wraplength=900,
                  foreground='#9c3b33').pack(fill='x', pady=(0, 5))
        plot_frame = ttk.Frame(self.tab)
        plot_frame.pack(fill='both', expand=True)
        self.figure = Figure(figsize=(10, 3.8), dpi=100, facecolor='white')
        self.axis = self.figure.add_subplot(111)
        self.axis.axis('off')
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side='bottom', fill='x')
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        table_frame = ttk.Frame(self.tab)
        table_frame.pack(fill='x', pady=(8, 2))
        columns = ('region', 'n_pairs', 'median', 'effect', 'q')
        self.table = ttk.Treeview(table_frame, columns=columns, show='headings', height=5)
        labels = {'region': 'Región', 'n_pairs': 'Pares', 'median': 'Mediana SD−NS',
                  'effect': 'Efecto rank-biserial', 'q': 'q FDR'}
        widths = {'region': 115, 'n_pairs': 65, 'median': 130, 'effect': 150, 'q': 90}
        for column in columns:
            self.table.heading(column, text=labels[column])
            self.table.column(column, width=widths[column], anchor='center')
        self.table.pack(fill='x')
        self.summary_text = tk.StringVar(value='')
        ttk.Label(self.tab, textvariable=self.summary_text, wraplength=950,
                  justify='left').pack(fill='x', pady=(7, 0))

    def refresh(self):
        result = load_group_view_data(
            self.analysis_dir, self.eye.get(), self.band.get(),
            self.figure_type.get(), self.power_type.get(),
        )
        self.table.delete(*self.table.get_children())
        self.message.set('')
        if result['missing']:
            self.axis.clear()
            self.axis.text(0.5, 0.5, 'Archivos de resultados incompletos',
                           ha='center', va='center', transform=self.axis.transAxes)
            self.axis.axis('off')
            self.canvas.draw_idle()
            self.message.set('Faltan archivos en analysis/: ' + ', '.join(result['missing']))
            self.summary_text.set('Generá previamente las estadísticas, los gráficos y group_summary.csv.')
            return

        try:
            with Image.open(result['paths']['figure']) as image:
                rendered = image.convert('RGB').copy()
            self.axis.clear()
            self.axis.imshow(rendered)
            self.axis.axis('off')
            self.figure.tight_layout()
            self.canvas.draw_idle()
        except (OSError, ValueError) as exc:
            self.message.set(f"No se pudo abrir la figura {result['paths']['figure']}: {exc}")
        for row in result['table']:
            self.table.insert('', 'end', values=(
                row['region'], row['n_pairs'], _fmt(row['median']),
                _fmt(row['effect']), _fmt(row['q']),
            ))
        summary = result['summary']
        self.summary_text.set(
            f"Participantes ({summary.get('participant_count', '—')}): "
            f"{summary.get('participants', '—')}\n"
            f"Modo de rechazo: {summary.get('rejection_mode', '—')}\n"
            f"Canales ausentes inferidos por sesión: {'; '.join(result['excluded'])}. "
            f"El conteo de canales es inferido del CSV de epochs."
        )


def _fmt(value):
    try:
        return f'{float(value):.4g}'
    except (TypeError, ValueError):
        return '—'
