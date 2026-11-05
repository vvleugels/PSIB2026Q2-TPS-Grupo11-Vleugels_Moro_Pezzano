"""«Resultados del grupo» tab and the per-study batch card.

Shows what the batch pipeline, the statistics and the figures produced (results/), and runs
those steps in a child process (eeg_viewer.pipeline) so the explorer stays usable.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from .pipeline import STEP_LABELS, STEPS, Options, Runner, missing_dependencies
from .results_data import FILTER_COLUMNS, HEADINGS, TABLES, Results, filter_rows, fmt, number, sort_rows
from .theme import (AMBER, AMBER_LIGHT, BG, CARD, ERROR, IRIS, IRIS_LIGHT, MUTED, TEXT, Tooltip,
                    autowrap, callout, card)

CANVAS_BG = '#10242a'
EYES_ES = {'open': 'Abiertos', 'closed': 'Cerrados'}
EYES_OPTIONS = {'Ambos': None, 'Abiertos': 'open', 'Cerrados': 'closed'}
VALUE_ES = {
    'open': 'abiertos', 'closed': 'cerrados', 'primary': 'principal', 'secondary': 'secundario',
    'increase': 'aumento', 'decrease': 'disminución', 'no change': 'sin cambio',
    'none (exploratory)': 'sin predicción', 'supported': 'apoyada', 'opposite': 'opuesta',
    'not significant': 'no significativa', 'exploratory: significant': 'exploratoria: significativa',
    'exploratory: not significant': 'exploratoria: no significativa', 'not tested': 'sin prueba',
}
FIGURE_NAMES = {
    'topomap_open': 'Topomapas · ojos abiertos', 'topomap_closed': 'Topomapas · ojos cerrados',
    'psd_regions_open': 'PSD por región · ojos abiertos', 'psd_regions_closed': 'PSD por región · ojos cerrados',
    'paired_open': 'Potencia pareada · ojos abiertos', 'paired_closed': 'Potencia pareada · ojos cerrados',
    'theta_beta': 'Theta/beta frontal',
}
HYPOTHESIS_COLUMNS = ('hypothesis', 'eyes', 'band', 'region', 'expected', 'observed', 'n',
                      'mean_diff', 'p_corrected', 'cohens_dz', 'agreement', 'largest_region')
BAND_SYMBOLS = {'delta': 'δ', 'theta': 'θ', 'alpha': 'α', 'beta': 'β'}
ALL = 'Todos'


def display_value(column: str, text: str) -> str:
    """Cell text for the GUI: Spanish words for known values, rounded numbers."""
    if column in ('eyes', 'analysis', 'expected', 'observed', 'agreement') and text in VALUE_ES:
        return VALUE_ES[text]
    return fmt(column, text)


def open_path(path: Path) -> None:
    """Open a file or folder with the system's default application."""
    if sys.platform == 'win32':
        os.startfile(path)
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', str(path)])
    else:
        subprocess.Popen(['xdg-open', str(path)])


class ResultsUI:
    # ------------------------------------------------------------------- state
    def init_results(self, results_dir: Path, project: Path):
        self.results_dir = Path(results_dir)
        self.project_dir = Path(project)
        self.results = Results(self.results_dir, self.project_dir)
        self.runner = Runner(self.project_dir, self.results_dir)
        self.study_batch = None
        self.run_eyes = tk.StringVar(value='Ambos')
        self.run_jobs = tk.StringVar(value=str(Options().jobs))
        self.run_force = tk.BooleanVar(value=False)
        self.run_permutations = tk.StringVar(value='5000')
        self.run_progress_text = tk.StringVar(value='')
        self.run_step_text = tk.StringVar(value='')
        self.results_instruction = tk.StringVar()
        self.step_status = {step: tk.StringVar() for step in STEPS}
        self.step_detail = {step: tk.StringVar() for step in STEPS}
        self.hypothesis_eyes = tk.StringVar(value=ALL)
        self.table_choice = tk.StringVar()
        self.table_filters = {column: tk.StringVar(value=ALL) for column in FILTER_COLUMNS}
        self.table_count = tk.StringVar()
        self.table_sort = (None, False)
        self.figure_caption = tk.StringVar(value='')
        self.figure_title = tk.StringVar(value='Figuras')
        self.figure_paths = []
        self.figure_pil = None
        self.figure_photo = None
        self.figure_scale = 1.0
        self.figure_fit = True
        self.figure_size = None
        self.batch_state = tk.StringVar(value='Elegí un estudio.')
        self.batch_epochs = tk.StringVar(value='')
        self.batch_bad = tk.StringVar(value='')
        self.batch_channel = tk.StringVar(value='')
        self.batch_warning = tk.StringVar(value='')

    # ------------------------------------------------------------------ layout
    def build_results(self, tab):
        """Inner notebook with Proceso / Hipótesis / Tablas / Figuras."""
        self.results_tabs = ttk.Notebook(tab)
        self.results_tabs.pack(fill='both', expand=True)
        pages = {}
        for key, text in (('process', 'Proceso'), ('hypotheses', 'Hipótesis'),
                          ('tables', 'Tablas'), ('figures', 'Figuras')):
            pages[key] = ttk.Frame(self.results_tabs, style='Card.TFrame', padding=10)
            self.results_tabs.add(pages[key], text=text)
        self.build_process_page(pages['process'])
        self.build_hypothesis_page(pages['hypotheses'])
        self.build_table_page(pages['tables'])
        self.build_figure_page(pages['figures'])
        self.refresh_results()

    def build_process_page(self, page):
        cards = ttk.Frame(page, style='Card.TFrame')
        cards.pack(fill='x')
        self.step_buttons, self.step_status_labels = {}, {}
        for column, step in enumerate(STEPS):
            cards.columnconfigure(column, weight=1, uniform='steps')
            outer, inner = card(cards, f'{column + 1}  {STEP_LABELS[step]}')
            outer.grid(row=0, column=column, sticky='nsew', padx=(0 if column == 0 else 8, 0))
            status = ttk.Label(inner, textvariable=self.step_status[step], style='CardBold.TLabel')
            status.pack(anchor='w', pady=(4, 0))
            self.step_status_labels[step] = status
            detail = ttk.Label(inner, textvariable=self.step_detail[step], style='CardMuted.TLabel')
            detail.pack(fill='x')
            autowrap(detail)
            options = ttk.Frame(inner, style='Card.TFrame')
            options.pack(fill='x', pady=8)
            if step == 'batch':
                self.option_row(options, 'Ojos', ttk.Combobox(options, textvariable=self.run_eyes, width=10,
                                                              values=list(EYES_OPTIONS), state='readonly'))
                self.jobs_box = ttk.Spinbox(options, from_=1, to=max(1, os.cpu_count() or 1),
                                            textvariable=self.run_jobs, width=6)
                self.option_row(options, 'Procesos en paralelo', self.jobs_box)
                self.force_toggle = ttk.Checkbutton(options, text='Recalcular todo', variable=self.run_force,
                                                    style='Card.TCheckbutton')
                self.force_toggle.pack(anchor='w', pady=(4, 0))
            elif step == 'stats':
                self.permutations_box = ttk.Spinbox(options, from_=100, to=100000, increment=1000,
                                                    textvariable=self.run_permutations, width=8)
                self.option_row(options, 'Permutaciones (clusters)', self.permutations_box)
            else:
                ttk.Label(options, text='Topomapas, PSD, gráficos pareados, theta/beta y tabla de hipótesis.',
                          style='CardMuted.TLabel', wraplength=260).pack(anchor='w')
            button = ttk.Button(inner, text='Ejecutar', command=lambda s=step: self.run_pipeline([s]))
            button.pack(fill='x')
            self.step_buttons[step] = button

        controls = ttk.Frame(page, style='Card.TFrame')
        controls.pack(fill='x', pady=(10, 6))
        self.run_all_button = ttk.Button(controls, text='Ejecutar todo (1 → 3)', style='Accent.TButton',
                                         command=lambda: self.run_pipeline(list(STEPS)))
        self.run_all_button.pack(side='left')
        self.cancel_run_button = ttk.Button(controls, text='Cancelar', command=self.cancel_pipeline)
        self.cancel_run_button.pack(side='left', padx=6)
        ttk.Button(controls, text='Abrir carpeta de resultados',
                   command=lambda: self.open_results_path(self.results_dir)).pack(side='right')
        self.log_button = ttk.Button(controls, text='Mostrar registro ▾', command=self.toggle_run_log)
        self.log_button.pack(side='right', padx=6)
        progress_row = ttk.Frame(page, style='Card.TFrame')
        progress_row.pack(fill='x')
        ttk.Label(progress_row, textvariable=self.run_step_text, style='CardBold.TLabel').pack(side='left')
        ttk.Label(progress_row, textvariable=self.run_progress_text, style='CardMuted.TLabel',
                  width=6, anchor='e').pack(side='right')
        self.run_progress = ttk.Progressbar(progress_row, maximum=100)
        self.run_progress.pack(side='left', fill='x', expand=True, padx=10)
        self.run_log = tk.Text(page, height=14, wrap='none', state='disabled', relief='flat',
                               background=BG, foreground=TEXT, padx=10, pady=8,
                               font='TkFixedFont')   # shown on demand or on error
        Tooltip(self.run_all_button, 'Ejecuta el lote, la estadística y las figuras en orden. '
                                     'El explorador sigue disponible mientras tanto.')
        Tooltip(self.cancel_run_button, 'Detiene el paso en curso. El lote es reanudable: '
                                        'lo ya calculado se conserva.')
        Tooltip(self.step_buttons['batch'], 'python -m batch.run_batch: filtra, divide en epochs, '
                                            'aplica el barrido y calcula la potencia de cada registro.')
        Tooltip(self.step_buttons['stats'], 'python -m analysis.stats: regiones, pruebas pareadas SD vs NS, '
                                            'FDR, orden de sesión y clusters.')
        Tooltip(self.step_buttons['figures'], 'python -m analysis.figures: figuras, leyendas y tabla de hipótesis.')
        Tooltip(self.force_toggle, 'Desactivado: solo procesa los registros nuevos o afectados por '
                                   'un cambio de config.py.')

    @staticmethod
    def option_row(parent, label, widget):
        row = ttk.Frame(parent, style='Card.TFrame')
        row.pack(fill='x', pady=2)
        ttk.Label(row, text=label, style='Card.TLabel').pack(side='left')
        widget.pack(in_=row, side='right')
        widget.lift(row)   # created before the row frame: raise it above

    def build_hypothesis_page(self, page):
        bar = ttk.Frame(page, style='Card.TFrame')
        bar.pack(fill='x', pady=(0, 8))
        ttk.Label(bar, text='Ojos', style='Card.TLabel').pack(side='left')
        box = ttk.Combobox(bar, textvariable=self.hypothesis_eyes, values=[ALL, 'Abiertos', 'Cerrados'],
                           state='readonly', width=10)
        box.pack(side='left', padx=6)
        box.bind('<<ComboboxSelected>>', lambda e: self.fill_hypotheses())
        ttk.Button(bar, text='Abrir hypotheses.md',
                   command=lambda: self.open_results_path(self.project_dir / 'analysis' / 'hypotheses.md')
                   ).pack(side='right')
        note = tk.Frame(page, background=IRIS_LIGHT)
        note.pack(fill='x', pady=(0, 8))
        tk.Frame(note, background=IRIS, width=4).pack(side='left', fill='y')
        self.hypothesis_text = tk.StringVar()
        text = ttk.Label(note, textvariable=self.hypothesis_text, style='Info.TLabel', padding=(12, 8),
                         justify='left')
        text.pack(side='left', fill='x', expand=True)
        autowrap(text)
        self.hypothesis_tree = self.make_tree(page, HYPOTHESIS_COLUMNS)
        self.hypothesis_tree.tag_configure('apoyada', background='#d5eee2')
        self.hypothesis_tree.tag_configure('opuesta', background='#f9d8d5')
        self.hypothesis_tree.tag_configure('nosig', foreground=MUTED)
        legend = ttk.Label(page, style='CardMuted.TLabel', text=(
            'Verde: dirección esperada y p corregido < 0,05 · Rojo: significativa en la dirección '
            'opuesta · Gris: no significativa. Dif. media en log10 (theta/beta: log10 del cociente). '
            'Diferencias espectrales a nivel de grupo, sin interpretación clínica.'))
        legend.pack(fill='x', pady=(6, 0))
        autowrap(legend)

    def build_table_page(self, page):
        bar = ttk.Frame(page, style='Card.TFrame')
        bar.pack(fill='x', pady=(0, 6))
        ttk.Label(bar, text='Tabla', style='Card.TLabel').pack(side='left')
        self.table_box = ttk.Combobox(bar, textvariable=self.table_choice, state='readonly', width=42)
        self.table_box.pack(side='left', padx=6)
        self.table_box.bind('<<ComboboxSelected>>', lambda e: self.show_table(reset_filters=True))
        ttk.Button(bar, text='Abrir CSV', command=self.open_table_csv).pack(side='right')
        ttk.Label(bar, textvariable=self.table_count, style='CardMuted.TLabel').pack(side='right', padx=10)
        self.filter_bar = ttk.Frame(page, style='Card.TFrame')
        self.filter_bar.pack(fill='x', pady=(0, 6))
        self.filter_widgets = {}
        names = {'eyes': 'Ojos', 'ses': 'Sesión', 'band': 'Banda', 'region': 'Región', 'sub': 'Participante'}
        for column in FILTER_COLUMNS:
            frame = ttk.Frame(self.filter_bar, style='Card.TFrame')
            ttk.Label(frame, text=names.get(column, column), style='Card.TLabel').pack(side='left')
            box = ttk.Combobox(frame, textvariable=self.table_filters[column], state='readonly', width=11)
            box.pack(side='left', padx=(4, 12))
            box.bind('<<ComboboxSelected>>', lambda e: self.show_table())
            self.filter_widgets[column] = (frame, box)
        self.clear_filters_button = ttk.Button(self.filter_bar, text='Quitar filtros', style='Tool.TButton',
                                               command=lambda: self.show_table(reset_filters=True))
        frame = ttk.Frame(page, style='Card.TFrame')
        frame.pack(fill='both', expand=True)
        self.table_tree = ttk.Treeview(frame, show='headings', selectmode='browse')
        vertical = ttk.Scrollbar(frame, orient='vertical', command=self.table_tree.yview)
        horizontal = ttk.Scrollbar(frame, orient='horizontal', command=self.table_tree.xview)
        self.table_tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        vertical.pack(side='right', fill='y')
        horizontal.pack(side='bottom', fill='x')
        self.table_tree.pack(fill='both', expand=True)
        self.table_tree.tag_configure('significativa', background='#fff4dc')

    def build_figure_page(self, page):
        left = ttk.Frame(page, style='Card.TFrame')
        left.pack(side='left', fill='y', padx=(0, 10))
        ttk.Label(left, text='FIGURAS', style='CardTitle.TLabel').pack(anchor='w')
        self.figure_listbox = tk.Listbox(left, width=30, height=12, exportselection=False, relief='flat',
                                         highlightthickness=1, highlightbackground='#d9e1e2',
                                         highlightcolor=IRIS, selectbackground=IRIS,
                                         selectforeground='white', activestyle='none')
        self.figure_listbox.pack(fill='y', expand=True, pady=4)
        self.figure_listbox.bind('<<ListboxSelect>>', lambda e: self.show_figure())
        ttk.Button(left, text='Abrir carpeta de figuras',
                   command=lambda: self.open_results_path(self.results.figures_dir)).pack(fill='x')
        right = ttk.Frame(page, style='Card.TFrame')
        right.pack(side='left', fill='both', expand=True)
        note = tk.Frame(right, background=IRIS_LIGHT)
        note.pack(fill='x', pady=(0, 6))
        tk.Frame(note, background=IRIS, width=4).pack(side='left', fill='y')
        text = ttk.Frame(note, style='Info.TFrame', padding=(12, 8))
        text.pack(side='left', fill='x', expand=True)
        ttk.Label(text, textvariable=self.figure_title, style='InfoTitle.TLabel').pack(anchor='w')
        caption = ttk.Label(text, textvariable=self.figure_caption, style='Info.TLabel', justify='left')
        caption.pack(fill='x', anchor='w')
        autowrap(caption)
        tools = ttk.Frame(right, style='Card.TFrame')
        tools.pack(fill='x', pady=(0, 4))
        for label, factor, tip in (('+', 1.25, 'Acercar.'), ('−', .8, 'Alejar.'),
                                   ('Ajustar', None, 'Ajustar la figura a la ventana.')):
            button = ttk.Button(tools, text=label, style='Tool.TButton', width=7 if factor is None else 4,
                                command=self.fit_figure if factor is None else
                                (lambda f=factor: self.zoom_figure(f)))
            button.pack(side='right', padx=(4, 0))
            Tooltip(button, tip)
        ttk.Label(tools, text='Arrastrar para mover · Ctrl + rueda para hacer zoom',
                  style='CardMuted.TLabel').pack(side='left')
        viewer = ttk.Frame(right, style='Card.TFrame')
        viewer.pack(fill='both', expand=True)
        self.figure_canvas = tk.Canvas(viewer, background=CANVAS_BG, highlightthickness=0, cursor='fleur')
        vertical = ttk.Scrollbar(viewer, orient='vertical', command=self.figure_canvas.yview)
        horizontal = ttk.Scrollbar(viewer, orient='horizontal', command=self.figure_canvas.xview)
        self.figure_canvas.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        vertical.pack(side='right', fill='y')
        horizontal.pack(side='bottom', fill='x')
        self.figure_canvas.pack(fill='both', expand=True)
        self.figure_canvas.bind('<ButtonPress-1>', lambda e: self.figure_canvas.scan_mark(e.x, e.y))
        self.figure_canvas.bind('<B1-Motion>', lambda e: self.figure_canvas.scan_dragto(e.x, e.y, gain=1))
        self.figure_canvas.bind('<Configure>', lambda e: self.draw_figure())

    def build_batch_card(self, parent):
        """Sidebar card with the batch outcome of the open study."""
        inner = self.card(parent, 'Resultado del lote')
        for variable, style in ((self.batch_state, 'CardBold.TLabel'), (self.batch_epochs, 'Card.TLabel'),
                                (self.batch_bad, 'CardMuted.TLabel')):
            label = ttk.Label(inner, textvariable=variable, style=style, justify='left')
            label.pack(fill='x', anchor='w', pady=(2, 0))
            autowrap(label)
        self.batch_warning_label = tk.Label(inner, textvariable=self.batch_warning, background=AMBER_LIGHT,
                                            foreground=TEXT, anchor='w', justify='left', padx=8, pady=4)
        self.batch_warning_label.bind('<Configure>', lambda e: self.batch_warning_label.configure(
            wraplength=max(e.width - 16, 100)))
        self.batch_channel_label = ttk.Label(inner, textvariable=self.batch_channel, style='Card.TLabel',
                                             justify='left', font='TkFixedFont')
        self.batch_channel_label.pack(fill='x', anchor='w', pady=(6, 0))
        row = ttk.Frame(inner, style='Card.TFrame')
        row.pack(fill='x', pady=(8, 0))
        self.batch_exclusions_button = ttk.Button(row, text='Ver exclusiones', style='Tool.TButton',
                                                  command=self.show_study_exclusions)
        self.batch_exclusions_button.pack(side='left')
        ttk.Button(row, text='Ver grupo ▸', style='Tool.TButton',
                   command=lambda: self.view_tabs.select(2)).pack(side='right')
        Tooltip(self.batch_exclusions_button, 'Abre la tabla de exclusiones del lote filtrada a este estudio: '
                                              'registro, canales malos y epochs rechazadas con su motivo.')

    # --------------------------------------------------------------- tree helper
    def make_tree(self, parent, columns):
        frame = ttk.Frame(parent, style='Card.TFrame')
        frame.pack(fill='both', expand=True)
        tree = ttk.Treeview(frame, columns=columns, show='headings', selectmode='browse')
        vertical = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=vertical.set)
        vertical.pack(side='right', fill='y')
        tree.pack(fill='both', expand=True)
        for column in columns:
            tree.heading(column, text=HEADINGS.get(column, column))
            tree.column(column, width=90 if column not in ('agreement', 'largest_region') else 150,
                        minwidth=50, stretch=True)
        return tree

    # ----------------------------------------------------------------- refresh
    def refresh_results(self):
        """Re-read results/ and update every view (after a pipeline step or «Actualizar»)."""
        self.results = Results(self.results_dir, self.project_dir)
        self.update_step_status()
        self.fill_hypotheses()
        tables = self.results.available_tables()
        self.table_box.configure(values=[TABLES[t] for t in tables])
        if tables and self.table_choice.get() not in self.table_box.cget('values'):
            self.table_choice.set(TABLES[tables[0]])
        self.show_table()
        self.fill_figure_list()
        self.update_run_controls()

    def update_step_status(self):
        status = self.results.status()
        for step in STEPS:
            state = status[step]
            if not state.done:
                text, color = '✖  Sin resultados', ERROR
            elif state.stale:
                text, color = '⚠  Desactualizado', '#9b5600'
            else:
                text, color = '✔  Listo', IRIS
            self.step_status[step].set(text)
            self.step_status_labels[step].configure(foreground=color)
            when = state.updated.strftime('%d/%m/%Y %H:%M') if state.updated else ''
            missing = missing_dependencies(step)
            extra = (f'\nFalta {", ".join(missing)}: pip install -r requirements-analysis.txt'
                     if missing else '')
            self.step_detail[step].set(f'{state.detail}' + (f' · {when}' if when else '') + extra)
        self.results_instruction.set(self.results_instruction_text(status))
        if hasattr(self, 'stepper'):
            self.refresh_guide()   # the main amber callout shows this text on the results tab

    def results_instruction_text(self, status=None) -> str:
        status = status or self.results.status()
        if self.runner.running:
            return 'Ejecutando… podés seguir usando el explorador. Los resultados se actualizan al terminar.'
        for index, step in enumerate(STEPS, 1):
            if not status[step].done:
                return f'Falta el paso {index} ({STEP_LABELS[step]}). Presioná «Ejecutar todo» o «Ejecutar» en ese paso.'
        for index, step in enumerate(STEPS, 1):
            if status[step].stale:
                return (f'Resultados desactualizados desde el paso {index} ({STEP_LABELS[step]}): '
                        'volvé a ejecutarlo para que todo use la misma configuración.')
        return ('Resultados completos. Revisá «Hipótesis», «Tablas» y «Figuras». Ojos abiertos es el '
                'análisis principal; ojos cerrados, el secundario.')

    # ------------------------------------------------------------- hypotheses
    def fill_hypotheses(self):
        tree = self.hypothesis_tree
        tree.delete(*tree.get_children())
        rows = self.results.hypothesis_rows()
        eyes = {'Abiertos': 'open', 'Cerrados': 'closed'}.get(self.hypothesis_eyes.get())
        for row in filter_rows(rows, eyes=eyes):
            agreement = row.get('agreement', '')
            tag = ('apoyada' if agreement == 'supported' else 'opuesta' if agreement == 'opposite'
                   else 'nosig' if agreement == 'not significant' else '')
            tree.insert('', 'end', values=[display_value(c, row.get(c, '')) for c in HYPOTHESIS_COLUMNS],
                        tags=(tag,) if tag else ())
        self.hypothesis_text.set(self.hypothesis_statements() if rows else
                                 'Todavía no hay tabla de hipótesis: ejecutá los pasos en «Proceso».')

    def hypothesis_statements(self) -> str:
        try:
            sys.path.insert(0, str(self.project_dir))
            from analysis.hypotheses import HYPOTHESES   # needs pandas; text falls back below
        except Exception:
            return 'H1–H4: ver analysis/hypotheses.md (referencias pendientes, TODO).'
        lines = [f'{h.id}: {h.statement}' for h in HYPOTHESES]
        return '\n'.join(lines) + '\nReferencias: pendientes de verificación (TODO en analysis/hypotheses.md).'

    # ----------------------------------------------------------------- tables
    def current_table(self) -> str | None:
        label = self.table_choice.get()
        return next((name for name, text in TABLES.items() if text == label), None)

    def show_table(self, reset_filters: bool = False, sort: str | None = None):
        name = self.current_table()
        tree = self.table_tree
        tree.delete(*tree.get_children())
        if name is None:
            tree.configure(columns=())
            self.table_count.set('Sin tablas: ejecutá los pasos en «Proceso».')
            return
        rows = self.results.table(name)
        columns = list(rows[0]) if rows else []
        if reset_filters:
            for variable in self.table_filters.values():
                variable.set(ALL)
            self.table_sort = (None, False)
        # Filters only for columns this table has; values in the table's own order.
        for column, (frame, box) in self.filter_widgets.items():
            if column in columns:
                values = sorted({r[column] for r in rows if r.get(column)},
                                key=lambda v: (len(v), v) if column == 'sub' else v)
                labels = [ALL] + [EYES_ES.get(v, v) if column == 'eyes' else v for v in values]
                box.configure(values=labels)
                frame.pack(side='left')
            else:
                self.table_filters[column].set(ALL)
                frame.pack_forget()
        self.clear_filters_button.pack(side='right')
        active = {}
        for column, variable in self.table_filters.items():
            value = variable.get()
            if value and value != ALL:
                active[column] = {'Abiertos': 'open', 'Cerrados': 'closed'}.get(value, value) \
                    if column == 'eyes' else value
        shown = filter_rows(rows, **active)
        if sort is not None:
            current, descending = self.table_sort
            self.table_sort = (sort, not descending if current == sort else False)
        if self.table_sort[0]:
            shown = sort_rows(shown, *self.table_sort)
        tree.configure(columns=columns)
        for column in columns:
            arrow = (' ▼' if self.table_sort[1] else ' ▲') if column == self.table_sort[0] else ''
            tree.heading(column, text=HEADINGS.get(column, column) + arrow,
                         command=lambda c=column: self.show_table(sort=c))
            width = 260 if column in ('reason', 'channels', 'note', 'file') else 95
            tree.column(column, width=width, minwidth=50, stretch=column in ('reason', 'channels'))
        significant = next((c for c in ('p_fdr_wilcoxon', 'p_cluster', 'p_fdr_mannwhitney') if c in columns), None)
        for row in shown:
            p = number(row.get(significant, '')) if significant else None
            tag = ('significativa',) if p is not None and p < 0.05 else ()
            tree.insert('', 'end', values=[display_value(c, row.get(c, '')) for c in columns], tags=tag)
        self.table_count.set(f'{len(shown)} de {len(rows)} filas' +
                             (' · resaltadas: p < 0,05' if significant else ''))

    def open_table_csv(self):
        name = self.current_table()
        if name:
            self.open_results_path(self.results_dir / f'{name}.csv')

    def show_study_exclusions(self):
        """Exclusions table filtered to the open study."""
        if self.recording is None:
            return
        study = self.recording.study
        self.view_tabs.select(2)
        self.results_tabs.select(2)
        if 'exclusions' not in self.results.available_tables():
            self.show_table()
            return
        self.table_choice.set(TABLES['exclusions'])
        self.show_table(reset_filters=True)
        eyes = {'eyesopen': 'Abiertos', 'eyesclosed': 'Cerrados'}.get(study.task, ALL)
        self.table_filters['sub'].set(study.participant)
        self.table_filters['ses'].set(study.session)
        self.table_filters['eyes'].set(eyes)
        self.show_table()

    # ---------------------------------------------------------------- figures
    def fill_figure_list(self):
        self.figure_paths = self.results.figure_list()
        self.figure_captions = self.results.captions()
        self.figure_listbox.delete(0, 'end')
        for path in self.figure_paths:
            self.figure_listbox.insert('end', FIGURE_NAMES.get(path.stem, path.stem))
        if self.figure_paths:
            self.figure_listbox.selection_set(0)
            self.show_figure()
        else:
            self.figure_pil = None
            self.figure_title.set('Figuras')
            self.figure_caption.set('Todavía no hay figuras: ejecutá el paso 3 en «Proceso».')
            self.draw_figure()

    def show_figure(self):
        selection = self.figure_listbox.curselection()
        if not selection:
            return
        path = self.figure_paths[selection[0]]
        try:
            with Image.open(path) as image:
                self.figure_pil = image.convert('RGB')
        except OSError as exc:
            self.figure_pil = None
            self.figure_caption.set(f'No se pudo abrir {path.name}: {exc}')
            self.draw_figure()
            return
        self.figure_title.set(FIGURE_NAMES.get(path.stem, path.stem))
        self.figure_caption.set(self.figure_captions.get(path.stem, 'Sin leyenda en captions.md.'))
        self.figure_fit = True
        self.figure_size = None
        self.draw_figure()
        self.figure_canvas.xview_moveto(0)
        self.figure_canvas.yview_moveto(0)

    def draw_figure(self):
        canvas = self.figure_canvas
        canvas.delete('all')
        cw, ch = max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)
        if self.figure_pil is None:
            canvas.configure(scrollregion=(0, 0, cw, ch))
            canvas.create_text(cw / 2, ch / 2, text='Sin figura', fill='#e5e7eb')
            return
        w, h = self.figure_pil.size
        if self.figure_fit:
            self.figure_scale = min(max(cw - 4, 50) / w, max(ch - 4, 50) / h, 1.0)
        dw, dh = max(1, round(w * self.figure_scale)), max(1, round(h * self.figure_scale))
        if (dw, dh) != self.figure_size:   # resample only when the displayed size changes
            method = Image.Resampling.LANCZOS if self.figure_scale < 1 else Image.Resampling.BILINEAR
            self.figure_photo = ImageTk.PhotoImage(self.figure_pil.resize((dw, dh), method), master=canvas)
            self.figure_size = (dw, dh)
        sw, sh = max(dw, cw), max(dh, ch)
        canvas.configure(scrollregion=(0, 0, sw, sh))
        canvas.create_image(sw / 2, sh / 2, image=self.figure_photo, anchor='center')

    def fit_figure(self):
        self.figure_fit = True
        self.draw_figure()

    def zoom_figure(self, factor):
        if self.figure_pil is None:
            return
        self.figure_fit = False
        self.figure_scale = max(.05, min(self.figure_scale * factor, 4.0))
        self.draw_figure()

    def figure_wheel(self, event, widget) -> bool:
        """Ctrl + wheel zooms the figure; plain wheel scrolls it. True if the event was handled."""
        while widget is not None and widget is not self.figure_canvas:
            widget = widget.master
        if widget is None:
            return False
        up = event.num == 4 or (event.num != 5 and event.delta > 0)
        if event.state & 0x4:
            self.zoom_figure(1.25 if up else .8)
        else:
            self.figure_canvas.yview_scroll(-3 if up else 3, 'units')
        return True

    # --------------------------------------------------------------- pipeline
    def run_options(self) -> Options:
        try:
            jobs = max(1, int(self.run_jobs.get()))
            permutations = max(100, int(self.run_permutations.get()))
        except ValueError:
            raise ValueError('Procesos en paralelo y permutaciones deben ser números enteros.')
        return Options(eyes=EYES_OPTIONS.get(self.run_eyes.get()), jobs=jobs, force=self.run_force.get(),
                       permutations=permutations)

    def run_pipeline(self, steps):
        if self.runner.running:
            return
        missing = sorted({name for step in steps for name in missing_dependencies(step)})
        if missing:
            messagebox.showerror('Faltan dependencias',
                                 f'Instalá {", ".join(missing)} con\n'
                                 'pip install -r requirements-analysis.txt\n'
                                 '(TPS.bat lo hace automáticamente).', parent=self.root)
            return
        try:
            options = self.run_options()
        except ValueError as exc:
            messagebox.showerror('Opciones inválidas', str(exc), parent=self.root)
            return
        self.run_log.configure(state='normal')
        self.run_log.delete('1.0', 'end')
        self.run_log.configure(state='disabled')
        self.run_progress.configure(mode='determinate', value=0)
        self.run_progress_text.set('')
        self.runner.start(steps, options)
        self.update_run_controls()
        self.refresh_guide()

    def cancel_pipeline(self):
        if self.runner.running:
            self.run_step_text.set('Cancelando…')
            self.runner.cancel()

    def toggle_run_log(self, show: bool | None = None):
        visible = bool(self.run_log.winfo_manager())
        if show is None:
            show = not visible
        if show and not visible:
            self.run_log.pack(fill='both', expand=True, pady=(8, 0))
            self.log_button.configure(text='Ocultar registro ▴')
        elif not show and visible:
            self.run_log.pack_forget()
            self.log_button.configure(text='Mostrar registro ▾')

    def append_run_log(self, text):
        self.run_log.configure(state='normal')
        self.run_log.insert('end', text + '\n')
        if int(self.run_log.index('end-1c').split('.')[0]) > 4000:   # keep the widget light
            self.run_log.delete('1.0', '500.0')
        self.run_log.see('end')
        self.run_log.configure(state='disabled')

    def update_run_controls(self):
        running = self.runner.running
        for button in [*self.step_buttons.values(), self.run_all_button]:
            button.configure(state='disabled' if running else 'normal')
        self.cancel_run_button.configure(state='normal' if running else 'disabled')

    def poll_pipeline(self):
        """Consume runner events on the Tk thread (called from App.poll)."""
        while True:
            try:
                event = self.runner.events.get_nowait()
            except Exception:
                break
            kind = event[0]
            if kind == 'start':
                step = event[1]
                self.run_step_text.set(f'{STEP_LABELS[step]}…')
                indeterminate = step != 'batch'
                self.run_progress.configure(mode='indeterminate' if indeterminate else 'determinate', value=0)
                if indeterminate:
                    self.run_progress.start(12)
                self.run_progress_text.set('')
                self.append_run_log(f'── {STEP_LABELS[step]} ──')
            elif kind == 'line':
                self.append_run_log(event[2])
            elif kind == 'progress':
                self.run_progress.configure(value=event[2])
                self.run_progress_text.set(f'{event[2]:.0f}%')
            elif kind == 'done':
                self.run_progress.stop()
                step, code = event[1], event[2]
                if code == 0:
                    self.run_progress.configure(mode='determinate', value=100)
                    self.run_progress_text.set('100%')
                    self.reload_after_step()
                else:
                    self.append_run_log(f'{STEP_LABELS[step]} terminó con código {code}.')
            elif kind == 'finished':
                ok, cancelled = event[1], event[2]
                self.run_progress.stop()
                self.run_step_text.set('Cancelado.' if cancelled else 'Listo.' if ok else 'Error: revisá el registro.')
                self.status.set('Resultados del grupo: ' + ('proceso cancelado; lo ya calculado se conserva.'
                                                           if cancelled else 'actualizados.' if ok else
                                                           'el proceso falló; revisá el registro.'))
                if not ok and not cancelled:
                    self.toggle_run_log(True)
                self.reload_after_step()
                self.update_run_controls()

    def reload_after_step(self):
        """Refresh the views and the open study's card with the new outputs."""
        self.refresh_results()
        if self.recording is not None:
            study = self.recording.study
            self.study_batch = self.results.study_summary(study.participant, study.session, study.task)
            self.update_batch_card()
            if self.store:
                self.refresh_epochs()

    def open_results_path(self, path):
        path = Path(path)
        if not path.exists():
            self.status.set(f'Todavía no existe: {path}')
            return
        try:
            open_path(path)
        except OSError as exc:
            self.status.set(f'No se pudo abrir {path}: {exc}')

    # ---------------------------------------------------------- per-study card
    def clear_batch_card(self, message='Elegí un estudio.'):
        self.study_batch = None
        self.batch_state.set(message)
        for variable in (self.batch_epochs, self.batch_bad, self.batch_channel, self.batch_warning):
            variable.set('')
        self.batch_warning_label.pack_forget()
        self.batch_exclusions_button.configure(state='disabled')

    def update_batch_card(self):
        summary = self.study_batch
        if summary is None:
            self.clear_batch_card()
            return
        if not summary.found:
            self.clear_batch_card('Sin datos del lote para este estudio. Ejecutalo en «Resultados del grupo».')
            return
        self.batch_exclusions_button.configure(state='normal')
        if summary.excluded:
            self.batch_state.set('✖ Excluido del análisis')
            self.batch_epochs.set(summary.reason)
        else:
            self.batch_state.set('✔ Incluido en el análisis')
            self.batch_epochs.set(f'Epochs usadas: {summary.n_used} · rechazadas: {summary.n_rejected}'
                                  ' (barrido automático: dentro de límites, no inspección visual)')
        bad = sorted(summary.bad_channels)
        self.batch_bad.set(f'Canales malos ({len(bad)}): ' + ', '.join(bad) if bad else
                           ('Sin canales malos.' if not summary.excluded else ''))
        channels = self.visible_channels() if hasattr(self, 'channel_list') else []
        channel = channels[0] if len(channels) == 1 else self.channel.get()
        self.batch_warning.set('')
        self.batch_warning_label.pack_forget()
        if channel and channel in summary.bad_channels:
            self.batch_warning.set(f'{channel} fue un canal malo en el lote: {summary.bad_channels[channel]}. '
                                   'No tiene potencias por banda.')
            self.batch_warning_label.pack(fill='x', pady=(6, 0), before=self.batch_channel_label)
            self.batch_channel.set('')
        elif channel and channel in summary.powers:
            lines = [f'{channel}   abs (µV²)   rel (%)']
            for band, symbol in BAND_SYMBOLS.items():
                absolute, relative = summary.powers[channel].get(band, (None, None))
                lines.append(f'{symbol} {band:<6} {absolute:>9.3g}   {relative:>6.1f}'
                             if absolute is not None and relative is not None else f'{symbol} {band:<6}   —')
            ratio = summary.theta_beta.get(channel)
            if ratio is not None:
                lines.append(f'θ/β      {ratio:>9.3g}')
            self.batch_channel.set('\n'.join(lines))
        else:
            self.batch_channel.set('Elegí un electrodo para ver sus potencias del lote.' if not summary.excluded else '')

    def batch_rejection(self, index):
        """Batch reason for rejecting epoch `index` (0-based) of the open study, or ''."""
        if self.study_batch is None or not self.study_batch.found:
            return ''
        return self.study_batch.rejected_epochs.get(index, '')
