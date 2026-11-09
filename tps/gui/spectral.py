"""Spectral tab and configuration, keeping computation out of the UI thread."""
import tkinter as tk
from tkinter import ttk, filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from tps.gui.theme import BG, CARD, IRIS, IRIS_LIGHT, TEXT, style_dialog
from tps.core.processing import FilterConfig
from tps.core.spectral import SpectralConfig, select_studies, run_analysis, plot_result, export_result


# Tabla de bandas: (id, encabezado, ancho). Al comparar NS/SD, cada banda ocupa una fila
# y la potencia y la relativa de ambas sesiones quedan lado a lado.
SINGLE_COLUMNS = (('session', 'Sesión', 150), ('band', 'Banda', 80), ('range', 'Hz', 90),
                  ('absolute', 'Potencia (µV²)', 130), ('relative', 'Relativa (%)', 110))
COMPARE_COLUMNS = (('band', 'Banda', 80), ('range', 'Hz', 80),
                   ('abs_ns', 'Potencia NS (µV²)', 130), ('abs_sd', 'Potencia SD (µV²)', 130),
                   ('rel_ns', 'Relativa NS (%)', 110), ('rel_sd', 'Relativa SD (%)', 110))
SESSION_CODES = {'ses-1': 'ns', 'ses-2': 'sd'}


def format_relative(value):
    return f'{value:.4g}' if value is not None else 'No disponible'


class SpectralUI:
    def init_spectral(self):
        self.spectral_config = SpectralConfig()
        self.spectral_result = None
        self.spectral_revision = 0
        self.spectral_pending = False
        self.spectral_mode = tk.StringVar(value='Estudio actual')
        self.include_unreviewed = tk.BooleanVar(value=False)
        self.spectral_db = tk.BooleanVar(value=True)
        self.export_png = tk.BooleanVar(value=False)
        self.spectral_message = tk.StringVar(value='Seleccioná un estudio y un electrodo.')

    def build_spectral(self, parent, toolbar_class):
        # Keep a usable plot height on small windows; the whole tab can scroll.
        self.spectral_view = tk.Canvas(parent, highlightthickness=0, background=CARD)
        tab_scroll = ttk.Scrollbar(parent, command=self.spectral_view.yview)
        tab_scroll.pack(side='right', fill='y')
        self.spectral_view.configure(yscrollcommand=tab_scroll.set)
        self.spectral_view.pack(fill='both', expand=True)
        body = ttk.Frame(self.spectral_view, style='Card.TFrame')
        body_window = self.spectral_view.create_window(0, 0, window=body, anchor='nw')
        def resize(event):
            height = max(780, event.height)
            self.spectral_view.itemconfigure(body_window, width=event.width, height=height)
            self.spectral_view.configure(scrollregion=(0, 0, event.width, height))
        self.spectral_view.bind('<Configure>', resize)
        parent = body
        controls = ttk.Frame(parent, style='Card.TFrame', padding=(0, 0, 0, 0))
        controls.pack(fill='x')
        mode = ttk.Combobox(controls, textvariable=self.spectral_mode,
                            values=('Estudio actual', 'Comparar NS/SD'), state='readonly', width=22)
        mode.pack(side='left')
        mode.bind('<<ComboboxSelected>>', lambda e: self.invalidate_spectrum())
        self.calculate_button = ttk.Button(controls, text='Calcular', style='Accent.TButton',
                                           command=self.calculate_spectrum)
        self.calculate_button.pack(side='left', padx=6)
        self.spectral_settings_button = ttk.Button(controls, text='Parámetros espectrales',
                                                   command=self.spectral_dialog)
        self.spectral_settings_button.pack(side='left')
        options = ttk.Frame(parent, style='Card.TFrame', padding=(0, 6))
        options.pack(fill='x')
        ttk.Checkbutton(options, text='Incluir también las no revisadas (exploratorio)',
                        variable=self.include_unreviewed, style='Card.TCheckbutton',
                        command=self.invalidate_spectrum).pack(side='left')
        ttk.Checkbutton(options, text='Escala dB', variable=self.spectral_db,
                        style='Card.TCheckbutton', command=self.draw_spectrum).pack(side='right')
        # Nota teal con el estado del resultado, como las notas de figuras de la guía.
        note = tk.Frame(parent, background=IRIS_LIGHT)
        note.pack(fill='x', pady=(0, 6))
        tk.Frame(note, background=IRIS, width=4).pack(side='left', fill='y')
        message = ttk.Label(note, textvariable=self.spectral_message, style='Info.TLabel',
                            padding=(12, 8))
        message.pack(side='left', fill='x', expand=True)
        message.bind('<Configure>', lambda e: message.configure(wraplength=max(e.width - 30, 200)))
        summary_frame = ttk.Frame(parent, style='Card.TFrame')
        summary_frame.pack(fill='x')
        self.spectral_summary = tk.Text(summary_frame, height=4, width=1, wrap='word',
                                        state='disabled', relief='flat', background=BG,
                                        foreground=TEXT, padx=10, pady=6, borderwidth=0,
                                        font='TkDefaultFont')
        scroll = ttk.Scrollbar(summary_frame, command=self.spectral_summary.yview)
        scroll.pack(side='right', fill='y')
        self.spectral_summary.configure(yscrollcommand=scroll.set)
        self.spectral_summary.pack(fill='x', expand=True)
        export_bar = ttk.Frame(parent, style='Card.TFrame', padding=(0, 8, 0, 0))
        export_bar.pack(side='bottom', fill='x')
        self.export_button = ttk.Button(export_bar, text='Exportar resultados', command=self.export_spectrum)
        self.export_button.pack(side='left')
        ttk.Checkbutton(export_bar, text='Incluir gráfico PNG', variable=self.export_png,
                        style='Card.TCheckbutton').pack(side='left', padx=8)
        table_frame = ttk.Frame(parent, style='Card.TFrame', padding=(0, 6, 0, 0))
        table_frame.pack(side='bottom', fill='x')
        columns = ('session', 'band', 'range', 'absolute', 'relative')
        self.band_table = ttk.Treeview(table_frame, columns=columns, show='headings', height=4)
        self.set_band_columns(SINGLE_COLUMNS)
        scroll = ttk.Scrollbar(table_frame, command=self.band_table.yview)
        scroll.pack(side='right', fill='y')
        self.band_table.configure(yscrollcommand=scroll.set)
        self.band_table.pack(fill='x', expand=True)
        self.spectral_figure = Figure(figsize=(8, 4), dpi=100, layout='constrained')
        self.spectral_canvas = FigureCanvasTkAgg(self.spectral_figure, parent)
        self.spectral_toolbar = toolbar_class(self.spectral_canvas, parent, pack_toolbar=False)
        self.spectral_toolbar.configure(background='white')
        self.spectral_toolbar.pack(side='bottom', fill='x')
        self.spectral_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.invalidate_spectrum('Seleccioná un estudio y un electrodo.')

    def set_band_columns(self, columns):
        self.band_table.configure(columns=[name for name, label, width in columns])
        for name, label, width in columns:
            self.band_table.heading(name, text=label)
            self.band_table.column(name, width=width, minwidth=60)

    def fill_band_table(self, sessions):
        """Una fila por sesión y banda; al comparar NS/SD, una fila por banda con ambas sesiones."""
        self.band_table.delete(*self.band_table.get_children())
        codes = [SESSION_CODES.get(session['session']) for session in sessions]
        if len(sessions) < 2 or sorted(codes) != ['ns', 'sd']:
            self.set_band_columns(SINGLE_COLUMNS)
            for session in sessions:
                for band in session['bands']:
                    self.band_table.insert('', 'end', values=(
                        session['condition'], band['band'], f"{band['low_hz']:g}–{band['high_hz']:g}",
                        f"{band['absolute_uv2']:.6g}", format_relative(band['relative_percent'])))
            self.band_table.configure(height=4)
            return
        self.set_band_columns(COMPARE_COLUMNS)
        by_code = {code: {band['band']: band for band in session['bands']}
                   for code, session in zip(codes, sessions)}
        bands = [band for band in sessions[codes.index('ns')]['bands']]
        for band in bands:
            ns, sd = by_code['ns'].get(band['band']), by_code['sd'].get(band['band'])
            self.band_table.insert('', 'end', values=(
                band['band'], f"{band['low_hz']:g}–{band['high_hz']:g}",
                f"{ns['absolute_uv2']:.6g}" if ns else '—', f"{sd['absolute_uv2']:.6g}" if sd else '—',
                format_relative(ns['relative_percent']) if ns else '—',
                format_relative(sd['relative_percent']) if sd else '—'))
        self.band_table.configure(height=max(4, len(bands)))

    def invalidate_spectrum(self, message='Resultados pendientes de cálculo.'):
        self.spectral_revision += 1
        self.spectral_result = None
        self.spectral_message.set(message)
        if hasattr(self, 'spectral_canvas'):
            self.band_table.delete(*self.band_table.get_children())
            self.set_spectral_summary('')
            self.spectral_figure.clear()
            ax = self.spectral_figure.add_subplot(111)
            ax.set_axis_off()
            ax.text(.5, .5, 'Calculá la PSD para ver los resultados', ha='center', va='center',
                    transform=ax.transAxes, color='#62767c')
            self.spectral_toolbar.update()
            self.spectral_canvas.draw_idle()
            self.export_button.configure(state='disabled')

    def set_spectral_summary(self, text):
        self.spectral_summary.configure(state='normal')
        self.spectral_summary.delete('1.0', 'end')
        self.spectral_summary.insert('1.0', text)
        self.spectral_summary.configure(state='disabled')

    def update_spectral_controls(self):
        enabled = self.recording is not None and bool(self.channel.get()) and not self.busy
        self.calculate_button.configure(state='normal' if enabled else 'disabled')
        self.spectral_settings_button.configure(state='normal' if self.recording and not self.busy else 'disabled')
        self.export_button.configure(state='normal' if self.spectral_result is not None and not self.busy else 'disabled')

    def spectral_dialog(self):
        if self.recording is None:
            return
        dialog = tk.Toplevel(self.root)
        dialog.title('Parámetros espectrales')
        dialog.transient(self.root)
        dialog.resizable(False, False)
        style_dialog(dialog)
        body = ttk.Frame(dialog, padding=16)
        body.pack(fill='both')
        values = (self.spectral_config.seconds, self.spectral_config.overlap, *self.spectral_config.edges)
        variables = [tk.StringVar(value=f'{v:g}') for v in values]
        labels = ('Segmento Welch (s)', 'Solapamiento (%)', 'Inicio delta (Hz)',
                  'Delta / theta (Hz)', 'Theta / alfa (Hz)', 'Alfa / beta (Hz)', 'Fin beta (Hz)')
        for row, (label, var) in enumerate(zip(labels, variables)):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky='w', pady=4)
            ttk.Entry(body, textvariable=var, width=14).grid(row=row, column=1, padx=12)
        message = tk.StringVar()
        ttk.Label(body, textvariable=message, wraplength=400, style='Muted.TLabel').grid(
            row=7, column=0, columnspan=2, pady=8)
        def candidate():
            nums = [float(v.get().strip().replace(',', '.')) for v in variables]
            return SpectralConfig(nums[0], nums[1], tuple(nums[2:]))
        def preview(*args):
            try:
                effective = candidate().effective(self.recording.sfreq)
                message.set(f"Efectivo: {effective['seconds']:g} s · "
                            f"{effective['overlap_percent']:g}% · Δf={effective['frequency_step']:g} Hz\n"
                            f"Segmento: {effective['nperseg']} muestras · Solapamiento: {effective['noverlap']} muestras\n"
                            'Hann periódica · Media por segmento · Sin relleno de ceros')
            except (ValueError, OverflowError) as exc:
                message.set(str(exc))
        def apply():
            try:
                config = candidate()
                config.effective(self.recording.sfreq)
            except (ValueError, OverflowError) as exc:
                message.set(str(exc))
                return
            self.spectral_config = config
            self.invalidate_spectrum()
            dialog.destroy()
        for var in variables:
            var.trace_add('write', preview)
        def reset():
            defaults = SpectralConfig()
            for var, value in zip(variables, (defaults.seconds, defaults.overlap, *defaults.edges)):
                var.set(f'{value:g}')
        buttons = ttk.Frame(body)
        buttons.grid(row=8, column=0, columnspan=2, pady=6)
        for label, callback in (('Restablecer', reset), ('Cancelar', dialog.destroy), ('Aplicar', apply)):
            ttk.Button(buttons, text=label, style='Accent.TButton' if label == 'Aplicar' else 'TButton',
                       command=callback).pack(side='left', padx=3)
        preview()
        dialog.grab_set()

    def calculate_spectrum(self):
        if self.busy or not self.recording or not self.channel.get():
            return
        if not self.flush_autosave():
            self.spectral_message.set('No se pudieron guardar las anotaciones. Reintentá el guardado antes de calcular.')
            return
        if self.store.blocked:
            self.spectral_message.set(self.store.blocked)
            return
        try:
            studies = select_studies(self.recording.study, self.inventory,
                                     self.spectral_mode.get() == 'Comparar NS/SD')
            self.spectral_config.effective(self.recording.sfreq)
        except ValueError as exc:
            self.invalidate_spectrum(str(exc))
            return
        self.invalidate()
        self.invalidate_spectrum('Calculando Welch por epoch…')
        revision = self.spectral_revision
        channel, config, filtering = self.channel.get(), self.spectral_config, self.config
        include = self.include_unreviewed.get()
        ica_selections = None
        if self.recording.ica_context is not None:
            from tps.core.ica import snapshot
            ica_selections = snapshot(studies, self.data_dir, self.annotations_dir, 'confirmed')
            current = self.recording.ica_context
            ica_selections[str(self.recording.study.path)] = dict(
                mode='confirmed', source=current['source'], revision=current['revision'],
                model_sha256=current['model_sha256'])
        expected = {str(self.recording.study.path): (self.store.identity, self.store.digest)}
        data_dir, annotations_dir = self.data_dir, self.annotations_dir
        self.spectral_pending = True
        def work():
            return run_analysis(studies, channel, filtering, config, data_dir, annotations_dir, include, expected, ica_selections)
        def done(result):
            self.spectral_pending = False
            if revision != self.spectral_revision:
                return
            self.spectral_result = result
            self.spectral_message.set('Exploratorio · Incluye epochs no revisadas' if include
                                      else 'Resultado basado únicamente en epochs aceptadas')
            self.status.set('Análisis espectral listo · ' +
                            ('Exploratorio' if include else 'Solo epochs aceptadas'))
            self.render_spectral_result()
        def failed(exc):
            self.spectral_pending = False
            if revision == self.spectral_revision:
                self.spectral_message.set(str(exc))
        self.submit(work, done, on_error=failed)

    def render_spectral_result(self):
        result = self.spectral_result
        self.fill_band_table(result['sessions'])
        lines = []
        for session in result['sessions']:
            count = session['counts']
            lines.append(f"{session['condition']}: {count['used']} epochs / {count['seconds']:g} s · "
                         f"Artefactos: {count['artifacts']} · No revisadas: {count['pending']} · "
                         f"Parciales excluidas: {count['partial']} · Referencia: {session['reference']}")
            lines.extend(session['warnings'])
        if len(result['sessions']) > 1:
            lines.append('NS = sueño normal (ses-1) · SD = privación del sueño (ses-2).')
        effective = result['sessions'][0]['effective']
        lines.append(f"Welch: {effective['seconds']:g} s · {effective['overlap_percent']:g}% · "
                     f"Δf={effective['frequency_step']:g} Hz · Hann · Referencia original. "
                     f"Relativa respecto de {result['spectral']['edges'][0]:g}–{result['spectral']['edges'][-1]:g} Hz.")
        for session in result['sessions']:
            correction = session.get('ica')
            lines.append(session['session'] + ': ' + ('ICA ' + correction['revision'][:8] +
                         ' · componentes excluidos ' + str([i+1 for i in correction['excluded_components']]) +
                         ' · canales sin corregir ' + ', '.join(correction['excluded_channels']) if correction else 'Sin ICA'))
        filtering = result['filter']
        lines.append('Filtrado: ' + FilterConfig(
            filtering['enabled'], filtering['low'], filtering['high'], filtering.get('notch')).label)
        self.set_spectral_summary('\n'.join(lines))
        self.draw_spectrum()

    def draw_spectrum(self):
        if self.spectral_result is None:
            return
        if self.spectral_toolbar.mode:
            if self.spectral_toolbar.mode.name == 'PAN':
                self.spectral_toolbar.pan()
            else:
                self.spectral_toolbar.zoom()
        plot_result(self.spectral_figure, self.spectral_result, self.spectral_db.get())
        self.spectral_toolbar.update()
        self.spectral_toolbar.push_current()
        self.spectral_canvas.draw_idle()

    def export_spectrum(self):
        if self.spectral_result is None or self.busy:
            return
        parent = filedialog.askdirectory(parent=self.root, title='Carpeta para exportar resultados')
        if not parent:
            return
        result, png, db = self.spectral_result, self.export_png.get(), self.spectral_db.get()
        self.submit(lambda: export_result(result, parent, png, db),
                    lambda path: self.status.set(f'Resultados exportados: {path}'),
                    on_error=lambda exc: self.status.set(f'No se pudo exportar: {exc}'))
