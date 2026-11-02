"""Spectral tab and configuration, keeping computation out of the UI thread."""
import tkinter as tk
from tkinter import ttk, filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .spectral import SpectralConfig, select_studies, run_analysis, plot_result, export_result


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
        self.spectral_view = tk.Canvas(parent, highlightthickness=0)
        tab_scroll = ttk.Scrollbar(parent, command=self.spectral_view.yview)
        tab_scroll.pack(side='right', fill='y')
        self.spectral_view.configure(yscrollcommand=tab_scroll.set)
        self.spectral_view.pack(fill='both', expand=True)
        body = ttk.Frame(self.spectral_view)
        body_window = self.spectral_view.create_window(0, 0, window=body, anchor='nw')
        def resize(event):
            height = max(560, event.height)
            self.spectral_view.itemconfigure(body_window, width=event.width, height=height)
            self.spectral_view.configure(scrollregion=(0, 0, event.width, height))
        self.spectral_view.bind('<Configure>', resize)
        parent = body
        controls = ttk.Frame(parent, padding=(8, 8, 8, 0))
        controls.pack(fill='x')
        mode = ttk.Combobox(controls, textvariable=self.spectral_mode,
                            values=('Estudio actual', 'Comparar NS/SD'), state='readonly', width=22)
        mode.pack(side='left')
        mode.bind('<<ComboboxSelected>>', lambda e: self.invalidate_spectrum())
        self.calculate_button = ttk.Button(controls, text='Calcular', command=self.calculate_spectrum)
        self.calculate_button.pack(side='left', padx=6)
        self.spectral_settings_button = ttk.Button(controls, text='Parámetros espectrales',
                                                   command=self.spectral_dialog)
        self.spectral_settings_button.pack(side='left')
        options = ttk.Frame(parent, padding=(8, 4))
        options.pack(fill='x')
        ttk.Checkbutton(options, text='Incluir también las no revisadas (exploratorio)',
                        variable=self.include_unreviewed, command=self.invalidate_spectrum).pack(side='left')
        ttk.Checkbutton(options, text='Escala dB', variable=self.spectral_db,
                        command=self.draw_spectrum).pack(side='right')
        ttk.Label(parent, textvariable=self.spectral_message, wraplength=750,
                  padding=(8, 4)).pack(fill='x')
        summary_frame = ttk.Frame(parent, padding=(8, 0))
        summary_frame.pack(fill='x')
        self.spectral_summary = tk.Text(summary_frame, height=4, width=1, wrap='word',
                                        font=('DejaVu Sans', 9), state='disabled', relief='flat')
        scroll = ttk.Scrollbar(summary_frame, command=self.spectral_summary.yview)
        scroll.pack(side='right', fill='y')
        self.spectral_summary.configure(yscrollcommand=scroll.set)
        self.spectral_summary.pack(fill='x', expand=True)
        export_bar = ttk.Frame(parent, padding=8)
        export_bar.pack(side='bottom', fill='x')
        self.export_button = ttk.Button(export_bar, text='Exportar resultados', command=self.export_spectrum)
        self.export_button.pack(side='left')
        ttk.Checkbutton(export_bar, text='Incluir gráfico PNG', variable=self.export_png).pack(side='left', padx=8)
        table_frame = ttk.Frame(parent, padding=(8, 0))
        table_frame.pack(side='bottom', fill='x')
        columns = ('session', 'band', 'range', 'absolute', 'relative')
        self.band_table = ttk.Treeview(table_frame, columns=columns, show='headings', height=4)
        for name, label, width in zip(columns, ('Sesión', 'Banda', 'Hz', 'Potencia (µV²)', 'Relativa (%)'),
                                      (150, 80, 90, 130, 110)):
            self.band_table.heading(name, text=label)
            self.band_table.column(name, width=width, minwidth=60)
        scroll = ttk.Scrollbar(table_frame, command=self.band_table.yview)
        scroll.pack(side='right', fill='y')
        self.band_table.configure(yscrollcommand=scroll.set)
        self.band_table.pack(fill='x', expand=True)
        self.spectral_figure = Figure(figsize=(8, 4), dpi=100, layout='constrained')
        self.spectral_canvas = FigureCanvasTkAgg(self.spectral_figure, parent)
        self.spectral_toolbar = toolbar_class(self.spectral_canvas, parent, pack_toolbar=False)
        self.spectral_toolbar.pack(side='bottom', fill='x')
        self.spectral_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.invalidate_spectrum('Seleccioná un estudio y un electrodo.')

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
        ttk.Label(body, textvariable=message, wraplength=400).grid(row=7, column=0, columnspan=2, pady=8)
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
            ttk.Button(buttons, text=label, command=callback).pack(side='left', padx=3)
        preview()
        dialog.grab_set()

    def calculate_spectrum(self):
        if self.busy or not self.recording or not self.channel.get():
            return
        if self.store.dirty:
            self.spectral_message.set('Guardá las anotaciones antes de calcular.')
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
        expected = {str(self.recording.study.path): (self.store.identity, self.store.digest)}
        data_dir, annotations_dir = self.data_dir, self.annotations_dir
        self.spectral_pending = True
        def work():
            return run_analysis(studies, channel, filtering, config, data_dir, annotations_dir, include, expected)
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
        self.band_table.delete(*self.band_table.get_children())
        lines = []
        for session in result['sessions']:
            count = session['counts']
            lines.append(f"{session['condition']}: {count['used']} epochs / {count['seconds']:g} s · "
                         f"Artefactos: {count['artifacts']} · No revisadas: {count['pending']} · "
                         f"Parciales excluidas: {count['partial']} · Referencia: {session['reference']}")
            lines.extend(session['warnings'])
            for band in session['bands']:
                relative = band['relative_percent']
                self.band_table.insert('', 'end', values=(session['condition'], band['band'],
                    f"{band['low_hz']:g}–{band['high_hz']:g}", f"{band['absolute_uv2']:.6g}",
                    f'{relative:.4g}' if relative is not None else 'No disponible'))
        effective = result['sessions'][0]['effective']
        lines.append(f"Welch: {effective['seconds']:g} s · {effective['overlap_percent']:g}% · "
                     f"Δf={effective['frequency_step']:g} Hz · Hann · Referencia original. "
                     f"Relativa respecto de {result['spectral']['edges'][0]:g}–{result['spectral']['edges'][-1]:g} Hz.")
        lines.append('Filtrado: ' + (f"{result['filter']['low']:g}–{result['filter']['high']:g} Hz"
                                     if result['filter']['enabled'] else 'desactivado'))
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
