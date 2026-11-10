"""Epoch review controls and optional multichannel display for the desktop app."""
import tkinter as tk
from tkinter import ttk

from types import SimpleNamespace

from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np

from tps.core.annotations import STATES, REASONS
from tps.core.processing import FilterConfig
from tps.gui.theme import (AMBER, AMBER_DARK, AMBER_LIGHT, BORDER, ERROR, IRIS, IRIS_DARK,
                           IRIS_LIGHT, TEXT, style_dialog)

COLORS = {'Sin revisar': '#edf1f2', 'Aceptada': '#d5eee2', 'Con artefactos': '#f9d8d5'}
DISPLAY_MODES = ('Un electrodo', 'Varios (hasta 8)', 'Todos apilados')
STACK_PAGES = ('Todos', '32', '16')   # channels visible at once; the rest via the plot scrollbar
TRACE = '#0e7c86'
DRAG_PIXELS = 6    # mouse movement below this is a click, not a drag
BAD_TRACE = '#b8c2c4'
BAD_LABEL = '#a12b25'


def filter_state_text(config) -> str:
    """Plain-language filter state: 'Sin filtrar' or what is on."""
    parts = [f'pasabanda {config.low:g}–{config.high:g} Hz'] if config.enabled else []
    if config.notch is not None:
        parts.append(f'notch {config.notch:g} Hz')
    return 'Filtrada · ' + ' + '.join(parts) if parts else 'Sin filtrar'


class ReviewUI:
    def init_review(self):
        self.config = FilterConfig()
        self.filter_reviewed = False
        self.store = None
        self.selected_epoch = None
        self.displayed = None
        self.active_patient = ''
        self.active_study_index = -1
        self.multi = tk.BooleanVar(value=False)
        # Display: one electrode, up to 8 in panels, or every channel stacked in one plot (eegplot-like).
        self.display_mode = tk.StringVar(value=DISPLAY_MODES[0])
        self.stacked = False
        self.stack_scale = 80.0          # µV between neighbouring traces
        self.stack_scale_text = tk.StringVar(value='80')
        self.stack_page = tk.StringVar(value='32')
        self.signal_label = tk.StringVar(value='Original')
        self.save_label = tk.StringVar(value='Sin estudio')
        self.count_label = tk.StringVar(value='')
        self.epoch_label = tk.StringVar(value='Seleccioná una época')
        self.context_label = tk.StringVar(value='Sin evaluación previa')
        self.reason = tk.StringVar()
        self.note = tk.StringVar()
        self.mark_state = tk.StringVar(value=STATES[0])
        self.last_multi = ()
        self.loading_editor = False
        self.autosave_job = None
        for variable in (self.mark_state, self.reason):
            variable.trace_add('write', self.editor_changed)
        self.note.trace_add('write', lambda *args: self.editor_changed(debounce=True))

    def build_multi_selector(self, selection):
        row = ttk.Frame(selection, style='Card.TFrame')
        row.pack(fill='x', pady=(8, 0))
        ttk.Label(row, text='Vista', style='Card.TLabel').pack(side='left')
        self.multi_toggle = ttk.Combobox(row, textvariable=self.display_mode, values=DISPLAY_MODES,
                                         state='readonly', width=18)
        self.multi_toggle.pack(side='right')
        self.multi_toggle.bind('<<ComboboxSelected>>', lambda e: self.display_mode_changed())
        # Stacked view options: scale between traces and channels per screen.
        self.stack_frame = ttk.Frame(selection, style='Card.TFrame')
        ttk.Label(self.stack_frame, text='Escala (µV)', style='Card.TLabel').pack(side='left')
        ttk.Button(self.stack_frame, text='−', width=2, style='Tool.TButton',
                   command=lambda: self.zoom_stack(.8)).pack(side='left', padx=(6, 2))
        scale_entry = ttk.Entry(self.stack_frame, textvariable=self.stack_scale_text, width=5, justify='center')
        scale_entry.pack(side='left')
        scale_entry.bind('<Return>', lambda e: self.set_stack_scale())
        ttk.Button(self.stack_frame, text='+', width=2, style='Tool.TButton',
                   command=lambda: self.zoom_stack(1.25)).pack(side='left', padx=(2, 0))
        page = ttk.Combobox(self.stack_frame, textvariable=self.stack_page, values=STACK_PAGES,
                            state='readonly', width=6)
        page.pack(side='right')
        page.bind('<<ComboboxSelected>>', lambda e: (self.resize_plot(), self.canvas.draw_idle()))
        ttk.Label(self.stack_frame, text='Canales', style='Card.TLabel').pack(side='right', padx=(0, 4))
        self.multi_frame = ttk.Frame(selection, style='Card.TFrame')
        self.channel_list = tk.Listbox(self.multi_frame, selectmode='multiple', exportselection=False,
                                       height=5, relief='flat', highlightthickness=1,
                                       highlightbackground=BORDER, highlightcolor=IRIS,
                                       selectbackground=IRIS, selectforeground='white',
                                       foreground=TEXT, activestyle='none')
        scroll = ttk.Scrollbar(self.multi_frame, command=self.channel_list.yview)
        self.channel_list.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.channel_list.pack(fill='x', expand=True)
        self.channel_list.bind('<<ListboxSelect>>', self.multi_changed)

    def build_epoch_list(self, parent):
        epochs = self.card(parent, 'ÉPOCAS · 5 SEGUNDOS', expand=True)
        self.epoch_tree = ttk.Treeview(epochs, columns=('interval', 'state', 'noise'), height=8,
                                      selectmode='browse')
        self.epoch_tree.heading('#0', text='Nº')
        self.epoch_tree.column('#0', width=32, stretch=False)
        self.epoch_tree.heading('interval', text='Intervalo (s)')
        self.epoch_tree.column('interval', width=85, stretch=False)
        self.epoch_tree.heading('state', text='Estado')
        self.epoch_tree.column('state', width=105)
        self.epoch_tree.heading('noise', text='Aviso')
        self.epoch_tree.column('noise', width=65)
        self.epoch_tree.tag_configure('sospechosa', foreground='#9b5600')
        scroll = ttk.Scrollbar(epochs, orient='vertical', command=self.epoch_tree.yview)
        scroll.pack(side='right', fill='y')
        self.epoch_tree.configure(yscrollcommand=scroll.set)
        self.epoch_tree.pack(fill='both', expand=True)
        for state in STATES:
            self.epoch_tree.tag_configure(state, background=COLORS[state])
        self.epoch_tree.bind('<<TreeviewSelect>>', self.epoch_from_list)
        ttk.Label(epochs, textvariable=self.count_label, style='CardBold.TLabel',
                  wraplength=280).pack(anchor='w', pady=(6, 2))
        ttk.Label(epochs, textvariable=self.save_label, style='CardMuted.TLabel',
                  wraplength=280).pack(anchor='w')
        self.retry_save_button = ttk.Button(epochs, text='Reintentar guardado',
                                            command=self.save_annotations, state='disabled')
        self.retry_save_button.pack(anchor='w', pady=5)

    def build_review_editor(self, chart):
        """Editor de la época elegida: estado, motivo y nota en una fila; contexto debajo."""
        editor = ttk.Frame(chart, style='Card.TFrame')
        editor.pack(side='bottom', fill='x', pady=(6, 0))
        row = ttk.Frame(editor, style='Card.TFrame')
        row.pack(fill='x')
        ttk.Label(row, textvariable=self.epoch_label, style='CardBold.TLabel').pack(side='left', padx=(0, 12))
        self.mark_buttons = []
        for label, key in zip(STATES, ('U', 'A', 'R')):
            button = ttk.Radiobutton(row, text=f'{label} ({key})', value=label,
                                     variable=self.mark_state, style='Card.TRadiobutton')
            button.pack(side='left', padx=(0, 8))
            self.mark_buttons.append(button)
        self.reason_box = ttk.Combobox(row, textvariable=self.reason, values=REASONS,
                                       state='disabled', width=18)
        self.reason_box.pack(side='left', padx=(4, 0))
        ttk.Label(row, text='Nota', style='Card.TLabel').pack(side='left', padx=(10, 6))
        self.note_entry = ttk.Entry(row, textvariable=self.note, validate='key',
                                    validatecommand=(self.root.register(lambda value: len(value) <= 500), '%P'))
        self.note_entry.pack(side='left', fill='x', expand=True)
        details = ttk.Frame(editor, style='Card.TFrame')
        details.pack(fill='x', pady=(3, 0))
        for variable in (self.screening_detail, self.context_label):
            label = ttk.Label(details, textvariable=variable, style='CardMuted.TLabel')
            label.pack(side='left', anchor='w', padx=(0, 16))
        for key, state in zip(('u', 'a', 'r'), STATES):
            self.root.bind(f'<KeyPress-{key}>', lambda event, s=state: self.shortcut(event, s))
            self.root.bind(f'<KeyPress-{key.upper()}>', lambda event, s=state: self.shortcut(event, s))

    def configuration_dialog(self):
        """Filter settings with the current state, quick presets and a live «al aplicar» summary.

        The cut-off fields are greyed out while the band-pass is off, so it is clear that
        typing values is not enough: the filter must be switched on (or a preset chosen)."""
        dialog = tk.Toplevel(self.root)
        dialog.title('Configuración de filtrado')
        dialog.transient(self.root)
        dialog.resizable(False, False)
        style_dialog(dialog)
        body = ttk.Frame(dialog, padding=20)
        body.pack(fill='both')
        current = self.config
        ttk.Label(body, text='Estado actual: ' + filter_state_text(current), style='Bold.TLabel',
                  foreground=IRIS_DARK if current.active else AMBER_DARK).pack(anchor='w')
        ttk.Label(body, text='Por defecto la señal se muestra sin filtrar. Activá el pasabanda y/o el notch, '
                             'o elegí un ajuste rápido, y presioná «Aplicar».',
                  style='Muted.TLabel', wraplength=430).pack(anchor='w', pady=(2, 10))

        enabled = tk.BooleanVar(value=current.enabled)
        low = tk.StringVar(value=f'{current.low:g}')
        high = tk.StringVar(value=f'{current.high:g}')
        notch_enabled = tk.BooleanVar(value=current.notch is not None)
        notch_frequency = tk.StringVar(value=f'{current.notch or 50:g}')

        def preset(band, notch):
            enabled.set(band)
            if band:
                low.set('0.2'); high.set('45')
            notch_enabled.set(notch is not None)
            if notch is not None:
                notch_frequency.set(str(notch))
            refresh()

        ttk.Label(body, text='Ajustes rápidos', style='Bold.TLabel').pack(anchor='w')
        presets = ttk.Frame(body)
        presets.pack(fill='x', pady=(4, 12))
        for column, (text, band, notch) in enumerate((('Sin filtrar', False, None),
                                                      ('Pasabanda 0,2–45 Hz', True, None),
                                                      ('0,2–45 Hz + notch 50 Hz', True, 50))):
            presets.columnconfigure(column, weight=1, uniform='presets')
            ttk.Button(presets, text=text, command=lambda b=band, n=notch: preset(b, n)).grid(
                row=0, column=column, sticky='ew', padx=(0 if column == 0 else 6, 0))

        ttk.Checkbutton(body, text='Activar pasabanda (FIR)', variable=enabled,
                        command=lambda: refresh()).pack(anchor='w')
        band_row = ttk.Frame(body, padding=(24, 4, 0, 8))
        band_row.pack(fill='x')
        entries = []
        for column, (label, variable) in enumerate((('Corte inferior (Hz)', low), ('Corte superior (Hz)', high))):
            band_row.columnconfigure(column, weight=1, uniform='cuts')
            cell = ttk.Frame(band_row)
            cell.grid(row=0, column=column, sticky='ew', padx=(0 if column == 0 else 10, 0))
            ttk.Label(cell, text=label).pack(anchor='w')
            entry = ttk.Entry(cell, textvariable=variable, width=10)
            entry.pack(fill='x')
            entries.append(entry)
        notch_row = ttk.Frame(body)
        notch_row.pack(fill='x', pady=(0, 10))
        ttk.Checkbutton(notch_row, text='Activar notch (red eléctrica)', variable=notch_enabled,
                        command=lambda: refresh()).pack(side='left')
        notch_box = ttk.Combobox(notch_row, textvariable=notch_frequency, values=('50', '60'), width=5)
        notch_box.pack(side='left', padx=8)
        notch_box.bind('<<ComboboxSelected>>', lambda e: refresh())
        ttk.Label(notch_row, text='Hz').pack(side='left')

        # What «Aplicar» will do, kept in sync with the controls.
        summary = tk.StringVar()
        note = tk.Frame(body, background=IRIS_LIGHT)
        note.pack(fill='x', pady=(0, 8))
        stripe = tk.Frame(note, background=IRIS, width=4)
        stripe.pack(side='left', fill='y')
        summary_label = ttk.Label(note, textvariable=summary, style='Info.TLabel', padding=(10, 8),
                                  wraplength=410)
        summary_label.pack(side='left', fill='x', expand=True)
        ttk.Label(body, text='Pasabanda FIR · Fase cero · Ventana Hamming · Notch IIR · Fase cero · Q = 30.\n'
                             'Se aplican al canal continuo completo, antes de dividir en épocas.',
                  style='Muted.TLabel').pack(anchor='w', pady=(0, 4))
        error = tk.StringVar()
        ttk.Label(body, textvariable=error, foreground=ERROR, wraplength=430).pack(fill='x')

        def refresh(*args):
            for entry in entries:
                entry.configure(state='normal' if enabled.get() else 'disabled')
            notch_box.configure(state='readonly' if notch_enabled.get() else 'disabled')
            parts = []
            if enabled.get():
                parts.append(f"pasabanda {low.get().strip() or '?'}–{high.get().strip() or '?'} Hz")
            if notch_enabled.get():
                parts.append(f'notch {notch_frequency.get()} Hz')
            if parts:
                summary.set('Al aplicar: la señal se filtrará con ' + ' y '.join(parts) + '.')
                stripe.configure(background=IRIS)
                note.configure(background=IRIS_LIGHT)
            else:
                summary.set('Al aplicar: la señal se mostrará SIN FILTRAR (pasabanda y notch apagados).')
                stripe.configure(background=AMBER)
                note.configure(background=AMBER_LIGHT)
            summary_label.configure(background=note.cget('background'))
            error.set('')

        for variable in (low, high):
            variable.trace_add('write', refresh)
        refresh()

        buttons = ttk.Frame(body)
        buttons.pack(fill='x', pady=(8, 0))
        def apply():
            try:
                candidate = FilterConfig(enabled.get(), float(low.get().replace(',', '.')),
                                         float(high.get().replace(',', '.')),
                                         float(notch_frequency.get()) if notch_enabled.get() else None)
                candidate.validate(self.recording.sfreq)
            except (ValueError, AttributeError) as exc:
                error.set(str(exc))
                return
            dialog.destroy()
            self.apply_config(candidate)
        ttk.Button(buttons, text='Aplicar', style='Accent.TButton', command=apply).pack(side='right')
        ttk.Button(buttons, text='Cancelar', command=dialog.destroy).pack(side='right', padx=8)
        dialog.bind('<Return>', lambda e: apply())
        dialog.bind('<Escape>', lambda e: dialog.destroy())
        dialog.grab_set()

    def apply_config(self, candidate):
        if not self.flush_autosave():
            return
        if self.recording is None:
            return
        self.invalidate()
        recording = self.recording
        channels = self.visible_channels()
        start = self.start
        self.status.set('Preparando configuración de filtrado…')
        def work():
            recording.filter.prepare(candidate)
            if channels:
                return recording.read_channels(channels, start, candidate)
            return None
        def done(segment):
            self.invalidate_spectrum()
            changed = candidate != self.config
            self.config = candidate
            self.filter_reviewed = True
            if changed and self.screening_use_filter.get():
                self.clear_screening()
            self.update_screening_signal()
            self.signal_label.set(candidate.label)
            self.refresh_ica()
            if segment is not None:
                self.segment_loaded(segment)
            else:
                self.status.set(f'Configuración aplicada: {candidate.label}')
        self.submit(work, done, on_error=lambda exc: self.status.set(
            f'No se aplicó el filtro: {exc} Se conserva {self.config.label.lower()}.'))

    def visible_channels(self):
        if self.stacked and self.recording is not None:
            return list(self.recording.channels)
        if self.multi.get():
            return [self.channel_list.get(i) for i in self.channel_list.curselection()]
        return [self.channel.get()] if self.channel.get() else []

    def display_mode_changed(self):
        mode = self.display_mode.get()
        self.stacked = mode == DISPLAY_MODES[2]
        self.multi.set(mode == DISPLAY_MODES[1])
        if self.stacked:
            self.stack_frame.pack(fill='x', pady=(6, 0), after=self.multi_toggle.master)
        else:
            self.stack_frame.pack_forget()
        self.toggle_multi()

    def zoom_stack(self, factor):
        self.stack_scale = min(5000.0, max(1.0, self.stack_scale * factor))
        self.stack_scale_text.set(f'{self.stack_scale:.3g}')
        self.restack()

    def set_stack_scale(self):
        try:
            value = float(self.stack_scale_text.get().replace(',', '.'))
            if not 0 < value <= 5000:
                raise ValueError
        except ValueError:
            self.stack_scale_text.set(f'{self.stack_scale:.3g}')
            self.status.set('La escala debe ser un número de µV entre 0 y 5000.')
            return
        self.stack_scale = value
        self.restack()

    def restack(self):
        if self.stacked and self.displayed is not None:
            self.draw_stacked(self.displayed, rebuild=True)

    def toggle_multi(self):
        if self.multi.get() and self.display_mode.get() != DISPLAY_MODES[1]:
            self.display_mode.set(DISPLAY_MODES[1])   # set from code (tests): keep the selector in sync
            self.stacked = False
            self.stack_frame.pack_forget()
        if self.multi.get():
            self.multi_frame.pack(fill='x', pady=6)
            self.channel_list.selection_clear(0, 'end')
            if self.recording and self.channel.get() in self.recording.channels:
                self.channel_list.selection_set(self.recording.channels.index(self.channel.get()))
            self.last_multi = self.channel_list.curselection()
        else:
            self.multi_frame.pack_forget()
        self.show_window(self.start)

    def multi_changed(self, event=None):
        chosen = self.channel_list.curselection()
        if len(chosen) > 8:
            self.channel_list.selection_clear(0, 'end')
            for index in self.last_multi:
                self.channel_list.selection_set(index)
            self.status.set('Seleccioná como máximo 8 electrodos.')
            return
        self.last_multi = chosen
        if self.multi.get():
            self.show_window(self.start)

    def reset_review(self):
        self.cancel_autosave()
        self.invalidate_spectrum()
        self.store = None
        self.selected_epoch = None
        self.displayed = None
        self.epoch_tree.delete(*self.epoch_tree.get_children())
        self.channel_list.delete(0, 'end')
        self.save_label.set('Sin estudio')
        self.count_label.set('')
        self.epoch_label.set('Seleccioná una época')
        self.context_label.set('Sin evaluación previa')
        self.note.set('')
        self.reason.set('')
        self.clear_screening()
        self.clear_bad_channels()

    def review_loaded(self, recording):
        self.store = recording.annotations
        self.channel_list.delete(0, 'end')
        for name in recording.channels:
            self.channel_list.insert('end', name)
        self.refresh_epochs()
        self.save_label.set(self.store.blocked or 'Guardado automático activo')
        self.selected_epoch = 0
        self.epoch_tree.selection_set('0')
        self.load_editor()
        # A configuration may be incompatible with a newly selected recording.
        try:
            self.config.validate(recording.sfreq)
        except ValueError:
            self.config = FilterConfig(False, .2, min(45., recording.sfreq / 4))
            self.status.set('El nuevo muestreo requiere otros cortes. Filtrado desactivado.')
        self.signal_label.set(self.config.label)
        self.update_screening_signal()
        self.refresh_bad_channels()

    def refresh_epochs(self):
        if not self.store:
            return
        counts = {s: 0 for s in STATES}
        for epoch, record in zip(self.store.epochs, self.store.records):
            state = record['state']
            counts[state] += 1
            # Screening findings first; otherwise flag epochs the batch pipeline rejected.
            badge = self.screening_badge(epoch.index) or (
                'Lote' if hasattr(self, 'batch_rejection') and self.batch_rejection(epoch.index) else '')
            values = (f'{epoch.start:g}–{epoch.stop:g}' + (' parcial' if epoch.partial else ''), state, badge)
            tags = (state, 'sospechosa') if badge else (state,)
            if self.epoch_tree.exists(str(epoch.index)):
                self.epoch_tree.item(str(epoch.index), values=values, tags=tags)
            else:
                self.epoch_tree.insert('', 'end', iid=str(epoch.index), text=str(epoch.index + 1),
                                       values=values, tags=tags)
        self.count_label.set(f"Aceptadas: {counts[STATES[1]]} · Artefactos: {counts[STATES[2]]}\n"
                             f"Pendientes: {counts[STATES[0]]} · Total: {len(self.store.epochs)}")

    def epoch_from_list(self, event=None):
        selection = self.epoch_tree.selection()
        if not self.store or not selection:
            return
        index = int(selection[0])
        if index == self.selected_epoch:
            return
        self.select_epoch(index, navigate=True)

    def select_epoch(self, index, navigate=False):
        if not self.flush_autosave():
            self.epoch_tree.selection_set(str(self.selected_epoch))
            return
        self.selected_epoch = index
        self.epoch_tree.selection_set(str(index))
        self.epoch_tree.see(str(index))
        self.load_editor()
        epoch = self.store.epochs[index]
        if navigate and not (self.start <= epoch.start and self.stop >= epoch.stop):
            self.show_window(epoch.start)
        else:
            self.draw_overlays()
        self.update_controls()

    # ------------------------------------------------------------ mouse on the plot
    def plot_press(self, event):
        """Left drag: zoom to the selected rectangle. Right drag: pan. Double click: restore."""
        self.drag = None
        if self.busy or event.inaxes not in self.figure.axes or event.xdata is None:
            return
        if event.button == 1 and event.dblclick:
            self.restore_view()
            return
        if event.button not in (1, 3):
            return
        ax = event.inaxes
        self.drag = dict(button=event.button, ax=ax, x=event.x, y=event.y, xdata=event.xdata,
                         ydata=event.ydata, xlim=ax.get_xlim(), ylim=ax.get_ylim(), rect=None, moved=False)
        if event.button == 3:
            self.canvas.get_tk_widget().configure(cursor='fleur')

    def plot_motion(self, event):
        drag = getattr(self, 'drag', None)
        if not drag or event.x is None:
            return
        ax = drag['ax']
        if not drag['moved'] and abs(event.x - drag['x']) + abs(event.y - drag['y']) < DRAG_PIXELS:
            return
        drag['moved'] = True
        if drag['button'] == 3:
            (x0, x1), (y0, y1) = drag['xlim'], drag['ylim']
            dx = (event.x - drag['x']) * (x1 - x0) / ax.bbox.width
            dy = (event.y - drag['y']) * (y1 - y0) / ax.bbox.height
            ax.set_xlim(x0 - dx, x1 - dx)     # shared x: every panel moves together
            ax.set_ylim(y0 - dy, y1 - dy)
        else:
            x, y = ax.transData.inverted().transform((event.x, event.y))
            if drag['rect'] is None:
                drag['rect'] = ax.add_patch(Rectangle((drag['xdata'], drag['ydata']), 0, 0, facecolor=IRIS,
                                                      alpha=.15, edgecolor=IRIS, linestyle='--', zorder=10))
            drag['rect'].set_bounds(min(drag['xdata'], x), min(drag['ydata'], y),
                                    abs(x - drag['xdata']), abs(y - drag['ydata']))
        self.canvas.draw_idle()

    def plot_release(self, event):
        drag, self.drag = getattr(self, 'drag', None), None
        self.canvas.get_tk_widget().configure(cursor='')
        if not drag:
            return
        ax = drag['ax']
        if drag['rect'] is not None:
            drag['rect'].remove()
        if drag['button'] == 1 and not drag['moved']:
            self.plot_clicked(SimpleNamespace(inaxes=ax, xdata=drag['xdata']))   # plain click: select the época
            return
        if drag['button'] == 1 and event.x is not None:
            x, y = ax.transData.inverted().transform((event.x, event.y))
            if abs(x - drag['xdata']) > 1e-3:
                ax.set_xlim(sorted((drag['xdata'], x)))
            if abs(y - drag['ydata']) > 1e-9:
                ax.set_ylim(sorted((drag['ydata'], y)))
            self.status.set('Zoom aplicado. Doble clic en el gráfico para volver a la ventana completa.')
        self.canvas.draw_idle()

    def restore_view(self):
        for ax, (xlim, ylim) in zip(self.figure.axes, getattr(self, 'home_limits', [])):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def plot_clicked(self, event):
        if self.busy or self.toolbar.mode or not self.store or event.inaxes not in self.figure.axes:
            return
        if event.xdata is None or not 0 <= event.xdata < self.recording.duration:
            return
        sample = int(event.xdata * self.recording.sfreq)
        for epoch in self.store.epochs:
            if epoch.first <= sample < epoch.last:
                self.select_epoch(epoch.index)
                break

    def load_editor(self):
        if self.store is None or self.selected_epoch is None:
            return
        epoch = self.store.epochs[self.selected_epoch]
        record = self.store.records[self.selected_epoch]
        self.epoch_label.set(f'Época {epoch.index + 1} · {epoch.start:g}–{epoch.stop:g} s'
                             + (' · Parcial' if epoch.partial else '') + ' · Marca global')
        self.loading_editor = True
        try:
            self.mark_state.set(record['state'])
            self.reason.set(record['reason'])
            self.note.set(record['note'])
        finally:
            self.loading_editor = False
        self.update_review_context()
        self.update_screening_detail()
        self.update_reason()

    def update_review_context(self):
        record = self.store.records[self.selected_epoch]
        context = record['filter']
        from tps.core.ica import context as ica_context
        mismatch = record.get('ica') != ica_context(self.recording)
        if context:
            config = FilterConfig(context['enabled'], context['low'], context['high'], context.get('notch'))
            screening = record.get('screening')
            label = ('Aceptación automática' if record['state'] == STATES[1] else 'Barrido automático') if screening else 'Última evaluación'
            limits = (f" · Pico a pico {screening['flat_uv']:g}–{screening['high_uv']:g} µV"
                      if screening else '')
            self.context_label.set(f"{label}: {config.label}{limits} · {', '.join(record['channels'])}"
                                   f" · {record['reviewed_at'][:19]} UTC")
        else:
            self.context_label.set('Sin evaluación previa')
        if record.get('ica'):
            self.context_label.set(self.context_label.get() + ' · ICA ' + record['ica']['revision'][:8])
        if mismatch and context:
            self.context_label.set(self.context_label.get() + ' · ATENCIÓN: revisión con otra configuración ICA')

    def update_reason(self):
        can_mark = self.store is not None and not self.store.blocked and not self.busy
        self.reason_box.configure(state='readonly' if can_mark and self.mark_state.get() == STATES[2]
                                  else 'disabled')

    def shortcut(self, event, state):
        if event.widget.winfo_toplevel() != self.root:
            return
        if event.widget.winfo_class() in ('Entry', 'TEntry', 'Text', 'TCombobox', 'Spinbox', 'TSpinbox'):
            return
        if self.busy or not self.store or self.store.blocked or self.displayed is None:
            return
        self.mark_state.set(state)
        if state == STATES[2]:
            self.reason_box.focus_set()
        return 'break'

    def cancel_autosave(self):
        if self.autosave_job is not None:
            self.root.after_cancel(self.autosave_job)
            self.autosave_job = None

    def flush_autosave(self):
        self.cancel_autosave()
        if self.store and self.store.dirty:
            return self.save_annotations()
        return True

    def editor_changed(self, *args, debounce=False):
        if self.loading_editor:
            return
        self.apply_mark(debounce=debounce)

    def apply_mark(self, debounce=False):
        if self.busy or not self.store or self.store.blocked or self.selected_epoch is None or self.displayed is None:
            return
        state = self.mark_state.get()
        reason = self.reason.get() if state == STATES[2] else ''
        note = self.note.get().strip()
        record = self.store.records[self.selected_epoch]
        if ((state, reason, note) == (record['state'], record['reason'], record['note'])
                and record.get('screening') is None):
            return
        try:
            self.store.mark(self.selected_epoch, state, reason, note,
                            self.displayed.channels, self.config)
        except ValueError as exc:
            self.status.set(str(exc))
            return
        self.cancel_autosave()
        self.refresh_epochs()
        self.update_review_context()
        if state != STATES[2]:
            self.loading_editor = True
            try:
                self.reason.set('')
            finally:
                self.loading_editor = False
        self.update_reason()
        self.draw_overlays()
        self.save_label.set('Guardando…')
        self.invalidate_spectrum('Anotaciones modificadas. Volvé a calcular el espectro.')
        if debounce:
            self.autosave_job = self.root.after(400, self.save_annotations)
        else:
            self.save_annotations()

    def save_annotations(self):
        self.cancel_autosave()
        if not self.store:
            return False
        try:
            self.store.save()
        except (OSError, ValueError) as exc:
            self.save_label.set('Error de guardado · Cambios en memoria' if self.store.dirty else 'Guardado bloqueado')
            self.status.set(f'No se pudo guardar: {exc}')
            self.update_controls()
            return False
        self.save_label.set('Guardado automáticamente')
        self.update_controls()
        return True

    def unsaved_choice(self):
        dialog = tk.Toplevel(self.root)
        dialog.title('Cambios sin guardar')
        dialog.transient(self.root)
        dialog.resizable(False, False)
        style_dialog(dialog)
        choice = 'cancel'
        body = ttk.Frame(dialog, padding=18)
        body.pack(fill='both')
        ttk.Label(body, text='Hay anotaciones sin guardar. ¿Qué querés hacer?',
                  wraplength=400).pack(pady=(0, 12))
        row = ttk.Frame(body)
        row.pack()
        def finish(value):
            nonlocal choice
            choice = value
            dialog.destroy()
        for label, value in (('Guardar', 'save'), ('Descartar', 'discard'), ('Cancelar', 'cancel')):
            ttk.Button(row, text=label, style='Accent.TButton' if value == 'save' else 'TButton',
                       command=lambda v=value: finish(v)).pack(side='left', padx=4)
        dialog.protocol('WM_DELETE_WINDOW', lambda: finish('cancel'))
        dialog.grab_set()
        self.root.wait_window(dialog)
        return choice

    def may_leave(self):
        if self.flush_autosave():
            return True
        choice = self.unsaved_choice()
        if choice == 'save':
            return self.save_annotations()
        return choice == 'discard'

    def draw_overlays(self):
        for artist in getattr(self, 'epoch_artists', []):
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self.epoch_artists = []
        if self.displayed is None or self.store is None:
            self.canvas.draw_idle()
            return
        for ax in self.figure.axes:
            for epoch, record in zip(self.store.epochs, self.store.records):
                if epoch.stop <= self.start or epoch.start >= self.stop:
                    continue
                selected = epoch.index == self.selected_epoch
                span = ax.axvspan(epoch.start, epoch.stop, facecolor=COLORS[record['state']],
                                 edgecolor='#d28b0b' if selected else 'none', linewidth=2,
                                 alpha=.5, zorder=0)
                center = (max(epoch.start, self.start) + min(epoch.stop, self.stop)) / 2
                warning = ' · REVISAR' if self.screening_issues(epoch.index) else ''
                label = ax.text(center, .98, f'E{epoch.index + 1} · {record["state"]}{warning}',
                                transform=ax.get_xaxis_transform(), ha='center', va='top',
                                fontsize=8, color='#784f00' if selected else '#62767c')
                self.epoch_artists.extend((span, label))
        self.canvas.draw_idle()

    def resize_plot(self, event=None):
        width = max(200, self.plot_view.winfo_width() - 2)
        count = max(1, len(self.figure.axes))
        height = max(self.plot_view.winfo_height(), 190 * count if count > 1 else 80)
        if self.stacked and self.displayed is not None and self.stack_page.get() != 'Todos':
            per_screen = int(self.stack_page.get())
            height = max(height, int(self.plot_view.winfo_height() * len(self.displayed.channels) / per_screen))
        self.plot_view.itemconfigure(self.plot_window, width=width, height=height)
        self.plot_view.configure(scrollregion=(0, 0, width, height))

    @staticmethod
    def padded_limits(values):
        """y limits with 15 % padding, like ax.margins(y=.15) on the data alone."""
        low, high = float(np.min(values)), float(np.max(values))
        pad = .15 * (high - low) if high > low else max(abs(high), 1.0) * .15
        return low - pad, high + pad

    def draw_stacked(self, segment, rebuild=False):
        """Every channel in one axes, offset by a common µV scale (like EEGLAB's eegplot).

        Each trace is centred on its median, so DC offsets do not hide amplitude differences;
        a flat or very noisy electrode stands out at a glance. Clicking a channel name marks
        it as bad; clicking inside the plot still selects the época."""
        previous = self.displayed
        self.displayed = segment
        names = segment.channels
        spacing = self.stack_scale
        offsets = spacing * np.arange(len(names))[::-1]
        values = segment.values_uv - np.median(segment.values_uv, axis=1, keepdims=True)
        lines = getattr(self, 'segment_lines', [])
        reuse = (not rebuild and previous is not None and previous.channels == names
                 and len(lines) == len(names) and len(self.figure.axes) == 1
                 and getattr(self, 'stack_drawn', None) == (names, spacing))
        if reuse:
            for line, row, offset in zip(lines, values, offsets):
                line.set_data(segment.times, row + offset)
            self.ax.set_xlim(segment.start, segment.stop)
        else:
            self.figure.clear()
            self.epoch_artists = []
            self.segment_lines = []
            ax = self.ax = self.figure.add_subplot(111)
            for row, offset in zip(values, offsets):
                line, = ax.plot(segment.times, row + offset, color=TRACE, linewidth=.6)
                line.get_path().should_simplify = False
                self.segment_lines.append(line)
            ax.set_xlim(segment.start, segment.stop)
            ax.set_ylim(-1.5 * spacing, offsets[0] + 1.5 * spacing)
            ax.set_yticks(offsets)
            ax.set_yticklabels(names, fontsize=7)
            for label in ax.get_yticklabels():
                label.set_picker(True)
            ax.set_xlabel('Tiempo desde el inicio (s)')
            ax.set_ylabel(f'{spacing:.3g} µV entre trazas')
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.grid(axis='x', color='#bdccce', linewidth=.8)
            for spine in ax.spines.values():
                spine.set_color('#d9e1e2')
            self.stack_drawn = (names, spacing)
            self.resize_plot()
        self.stack_offsets = offsets
        self.style_stacked(draw=False)
        self.reset_toolbar()
        self.toolbar.push_current()
        self.draw_overlays()

    def style_stacked(self, draw=True):
        """Grey traces and red names for bad channels (manual marks and the last detection)."""
        if not self.stacked or self.displayed is None or len(self.figure.axes) != 1:
            return
        bad = self.bad_channel_set() if hasattr(self, 'bad_channel_set') else set()
        manual = set(self.store.bad_channels) if self.store else set()
        for line, name in zip(self.segment_lines, self.displayed.channels):
            line.set_color(BAD_TRACE if name in bad else TRACE)
        for label in self.ax.get_yticklabels():
            name = label.get_text()
            label.set_color(BAD_LABEL if name in bad else TEXT)
            label.set_fontweight('bold' if name in manual else 'normal')
        if draw:
            self.canvas.draw_idle()

    def stacked_pick(self, event):
        """Click on a channel name in the stacked view: mark / unmark it as bad."""
        if not self.stacked or self.busy or self.toolbar.mode:
            return
        name = getattr(event.artist, 'get_text', lambda: '')()
        if self.displayed is not None and name in self.displayed.channels:
            self.toggle_bad_channel(name)

    def stacked_hover(self, event):
        """Name and value of the trace under the cursor in the stacked view."""
        if (not self.stacked or self.displayed is None or event.inaxes is not self.ax
                or event.xdata is None or event.ydata is None):
            return
        offsets = getattr(self, 'stack_offsets', None)
        if offsets is None or not len(offsets):
            return
        index = int(np.clip(np.round((offsets[0] - event.ydata) / self.stack_scale), 0, len(offsets) - 1))
        segment = self.displayed
        sample = int(np.clip(np.searchsorted(segment.times, event.xdata), 0, len(segment.times) - 1))
        value = segment.values_uv[index, sample] - np.median(segment.values_uv[index])
        self.cursor_text.set(f'{segment.channels[index]} · {event.xdata:.2f} s · {value:+.1f} µV')

    def draw_segment(self, segment):
        if self.stacked and len(segment.channels) > 1:
            self.draw_stacked(segment)
            return
        previous = self.displayed
        self.displayed = segment
        lines = getattr(self, 'segment_lines', [])
        if (previous is not None and previous.channels == segment.channels
                and len(lines) == len(segment.channels) and len(self.figure.axes) == len(lines)):
            # Same electrodes: update the existing lines instead of rebuilding the figure,
            # which skips the subplot creation and most of the constrained-layout work.
            for ax, line, values in zip(self.figure.axes, lines, segment.values_uv):
                line.set_data(segment.times, values)
                ax.set_xlim(segment.start, segment.stop)
                ax.set_ylim(*self.padded_limits(values))
            self.reset_toolbar()
            self.toolbar.push_current()
            self.draw_overlays()
            return
        self.figure.clear()
        self.epoch_artists = []
        self.segment_lines = []
        axes = self.figure.subplots(len(segment.channels), 1, sharex=True, squeeze=False).ravel()
        self.ax = axes[0]
        for ax, channel, values in zip(axes, segment.channels, segment.values_uv):
            line, = ax.plot(segment.times, values, color='#0e7c86', linewidth=.8)
            line.get_path().should_simplify = False
            self.segment_lines.append(line)
            ax.set_xlim(segment.start, segment.stop)
            ax.set_ylim(*self.padded_limits(values))
            ax.set_ylabel(f'{channel}\nµV')
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.grid(axis='x', color='#bdccce', linewidth=.8)
            ax.grid(axis='y', color='#edf1f1', linewidth=.5)
            for spine in ax.spines.values():
                spine.set_color('#d9e1e2')
        axes[-1].set_xlabel('Tiempo desde el inicio (s)')
        self.resize_plot()
        self.reset_toolbar()
        self.toolbar.push_current()
        self.draw_overlays()
