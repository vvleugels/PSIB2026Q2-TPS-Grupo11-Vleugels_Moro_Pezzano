"""Epoch review controls and optional multichannel display for the desktop app."""
import tkinter as tk
from tkinter import ttk

from matplotlib.ticker import MultipleLocator

from .annotations import STATES, REASONS
from .processing import FilterConfig
from .theme import BORDER, IRIS, TEXT, autowrap, style_dialog

COLORS = {'Sin revisar': '#edf1f2', 'Aceptada': '#d5eee2', 'Con artefactos': '#f9d8d5'}


class ReviewUI:
    def init_review(self):
        self.config = FilterConfig()
        self.store = None
        self.selected_epoch = None
        self.displayed = None
        self.active_patient = ''
        self.active_study_index = -1
        self.multi = tk.BooleanVar(value=False)
        self.signal_label = tk.StringVar(value='Original')
        self.save_label = tk.StringVar(value='Sin estudio')
        self.count_label = tk.StringVar(value='')
        self.epoch_label = tk.StringVar(value='Seleccioná una epoch')
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

    def build_review_sidebar(self, parent, selection):
        self.multi_toggle = ttk.Checkbutton(selection, text='Varios electrodos (hasta 8)',
                                            variable=self.multi, style='Card.TCheckbutton',
                                            command=self.toggle_multi)
        self.multi_toggle.pack(anchor='w', pady=(8, 0))
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
        self.build_screening(parent)
        epochs = self.card(parent, 'EPOCHS · 5 SEGUNDOS', expand=True)
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
        editor = ttk.Frame(chart, style='Card.TFrame')
        editor.pack(side='bottom', fill='x', pady=(8, 0))
        ttk.Label(editor, textvariable=self.epoch_label, style='CardBold.TLabel').pack(anchor='w')
        row = ttk.Frame(editor, style='Card.TFrame')
        row.pack(fill='x', pady=3)
        self.mark_buttons = []
        for label, key in zip(STATES, ('U', 'A', 'R')):
            button = ttk.Radiobutton(row, text=f'{label} ({key})', value=label,
                                     variable=self.mark_state, style='Card.TRadiobutton')
            button.pack(side='left', padx=(0, 10))
            self.mark_buttons.append(button)
        row = ttk.Frame(editor, style='Card.TFrame')
        row.pack(fill='x', pady=3)
        self.reason_box = ttk.Combobox(row, textvariable=self.reason, values=REASONS,
                                       state='disabled', width=20)
        self.reason_box.pack(side='left')
        ttk.Label(row, text='Nota', style='Card.TLabel').pack(side='left', padx=(10, 6))
        self.note_entry = ttk.Entry(row, textvariable=self.note, validate='key',
                                    validatecommand=(self.root.register(lambda value: len(value) <= 500), '%P'))
        self.note_entry.pack(side='left', fill='x', expand=True)
        for variable in (self.screening_detail, self.context_label):
            label = ttk.Label(editor, textvariable=variable, style='CardMuted.TLabel')
            label.pack(fill='x', anchor='w')
            autowrap(label)
        for key, state in zip(('u', 'a', 'r'), STATES):
            self.root.bind(f'<KeyPress-{key}>', lambda event, s=state: self.shortcut(event, s))
            self.root.bind(f'<KeyPress-{key.upper()}>', lambda event, s=state: self.shortcut(event, s))

    def configuration_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title('Configuración de filtrado')
        dialog.transient(self.root)
        dialog.resizable(False, False)
        style_dialog(dialog)
        body = ttk.Frame(dialog, padding=20)
        body.pack(fill='both')
        enabled = tk.BooleanVar(value=self.config.enabled)
        low = tk.StringVar(value=f'{self.config.low:g}')
        high = tk.StringVar(value=f'{self.config.high:g}')
        ttk.Checkbutton(body, text='Aplicar filtrado', variable=enabled).pack(anchor='w', pady=6)
        for label, variable in (('Corte inferior (Hz)', low), ('Corte superior (Hz)', high)):
            ttk.Label(body, text=label).pack(anchor='w')
            ttk.Entry(body, textvariable=variable).pack(fill='x', pady=(0, 8))
        ttk.Label(body, text='FIR · Fase cero · Ventana Hamming\nSe aplica sobre el canal continuo completo.',
                  style='Muted.TLabel').pack(anchor='w', pady=5)
        error = tk.StringVar()
        ttk.Label(body, textvariable=error, foreground='#a12b25', wraplength=420).pack(fill='x')
        ttk.Button(body, text='Restablecer valores del paper',
                   command=lambda: (low.set('0.2'), high.set('45'))).pack(fill='x', pady=8)
        buttons = ttk.Frame(body)
        buttons.pack(fill='x')
        def apply():
            try:
                candidate = FilterConfig(enabled.get(), float(low.get().replace(',', '.')),
                                         float(high.get().replace(',', '.')))
                candidate.validate(self.recording.sfreq)
            except (ValueError, AttributeError) as exc:
                error.set(str(exc))
                return
            dialog.destroy()
            self.apply_config(candidate)
        ttk.Button(buttons, text='Aplicar', style='Accent.TButton', command=apply).pack(side='right')
        ttk.Button(buttons, text='Cancelar', command=dialog.destroy).pack(side='right', padx=8)
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
            if changed and self.screening_use_filter.get():
                self.clear_screening()
            self.update_screening_signal()
            self.signal_label.set(candidate.label)
            if segment is not None:
                self.segment_loaded(segment)
            else:
                self.status.set(f'Configuración aplicada: {candidate.label}')
        self.submit(work, done, on_error=lambda exc: self.status.set(
            f'No se aplicó el filtro: {exc} Se conserva {self.config.label.lower()}.'))

    def visible_channels(self):
        if self.multi.get():
            return [self.channel_list.get(i) for i in self.channel_list.curselection()]
        return [self.channel.get()] if self.channel.get() else []

    def toggle_multi(self):
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
        self.epoch_label.set('Seleccioná una epoch')
        self.context_label.set('Sin evaluación previa')
        self.note.set('')
        self.reason.set('')
        self.clear_screening()

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

    def refresh_epochs(self):
        if not self.store:
            return
        counts = {s: 0 for s in STATES}
        for epoch, record in zip(self.store.epochs, self.store.records):
            state = record['state']
            counts[state] += 1
            badge = self.screening_badge(epoch.index)
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
        self.epoch_label.set(f'Epoch {epoch.index + 1} · {epoch.start:g}–{epoch.stop:g} s'
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
        if context:
            config = FilterConfig(context['enabled'], context['low'], context['high'])
            screening = record.get('screening')
            label = ('Aceptación automática' if record['state'] == STATES[1] else 'Barrido automático') if screening else 'Última evaluación'
            limits = (f" · Pico a pico {screening['flat_uv']:g}–{screening['high_uv']:g} µV"
                      if screening else '')
            self.context_label.set(f"{label}: {config.label}{limits} · {', '.join(record['channels'])}"
                                   f" · {record['reviewed_at'][:19]} UTC")
        else:
            self.context_label.set('Sin evaluación previa')

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
        self.plot_view.itemconfigure(self.plot_window, width=width, height=height)
        self.plot_view.configure(scrollregion=(0, 0, width, height))

    def draw_segment(self, segment):
        self.displayed = segment
        self.figure.clear()
        self.epoch_artists = []
        axes = self.figure.subplots(len(segment.channels), 1, sharex=True, squeeze=False).ravel()
        self.ax = axes[0]
        for ax, channel, values in zip(axes, segment.channels, segment.values_uv):
            line, = ax.plot(segment.times, values, color='#0e7c86', linewidth=.8)
            line.get_path().should_simplify = False
            ax.set_xlim(segment.start, segment.stop)
            ax.margins(y=.15)
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
