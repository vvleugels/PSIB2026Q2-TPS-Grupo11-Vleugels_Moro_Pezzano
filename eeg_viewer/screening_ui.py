"""Amplitude screening with automatic acceptance of epochs within limits."""
import tkinter as tk
from tkinter import ttk

from .screening import ScreeningConfig, scan_recording
from .processing import FilterConfig
from .theme import autowrap


class ScreeningUI:
    def init_screening(self):
        self.screening_result = None
        self.screening_pending = False
        self.screening_high = tk.StringVar(value='150')
        self.screening_flat = tk.StringVar(value='1')
        self.screening_all = tk.BooleanVar(value=True)
        self.screening_use_filter = tk.BooleanVar(value=True)
        self.screening_signal = tk.StringVar(value='Original · Filtrado desactivado')
        self.screening_summary = tk.StringVar(value='Sin barrido. Acepta las epochs sin avisos.')
        self.screening_detail = tk.StringVar(value='')
        for variable in (self.screening_high, self.screening_flat, self.screening_all,
                         self.screening_use_filter):
            variable.trace_add('write', self.clear_screening)

    def build_screening(self, parent):
        card = self.card(parent, 'POSIBLES ARTEFACTOS')
        intro = ttk.Label(card, style='CardMuted.TLabel',
                          text='Pico a pico por epoch de 5 s. Sin avisos: aceptación y guardado '
                               'automáticos. Conserva las decisiones manuales.')
        intro.pack(fill='x', pady=(2, 6))
        autowrap(intro)
        self.screening_filter_button = ttk.Checkbutton(
            card, text='Usar el filtro activo', variable=self.screening_use_filter,
            style='Card.TCheckbutton')
        self.screening_filter_button.pack(anchor='w')
        ttk.Label(card, textvariable=self.screening_signal, style='CardMuted.TLabel',
                  wraplength=280).pack(anchor='w')
        row = ttk.Frame(card, style='Card.TFrame')
        row.pack(fill='x', pady=6)
        self.screening_entries = []
        for label, variable in (('Alta >', self.screening_high), ('Plana <', self.screening_flat)):
            ttk.Label(row, text=label, style='Card.TLabel').pack(side='left', padx=(0, 4))
            entry = ttk.Entry(row, textvariable=variable, width=6)
            entry.pack(side='left', padx=(0, 8))
            self.screening_entries.append(entry)
        ttk.Label(row, text='µV', style='Card.TLabel').pack(side='left')
        self.screening_all_button = ttk.Checkbutton(
            card, text='Todos los electrodos EEG', variable=self.screening_all,
            style='Card.TCheckbutton')
        self.screening_all_button.pack(anchor='w')
        ttk.Label(card, text='Desactivado: solo los electrodos visibles.',
                  style='CardMuted.TLabel', wraplength=280).pack(anchor='w')
        self.screening_button = ttk.Button(card, text='Buscar posibles artefactos',
                                           style='Accent.TButton', command=self.start_screening)
        self.screening_button.pack(fill='x', pady=8)
        row = ttk.Frame(card, style='Card.TFrame')
        row.pack(fill='x')
        self.screening_nav = []
        for text, direction in (('← Sospechosa', -1), ('Sospechosa →', 1)):
            button = ttk.Button(row, text=text, style='Tool.TButton',
                                command=lambda step=direction: self.next_suspect(step))
            button.pack(side='left', expand=True, fill='x', padx=(0, 4) if direction < 0 else 0)
            self.screening_nav.append(button)
        summary = ttk.Label(card, textvariable=self.screening_summary, style='CardMuted.TLabel')
        summary.pack(fill='x', anchor='w', pady=(6, 0))
        autowrap(summary)

    def clear_screening(self, *args):
        if self.screening_pending:
            self.invalidate()
        self.screening_result = None
        self.screening_summary.set('Sin barrido. Acepta las epochs sin avisos.')
        self.screening_detail.set('')
        self.update_screening_signal()
        if hasattr(self, 'epoch_tree'):
            self.refresh_epochs()
            self.draw_overlays()
            self.update_controls()

    def update_screening_signal(self):
        filtering = self.config if self.screening_use_filter.get() else FilterConfig()
        self.screening_signal.set(filtering.label if filtering.active else
                                  ('Original · Filtrado desactivado' if self.screening_use_filter.get()
                                   else 'Original · Sin filtrar'))
        if self.recording and self.recording.ica_context:
            self.screening_signal.set(self.screening_signal.get().replace('Original', 'Sin pasabanda/notch') +
                                      ' · ICA ' + self.recording.ica_context['revision'][:8])

    def start_screening(self):
        if self.busy or not self.recording or not self.visible_channels() or not self.flush_autosave():
            return
        try:
            config = ScreeningConfig(float(self.screening_high.get().replace(',', '.')),
                                     float(self.screening_flat.get().replace(',', '.')))
            config.validate()
            channels = tuple(self.recording.channels if self.screening_all.get()
                             else self.visible_channels())
            if not channels:
                raise ValueError('Elegí al menos un electrodo o activá todos los EEG.')
        except ValueError as exc:
            self.screening_summary.set(str(exc))
            return
        self.clear_screening()
        self.invalidate()
        recording = self.recording
        filtering = self.config if self.screening_use_filter.get() else FilterConfig()
        self.screening_pending = True
        self.screening_summary.set(f'Buscando: {filtering.label}…')

        def done(result):
            self.screening_pending = False
            self.screening_result = result
            names = ', '.join(result['channels']) if len(result['channels']) <= 4 else f'{len(channels)} electrodos EEG'
            self.screening_summary.set(
                f"{len(result['findings'])}/{result['total']} epochs con avisos · {names}\n"
                f'{result["filter"].label} · Alta > {config.high_uv:g} µV · Plana < {config.flat_uv:g} µV'
                + (f" · {result['partial']} parcial" if result['partial'] else '')
                + (' · ICA ' + recording.ica_context['revision'][:8] if recording.ica_context else '')
                + '\nCon avisos: revisión manual.')
            try:
                counts = self.store.apply_screening(result)
            except ValueError as exc:
                self.screening_summary.set(self.screening_summary.get()
                                           + f'\nNo se pudieron actualizar las marcas: {exc}')
                self.status.set(str(exc))
            else:
                saved = True
                if counts['accepted'] or counts['pending']:
                    self.invalidate_spectrum('El barrido actualizó las marcas. Volvé a calcular.')
                    saved = self.save_annotations()
                self.screening_summary.set(self.screening_summary.get()
                                           + f"\n{counts['accepted']} aceptadas automáticamente"
                                           + f" · {counts['pending']} devueltas a sin revisar."
                                           + ('' if saved else '\nGuardado pendiente: reintentá el guardado.'))
            self.refresh_epochs()
            self.load_editor()
            self.draw_overlays()

        def failed(exc):
            self.screening_pending = False
            self.screening_summary.set(f'No se pudo completar el barrido: {exc}')

        self.submit(lambda: scan_recording(recording, channels, config, filtering), done, on_error=failed)

    def screening_issues(self, index):
        return self.screening_result['findings'].get(index, []) if self.screening_result else []

    def screening_badge(self, index):
        kinds = {issue['kind'] for issue in self.screening_issues(index)}
        return next(iter(kinds)) if len(kinds) == 1 else ('Varios' if kinds else '')

    def update_screening_detail(self):
        batch = self.batch_rejection(self.selected_epoch) if (
            hasattr(self, 'batch_rejection') and self.selected_epoch is not None) else ''
        batch = f'Rechazada por el lote: {batch}' if batch else ''
        if not self.screening_result or self.selected_epoch is None:
            self.screening_detail.set(batch)
            return
        issues = self.screening_issues(self.selected_epoch)
        details = [f"{issue['channel']}: {issue['kind'].lower()}"
                   + (f" ({issue['peak_to_peak_uv']:.2f} µV pico a pico)"
                      if issue['peak_to_peak_uv'] is not None else '') for issue in issues]
        signal = self.screening_result['filter'].label
        text = (f'{signal}: ' + '; '.join(details) if details else
                f'{signal}: sin avisos con estos umbrales y electrodos; requiere revisión visual.')
        self.screening_detail.set(text + (f'\n{batch}' if batch else ''))

    def next_suspect(self, direction):
        if self.busy or not self.screening_result:
            return
        indices = sorted(self.screening_result['findings'])
        if not indices:
            return
        current = self.selected_epoch if self.selected_epoch is not None else -1
        candidates = [index for index in indices if index > current] if direction > 0 else [
            index for index in indices if index < current]
        index = (candidates[0] if candidates else indices[0]) if direction > 0 else (
            candidates[-1] if candidates else indices[-1])
        self.select_epoch(index, navigate=True)
        self.view_tabs.select(0)

    def update_screening_controls(self):
        enabled = self.recording is not None and bool(self.visible_channels()) and not self.busy
        self.screening_button.configure(state='normal' if enabled else 'disabled')
        for widget in self.screening_entries + [self.screening_all_button, self.screening_filter_button]:
            widget.configure(state='normal' if not self.busy else 'disabled')
        has_suspects = enabled and self.screening_result and self.screening_result['findings']
        for button in self.screening_nav:
            button.configure(state='normal' if has_suspects else 'disabled')
