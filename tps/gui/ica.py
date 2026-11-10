"""Manual ICA review in the main explorer; fitting runs in a disposable process."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tkinter as tk
from tkinter import ttk, messagebox

import mne
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import welch

from tps.core import ica
from tps.core.processing import ContinuousFilter
from tps.gui.theme import IRIS, IRIS_LIGHT, MUTED, autowrap, card, style_dialog


class ICAUI:
    def build_ica(self, parent):
        self.ica_job = None
        self.ica_temp = None
        self.ica_dialog = None
        self.ica_maps = None
        self.ica_after = None
        self.ica_model = None
        self.ica_metadata = None
        # Lives inside the «Posibles artefactos» card, under the amplitude search.
        section = parent
        ttk.Separator(section, orient='horizontal').pack(fill='x', pady=(10, 6))
        ttk.Label(section, text='Corrección por ICA (opcional)', style='CardBold.TLabel').pack(anchor='w')
        intro = ttk.Label(section, style='CardMuted.TLabel',
                          text='Separa la señal en componentes para quitar parpadeos u otros artefactos '
                               'sin descartar épocas.')
        intro.pack(fill='x', pady=(0, 4))
        autowrap(intro)
        self.ica_status = tk.StringVar(value='Sin corrección ICA')
        ttk.Label(section, textvariable=self.ica_status, style='Card.TLabel', wraplength=270).pack(fill='x')
        self.ica_button = ttk.Button(section, text='Ajustar / revisar ICA', command=self.open_ica)
        self.ica_button.pack(fill='x', pady=4)
        self.ica_enabled = tk.BooleanVar(value=False)
        self.ica_toggle = ttk.Checkbutton(section, text='Usar ICA confirmada', variable=self.ica_enabled,
                                         style='Card.TCheckbutton', command=self.toggle_ica)
        self.ica_toggle.pack(anchor='w')

    def refresh_ica(self):
        if not hasattr(self, 'ica_button'):
            return
        recording = self.recording
        active = recording is not None and recording.ica_context is not None
        self.ica_enabled.set(active)
        self.ica_button.configure(state='normal' if recording and not self.busy else 'disabled')
        self.ica_toggle.configure(state='normal' if recording and not self.busy else 'disabled')
        if active:
            value = recording.ica_context
            text = 'ICA ' + value['revision'][:8] + ' · excluidos: ' + ', '.join(str(i+1) for i in value['excluded_components'])
            if value['excluded_channels']:
                text += '\nSin corregir: ' + ', '.join(value['excluded_channels'])
        else:
            text = 'ICA desactivada'
        self.ica_status.set(text)
        if recording:
            label = self.config.label if self.config.active or not active else 'Sin pasabanda/notch'
            self.signal_label.set(label + (' · ICA ' + recording.ica_context['revision'][:8] if active else ''))

    def cancel_ica(self, close=False):
        if getattr(self, 'ica_after', None):
            self.root.after_cancel(self.ica_after)
            self.ica_after = None
        if getattr(self, 'ica_job', None):
            self.ica_job.kill()
            self.ica_job.wait()
            self.ica_job = None
        if getattr(self, 'ica_temp', None):
            self.ica_temp.cleanup()
            self.ica_temp = None
        if close:
            if getattr(self, 'ica_maps', None) is not None and self.ica_maps.winfo_exists():
                self.ica_maps.destroy()
            self.ica_maps = None
            if getattr(self, 'ica_dialog', None):
                self.ica_dialog.destroy()
                self.ica_dialog = None
            self.ica_model = self.ica_metadata = None
        elif getattr(self, 'ica_dialog', None):
            self.ica_message.set('Ajuste cancelado; se conserva la corrección confirmada.')
            self.refresh_ica_dialog()

    def open_ica(self):
        """Guided ICA window: 1 fit, 2 review components, 3 choose what to remove, 4 compare and confirm.

        Steps unlock in order; the plot on the right shows the inspected component or the
        before/after comparison. Nothing is removed until «Confirmar y aplicar»."""
        if self.busy or not self.recording or not self.flush_autosave():
            return
        if self.ica_dialog:
            self.ica_dialog.lift()
            return
        dialog = self.ica_dialog = tk.Toplevel(self.root)
        dialog.title('Corrección por ICA · ' + self.recording.study.path.stem)
        dialog.geometry('1180x820')
        dialog.minsize(980, 620)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.protocol('WM_DELETE_WINDOW', lambda: self.cancel_ica(close=True) if not self.busy else None)
        style_dialog(dialog)
        body = ttk.Frame(dialog, padding=12)
        body.pack(fill='both', expand=True)
        side = ttk.Frame(body, width=340)
        side.pack(side='left', fill='y', padx=(0, 12))
        side.pack_propagate(False)
        self.ica_steps = {}

        def step(number, title, text):
            outer, inner = card(side, f'{number}  {title}')
            outer.pack(fill='x', pady=(0, 8))
            if text:
                label = ttk.Label(inner, text=text, style='CardMuted.TLabel')
                label.pack(fill='x', pady=(2, 6))
                autowrap(label)
            self.ica_steps[number] = inner
            return inner

        # 1 · Fit
        fit = step(1, 'Ajustar', 'Infomax, 1–45 Hz, épocas de 5 s. Tarda hasta un par de minutos.')
        row = self.fit_row = ttk.Frame(fit, style='Card.TFrame')
        row.pack(fill='x')
        self.fit_button = ttk.Button(row, text='Ajustar ICA', style='Accent.TButton', command=self.start_ica)
        self.fit_button.pack(side='left', fill='x', expand=True)
        self.fit_cancel = ttk.Button(row, text='Cancelar', command=self.cancel_ica, state='disabled')
        self.fit_cancel.pack(side='left', padx=(6, 0))
        self.fit_progress = ttk.Progressbar(fit, mode='indeterminate')
        self.fit_channels = tk.StringVar()
        self.fit_epochs = tk.StringVar()
        advanced = ttk.Frame(fit, style='Card.TFrame')
        toggle = ttk.Button(fit, text='Opciones avanzadas ▸', style='CardLink.TButton')

        def show_advanced():
            if advanced.winfo_manager():
                advanced.pack_forget(); toggle.configure(text='Opciones avanzadas ▸')
            else:
                advanced.pack(fill='x', after=toggle); toggle.configure(text='Opciones avanzadas ▾')
        toggle.configure(command=show_advanced)
        toggle.pack(anchor='w', pady=(4, 0))
        for label, var in [('Canales fuera del ajuste (nombres, separados por coma)', self.fit_channels),
                           ('Épocas fuera del ajuste (números desde 1, separados por coma)', self.fit_epochs)]:
            text = ttk.Label(advanced, text=label, style='CardMuted.TLabel')
            text.pack(fill='x')
            autowrap(text)
            ttk.Entry(advanced, textvariable=var).pack(fill='x', pady=(0, 4))
        note = ttk.Label(advanced, style='CardMuted.TLabel',
                         text='Solo afectan el ajuste, no las marcas de revisión. Conservá ejemplos de parpadeos.')
        note.pack(fill='x')
        autowrap(note)

        # 2 · Review components
        review = step(2, 'Revisar componentes', 'Los parpadeos suelen ser frontales y de baja frecuencia.')
        row = ttk.Frame(review, style='Card.TFrame')
        row.pack(fill='x')
        self.component_number = tk.StringVar(value='1')
        self.ica_prev = ttk.Button(row, text='◀', width=3, style='Tool.TButton', command=lambda: self.step_component(-1))
        self.ica_prev.pack(side='left')
        self.component_picker = ttk.Spinbox(row, from_=1, to=1, textvariable=self.component_number, width=4,
                                            command=self.inspect_ica)
        self.component_picker.pack(side='left', padx=(8, 0))
        self.component_picker.bind('<Return>', lambda e: self.inspect_ica())
        self.ica_total = ttk.Label(row, text='de —', style='CardMuted.TLabel')
        self.ica_total.pack(side='left', padx=(4, 8))
        self.ica_next = ttk.Button(row, text='▶', width=3, style='Tool.TButton', command=lambda: self.step_component(1))
        self.ica_next.pack(side='left')
        self.ica_maps_button = ttk.Button(review, text='Ver mapas de todos los componentes', command=self.open_ica_maps)
        self.ica_maps_button.pack(fill='x', pady=(6, 0))

        # 3 · Choose what to remove
        choose = step(3, 'Elegir qué quitar', 'Marcá los que son artefacto (puede ser ninguno).')
        self.exclusions_text = tk.StringVar(value='')
        self.ica_exclude_current = tk.BooleanVar(value=False)
        self.ica_exclude_button = ttk.Checkbutton(choose, text='Quitar el componente 1', style='Card.TCheckbutton',
                                                  variable=self.ica_exclude_current, command=self.toggle_current_exclusion)
        self.ica_exclude_button.pack(anchor='w')
        row = ttk.Frame(choose, style='Card.TFrame')
        row.pack(fill='x', pady=(4, 0))
        ttk.Label(row, text='A quitar:', style='Card.TLabel').pack(side='left')
        self.ica_exclusions_entry = ttk.Entry(row, textvariable=self.exclusions_text)
        self.ica_exclusions_entry.pack(side='left', fill='x', expand=True, padx=(6, 0))
        self.exclusions_text.trace_add('write', lambda *args: self.refresh_ica_dialog())

        # 4 · Compare and confirm
        apply = step(4, 'Comparar y confirmar', 'Nada se aplica hasta «Confirmar y aplicar».')
        self.ica_preview_button = ttk.Button(apply, text='Ver antes / después', command=self.preview_ica)
        self.ica_preview_button.pack(fill='x')
        self.ica_confirm_button = ttk.Button(apply, text='Confirmar y aplicar', style='Accent.TButton',
                                             command=self.confirm_ica)
        self.ica_confirm_button.pack(fill='x', pady=(6, 0))

        # Messages and plot on the right.
        main = ttk.Frame(body)
        main.pack(side='left', fill='both', expand=True)
        note = tk.Frame(main, background=IRIS_LIGHT)
        note.pack(fill='x', pady=(0, 8))
        tk.Frame(note, background=IRIS, width=4).pack(side='left', fill='y')
        self.ica_message = tk.StringVar(value='Paso 1: presioná «Ajustar ICA». Ningún componente se quita automáticamente.')
        message = ttk.Label(note, textvariable=self.ica_message, style='Info.TLabel', padding=(10, 8))
        message.pack(side='left', fill='x', expand=True)
        message.bind('<Configure>', lambda e: message.configure(wraplength=max(e.width - 24, 200)))
        self.ica_figure = Figure(figsize=(8, 5), dpi=100, layout='constrained')
        self.ica_canvas = FigureCanvasTkAgg(self.ica_figure, master=main)
        self.ica_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.ica_placeholder('Ajustá la ICA para ver aquí cada componente.')
        try:
            folder, metadata = ica.read_revision(self.annotations_dir, self.recording.study)
            ica.validate_revision(self.recording, self.data_dir, metadata)
            self.ica_model = mne.preprocessing.read_ica(folder / 'model-ica.fif', verbose=False)
            self.ica_metadata = metadata
            self.fit_channels.set(','.join(metadata['excluded_channels']))
            self.fit_epochs.set(','.join(str(i+1) for i in metadata['excluded_epochs']))
            self.exclusions_text.set(','.join(str(i+1) for i in metadata['excluded_components']))
            self.component_picker.configure(to=self.ica_model.n_components_)
            self.ica_message.set('Se cargó la corrección confirmada. Podés revisar los componentes, cambiar qué se '
                                 'quita y volver a confirmar, o ajustar de nuevo.')
        except (OSError, ValueError, KeyError) as exc:
            self.ica_model = self.ica_metadata = None
            if not isinstance(exc, ValueError) or 'falta una ICA confirmada' not in str(exc):
                self.ica_message.set(str(exc))
        self.refresh_ica_dialog()
        if self.ica_model is not None:
            dialog.after_idle(self.inspect_ica)

    def ica_placeholder(self, text):
        self.ica_figure.clear()
        ax = self.ica_figure.add_subplot(111)
        ax.set_axis_off()
        ax.text(.5, .5, text, ha='center', va='center', color=MUTED, fontsize=12, transform=ax.transAxes)
        self.ica_canvas.draw_idle()

    def refresh_ica_dialog(self):
        """Unlock the steps in order and keep the «Quitar este componente» box in sync."""
        if not getattr(self, 'ica_dialog', None):
            return
        fitting = self.ica_job is not None
        ready = self.ica_model is not None and not fitting
        self.fit_button.configure(state='disabled' if fitting else 'normal',
                                  text='Ajustar de nuevo' if self.ica_model is not None else 'Ajustar ICA')
        self.fit_cancel.configure(state='normal' if fitting else 'disabled')
        if fitting and not self.fit_progress.winfo_manager():
            self.fit_progress.pack(fill='x', pady=(6, 0), after=self.fit_row)
            self.fit_progress.start(12)
        elif not fitting and self.fit_progress.winfo_manager():
            self.fit_progress.stop()
            self.fit_progress.pack_forget()
        for widget in (self.ica_prev, self.ica_next, self.component_picker, self.ica_maps_button,
                       self.ica_exclude_button, self.ica_exclusions_entry, self.ica_preview_button,
                       self.ica_confirm_button):
            widget.configure(state='normal' if ready else 'disabled')
        self.ica_total.configure(text=f'de {self.ica_model.n_components_}' if self.ica_model is not None else 'de —')
        try:
            k = int(self.component_number.get())
            chosen = {i + 1 for i in self.ica_indices(self.exclusions_text.get())}
        except ValueError:
            return
        self.ica_exclude_button.configure(text=f'Quitar el componente {k}')
        self.ica_exclude_current.set(k in chosen)

    def step_component(self, delta):
        if self.ica_model is None:
            return
        try:
            k = int(self.component_number.get())
        except ValueError:
            k = 1
        self.component_number.set(str(min(self.ica_model.n_components_, max(1, k + delta))))
        self.inspect_ica()

    def toggle_current_exclusion(self):
        try:
            k = int(self.component_number.get())
            chosen = {i + 1 for i in self.ica_indices(self.exclusions_text.get())}
        except ValueError as exc:
            self.ica_message.set(str(exc))
            return
        chosen = chosen | {k} if self.ica_exclude_current.get() else chosen - {k}
        self.exclusions_text.set(','.join(str(i) for i in sorted(chosen)))
        if self.ica_maps is not None and self.ica_maps.winfo_exists() and hasattr(self.ica_maps, 'refresh'):
            self.ica_maps.refresh()

    @staticmethod
    def ica_indices(text):
        values = [int(x.strip()) - 1 for x in text.split(',') if x.strip()]
        if any(i < 0 for i in values):
            raise ValueError('Los números comienzan en 1.')
        return sorted(set(values))

    def start_ica(self):
        if self.busy or self.ica_job:
            return
        try:
            channels = [x.strip() for x in self.fit_channels.get().split(',') if x.strip()]
            epochs = self.ica_indices(self.fit_epochs.get())
            if not set(channels) <= set(self.recording.channels):
                raise ValueError('Hay nombres de canales desconocidos.')
            study = self.recording.study
            self.ica_temp = tempfile.TemporaryDirectory(prefix='tps-ica-')
            folder = Path(self.ica_temp.name)
            # Preserve the BIDS symlink: EEGLAB resolves its .fdt beside the .set.
            # resolve() moves the header into annex's separate object directory.
            ica.write_json(folder / 'request.json', dict(path=str(study.path.absolute()),
                           participant=study.participant, session=study.session, task=study.task,
                           data_dir=str(self.data_dir.resolve()), channels=channels, epochs=epochs))
            self.ica_job = subprocess.Popen([sys.executable, '-m', 'tps.pipeline.ica_worker', str(folder)],
                                            cwd=Path(__file__).resolve().parents[2],
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.ica_message.set('Ajustando ICA en otro proceso… Podés cancelar el ajuste.')
            self.ica_after = self.root.after(150, self.poll_ica)
            self.refresh_ica_dialog()
        except (OSError, ValueError) as exc:
            if self.ica_temp is not None and self.ica_job is None:
                self.ica_temp.cleanup(); self.ica_temp = None
            self.ica_message.set(str(exc))

    def poll_ica(self):
        self.ica_after = None
        if not self.ica_job:
            return
        if self.ica_job.poll() is None:
            self.ica_after = self.root.after(150, self.poll_ica)
            return
        folder = Path(self.ica_temp.name)
        try:
            if self.ica_job.returncode:
                raise ValueError(json.loads((folder / 'error.json').read_text())['error']
                                 if (folder / 'error.json').exists() else 'El proceso ICA terminó sin resultado.')
            metadata = json.loads((folder / 'fit.json').read_text())
            ica.validate_revision(self.recording, self.data_dir, metadata)
            model = mne.preprocessing.read_ica(folder / 'draft-ica.fif', verbose=False)
            self.ica_model, self.ica_metadata = model, metadata
            self.exclusions_text.set('')
            self.component_picker.configure(to=model.n_components_)
            self.component_number.set('1')
            self.ica_message.set(f'Listo: {model.n_components_} componentes. Paso 2: recorrelos con ◀ ▶ o abrí '
                                 'los mapas; en el paso 3 marcá los que son artefacto.')
            self.root.after_idle(self.inspect_ica)
        except (OSError, ValueError, KeyError) as exc:
            self.ica_message.set(str(exc))
        finally:
            self.ica_job = None
            self.ica_temp.cleanup(); self.ica_temp = None
            self.refresh_ica_dialog()

    def checked_ica(self):
        if self.ica_model is None or self.ica_job:
            raise ValueError('Primero completá el ajuste ICA.')
        channels = [x.strip() for x in self.fit_channels.get().split(',') if x.strip()]
        if (set(channels) != set(self.ica_metadata['excluded_channels']) or
                self.ica_indices(self.fit_epochs.get()) != self.ica_metadata['excluded_epochs']):
            raise ValueError('Cambió la selección de entrenamiento; volvé a ajustar ICA.')
        excluded = self.ica_indices(self.exclusions_text.get())
        if any(i >= self.ica_model.n_components_ for i in excluded):
            raise ValueError('Número de componente fuera de rango.')
        return excluded

    def open_ica_maps(self):
        if self.busy:
            return
        try:
            self.checked_ica()
            if self.ica_maps is not None and self.ica_maps.winfo_exists():
                self.ica_maps.lift()
                return
            from .ica_maps import ICAMapGrid
            def inspect(index):
                self.component_number.set(str(index+1))
                self.inspect_ica()
            self.ica_maps = ICAMapGrid(self.ica_dialog, self.ica_model, self.exclusions_text, inspect)
        except ValueError as exc:
            self.ica_message.set(str(exc))

    def inspect_ica(self):
        if self.busy:
            return
        try:
            self.checked_ica()
            k = int(self.component_number.get()) - 1
            if not 0 <= k < self.ica_model.n_components_:
                raise ValueError('Componente fuera de rango.')
            model, recording = self.ica_model, self.recording
            start = round(self.start * recording.sfreq)
            stop = min(recording.raw.n_times, start + round(10 * recording.sfreq))
            def work():
                raw = recording.raw.copy().crop(tmin=start / recording.sfreq, tmax=(stop-1) / recording.sfreq).load_data()
                try:
                    return model.get_sources(raw).get_data(picks=[k])[0]
                finally:
                    raw.close()
            def done(source):
                self.ica_figure.clear()
                # Wide, short time series on top (like the explorer); spectrum and map below, so no
                # panel is a tall narrow strip that is hard to read.
                grid = self.ica_figure.add_gridspec(2, 2, height_ratios=(1, 1.35), width_ratios=(1.5, 1))
                axes = [self.ica_figure.add_subplot(grid[0, :]), self.ica_figure.add_subplot(grid[1, 0]),
                        self.ica_figure.add_subplot(grid[1, 1])]
                axes[0].plot(np.arange(len(source))/recording.sfreq + start/recording.sfreq, source,
                             linewidth=.7, color='#0e7c86')
                axes[0].set(xlabel='Tiempo (s)', ylabel='Amplitud (u. a.)', title=f'Componente {k+1} · señal')
                axes[0].grid(alpha=.25)
                f, p = welch(source, recording.sfreq, nperseg=min(len(source), round(2*recording.sfreq)))
                axes[1].semilogy(f, np.maximum(p, np.finfo(float).tiny), color='#0e7c86')
                axes[1].set(xlim=(0, 60), xlabel='Frecuencia (Hz)', ylabel='Potencia (u. a.)', title='Espectro (PSD)')
                axes[1].grid(alpha=.25)
                try:
                    from .ica_maps import map_info
                    info, geometry = map_info(model)
                    values = model.get_components()[:, k]
                    maximum = np.max(np.abs(values))
                    mne.viz.plot_topomap(values / maximum if maximum else values, info, axes=axes[2],
                                         show=False, cmap='RdBu_r', vlim=(-1, 1), contours=6)
                    axes[2].set_title('Mapa topográfico')
                    axes[2].text(.5, -.1, '10–20 aproximado' if 'APROXIMADAS' in geometry else 'Geometría del modelo',
                                 transform=axes[2].transAxes, ha='center', fontsize=8)
                except (ValueError, RuntimeError):
                    axes[2].clear(); axes[2].text(.5, .5, 'Mapa no disponible\nFalta geometría', ha='center')
                    axes[2].axis('off')
                indicators = self.ica_metadata['components'][k]
                def number(value):
                    return 'no evaluable' if value is None else f'{value:.3g}'
                correlations = ', '.join(name + ': ' + number(value)
                                         for name, value in indicators['frontal_correlations'].items())
                labels = {'curtosis': 'curtosis alta', 'potencia_rapida': 'potencia rápida alta',
                          'correlacion_frontal_no_EOG': 'correlación frontal (no EOG)'}
                suggestions = ', '.join(labels.get(reason, reason) for reason in indicators['suggestions'])
                self.ica_message.set(
                    f"Componente {k+1} · Curtosis: {number(indicators['kurtosis'])} · "
                    f"Proporción de potencia rápida: {number(indicators['fast_ratio'])} · "
                    f"Correlación frontal: {correlations or 'no disponible'}. "
                    f"Indicadores: {suggestions or 'sin avisos'}; no son una clasificación automática.")
                self.ica_canvas.draw_idle()
                self.refresh_ica_dialog()
            self.submit(work, done, on_error=lambda exc: self.ica_message.set(str(exc)))
        except (ValueError, TypeError) as exc:
            self.ica_message.set(str(exc))

    def preview_ica(self):
        if self.busy:
            return
        try:
            excluded = self.checked_ica()
            channels = self.visible_channels()[:8] or [self.recording.channels[0]]
            recording, config, model = self.recording, self.config, self.ica_model
            start = round(self.start * recording.sfreq)
            stop = min(recording.raw.n_times, start + round(10 * recording.sfreq))
            def work():
                corrected = ica.corrected_copy(recording, model, excluded)
                try:
                    before, after = ContinuousFilter(recording.raw), ContinuousFilter(corrected)
                    return [(before.channel(c, config)[start:stop]*1e6,
                             after.channel(c, config)[start:stop]*1e6) for c in channels]
                finally:
                    corrected.close()
            def done(rows):
                self.ica_figure.clear()
                axes = np.atleast_1d(self.ica_figure.subplots(len(channels), 1, sharex=True))
                times = np.arange(start, stop)/recording.sfreq
                for ax, name, (before, after) in zip(axes, channels, rows):
                    ax.plot(times, before, label='Antes', alpha=.7)
                    ax.plot(times, after, label='Después', alpha=.8)
                    ax.set_ylabel(name + ' · µV'); ax.legend(loc='upper right')
                axes[-1].set_xlabel('Tiempo (s)')
                self.ica_message.set('Vista previa (todavía no aplicada): azul antes, naranja después, '
                                     f'con {config.label.lower()}. Si te convence, presioná «Confirmar y aplicar».')
                self.ica_canvas.draw_idle()
            self.submit(work, done, on_error=lambda exc: self.ica_message.set(str(exc)))
        except ValueError as exc:
            self.ica_message.set(str(exc))

    def confirm_ica(self):
        if self.busy or not self.flush_autosave():
            return
        try:
            excluded = self.checked_ica()
            if not messagebox.askyesno('Confirmar ICA',
                'Excluir componentes: ' + (', '.join(str(i+1) for i in excluded) or 'ninguno') +
                '\n¿Guardar esta corrección y aplicarla?', parent=self.ica_dialog):
                return
            recording, model, metadata = self.recording, self.ica_model, self.ica_metadata
            def work():
                corrected, value = ica.confirm(recording, self.data_dir, self.annotations_dir, model, metadata, excluded)
                ica.activate(recording, corrected, value)
                return value
            def done(value):
                self.ica_metadata = value
                self.ica_message.set('Corrección confirmada, guardada y aplicada. Podés cerrar esta ventana; '
                                     '«Usar ICA confirmada» en el panel lateral la activa o desactiva.')
                self.ica_changed()
            self.submit(work, done, on_error=lambda exc: self.ica_message.set(str(exc)))
        except ValueError as exc:
            self.ica_message.set(str(exc))

    def ica_changed(self):
        self.clear_screening()
        self.invalidate_spectrum('Cambió ICA. Volvé a calcular el espectro.')
        self.refresh_ica()
        self.update_screening_signal()
        if self.store and self.selected_epoch is not None:
            self.update_review_context()
        if self.visible_channels():
            self.show_window(self.start)

    def toggle_ica(self):
        if self.busy or not self.recording or not self.flush_autosave():
            self.refresh_ica()
            return
        enabled, recording = self.ica_enabled.get(), self.recording
        def work():
            root = ica.revision_root(self.annotations_dir, recording.study)
            pointer = json.loads((root / 'latest.json').read_text())
            corrected, metadata = None, None
            if enabled:
                folder, metadata = ica.read_revision(self.annotations_dir, recording.study, pointer['revision'])
                ica.validate_revision(recording, self.data_dir, metadata)
                model = mne.preprocessing.read_ica(folder / 'model-ica.fif', verbose=False)
                corrected = ica.corrected_copy(recording, model, metadata['excluded_components'])
            try:
                ica.write_json(root / 'latest.json', {**pointer, 'enabled': enabled})
            except Exception:
                if corrected is not None:
                    corrected.close()
                raise
            ica.activate(recording, corrected, metadata)
        self.submit(work, lambda _: self.ica_changed(), on_error=lambda exc: (
            self.status.set(f'No se pudo cambiar ICA: {exc}'), self.refresh_ica()))
