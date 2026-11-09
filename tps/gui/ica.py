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


class ICAUI:
    def build_ica(self, parent):
        self.ica_job = None
        self.ica_temp = None
        self.ica_dialog = None
        self.ica_after = None
        self.ica_model = None
        self.ica_metadata = None
        section = self.card(parent, 'ICA')
        self.ica_status = tk.StringVar(value='Sin corrección ICA')
        ttk.Label(section, textvariable=self.ica_status, wraplength=270).pack(fill='x')
        self.ica_button = ttk.Button(section, text='Ajustar / revisar ICA', command=self.open_ica)
        self.ica_button.pack(fill='x', pady=4)
        self.ica_enabled = tk.BooleanVar(value=False)
        self.ica_toggle = ttk.Checkbutton(section, text='Usar ICA confirmada', variable=self.ica_enabled,
                                         command=self.toggle_ica)
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
            if getattr(self, 'ica_dialog', None):
                self.ica_dialog.destroy()
                self.ica_dialog = None
            self.ica_model = self.ica_metadata = None
        elif getattr(self, 'ica_dialog', None):
            self.ica_message.set('Ajuste cancelado; se conserva la corrección confirmada.')
            self.fit_button.configure(state='normal')

    def open_ica(self):
        if self.busy or not self.recording or not self.flush_autosave():
            return
        if self.ica_dialog:
            self.ica_dialog.lift()
            return
        dialog = self.ica_dialog = tk.Toplevel(self.root)
        dialog.title('ICA · revisión manual de componentes')
        dialog.geometry('1050x780')
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.protocol('WM_DELETE_WINDOW', lambda: self.cancel_ica(close=True) if not self.busy else None)
        body = ttk.Frame(dialog, padding=12)
        body.pack(fill='both', expand=True)
        ttk.Label(body, text='Entrenamiento: 1–45 Hz · epochs de 5 s · Infomax. Conservá ejemplos de parpadeos.\n'
                  'Estas exclusiones son solo para el ajuste; no cambian las marcas de revisión.').pack(anchor='w')
        self.fit_channels = tk.StringVar()
        self.fit_epochs = tk.StringVar()
        for label, var in [('Canales fuera del ajuste (nombres separados por coma)', self.fit_channels),
                           ('Epochs fuera del ajuste (números desde 1, separados por coma)', self.fit_epochs)]:
            ttk.Label(body, text=label).pack(anchor='w')
            ttk.Entry(body, textvariable=var).pack(fill='x')
        row = ttk.Frame(body); row.pack(fill='x', pady=6)
        self.fit_button = ttk.Button(row, text='Ajustar ICA', command=self.start_ica)
        self.fit_button.pack(side='left')
        ttk.Button(row, text='Cancelar ajuste', command=self.cancel_ica).pack(side='left', padx=5)
        self.ica_message = tk.StringVar(value='Ningún componente se elimina automáticamente.')
        ttk.Label(body, textvariable=self.ica_message, wraplength=990).pack(fill='x')
        row = ttk.Frame(body); row.pack(fill='x', pady=5)
        self.component_number = tk.StringVar(value='1')
        ttk.Label(row, text='Componente').pack(side='left')
        self.component_picker = ttk.Spinbox(row, from_=1, to=1, textvariable=self.component_number, width=6)
        self.component_picker.pack(side='left', padx=5)
        ttk.Button(row, text='Inspeccionar', command=self.inspect_ica).pack(side='left')
        self.exclusions_text = tk.StringVar(value='')
        ttk.Label(row, text='Excluir componentes (1,2,…)').pack(side='left', padx=8)
        ttk.Entry(row, textvariable=self.exclusions_text, width=20).pack(side='left')
        ttk.Button(row, text='Antes / después', command=self.preview_ica).pack(side='left', padx=5)
        ttk.Button(row, text='Confirmar y aplicar', command=self.confirm_ica).pack(side='left')
        self.ica_figure = Figure(figsize=(10, 4), dpi=100, layout='constrained')
        self.ica_canvas = FigureCanvasTkAgg(self.ica_figure, master=body)
        self.ica_canvas.get_tk_widget().pack(fill='both', expand=True)
        try:
            folder, metadata = ica.read_revision(self.annotations_dir, self.recording.study)
            ica.validate_revision(self.recording, self.data_dir, metadata)
            self.ica_model = mne.preprocessing.read_ica(folder / 'model-ica.fif', verbose=False)
            self.ica_metadata = metadata
            self.fit_channels.set(','.join(metadata['excluded_channels']))
            self.fit_epochs.set(','.join(str(i+1) for i in metadata['excluded_epochs']))
            self.exclusions_text.set(','.join(str(i+1) for i in metadata['excluded_components']))
            self.component_picker.configure(to=self.ica_model.n_components_)
            self.ica_message.set('Revisión confirmada cargada. Podés inspeccionar o cambiar las exclusiones.')
        except (OSError, ValueError, KeyError) as exc:
            self.ica_model = self.ica_metadata = None
            self.ica_message.set(str(exc))

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
            ica.write_json(folder / 'request.json', dict(path=str(study.path.resolve()),
                           participant=study.participant, session=study.session, task=study.task,
                           data_dir=str(self.data_dir.resolve()), channels=channels, epochs=epochs))
            self.ica_job = subprocess.Popen([sys.executable, '-m', 'tps.pipeline.ica_worker', str(folder)],
                                            cwd=Path(__file__).resolve().parents[2],
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.fit_button.configure(state='disabled')
            self.ica_message.set('Ajustando ICA en otro proceso… Podés cancelar el ajuste.')
            self.ica_after = self.root.after(150, self.poll_ica)
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
            self.ica_message.set(f'{model.n_components_} componentes. Inspeccionalos antes de confirmar exclusiones.')
        except (OSError, ValueError, KeyError) as exc:
            self.ica_message.set(str(exc))
        finally:
            self.ica_job = None
            self.ica_temp.cleanup(); self.ica_temp = None
            self.fit_button.configure(state='normal')

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
                axes = self.ica_figure.subplots(1, 3)
                axes[0].plot(np.arange(len(source))/recording.sfreq + start/recording.sfreq, source)
                axes[0].set(xlabel='Tiempo (s)', title=f'Componente {k+1}')
                f, p = welch(source, recording.sfreq, nperseg=min(len(source), round(2*recording.sfreq)))
                axes[1].semilogy(f, np.maximum(p, np.finfo(float).tiny)); axes[1].set(xlim=(0, 60), title='PSD del componente')
                try:
                    model.plot_components(picks=[k], axes=axes[2], show=False, colorbar=False)
                    axes[2].set_title(f'Componente {k+1}')
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
            self.submit(work, done, on_error=lambda exc: self.ica_message.set(str(exc)))
        except (ValueError, TypeError) as exc:
            self.ica_message.set(str(exc))

    def preview_ica(self):
        if self.busy:
            return
        try:
            excluded = self.checked_ica()
            channels = self.visible_channels() or [self.recording.channels[0]]
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
                self.ica_message.set('Vista previa; todavía no aplicada. ' + config.label)
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
                self.ica_message.set('Corrección confirmada y guardada. Las marcas previas conservan su contexto anterior.')
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
