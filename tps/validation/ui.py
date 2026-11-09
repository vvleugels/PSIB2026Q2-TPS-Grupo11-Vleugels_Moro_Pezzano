"""Standalone Tk validation workbench; production GUI deliberately stays unchanged."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.signal import welch

from tps.validation.engine import RuleProfile
from tps.validation.session import ValidationSession
from tps.validation.store import export_new


def indices(text, maximum):
    """Human-facing indices are one-based, stored indices are zero-based."""
    result = sorted({int(s.strip()) - 1 for s in text.split(',') if s.strip()})
    if any(i < 0 or i >= maximum for i in result):
        raise ValueError(f'Ingresar números entre 1 y {maximum}, separados por comas.')
    return result


class ValidationApp:
    def __init__(self, root, revision):
        self.root, self.revision = root, revision
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='eeg-validation')
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.busy, self.closed = False, False
        self.controls = []
        self.epoch = tk.IntVar(value=1)
        self.channel = tk.StringVar(value=revision.data['source']['channels'][0])
        self.final = tk.BooleanVar(value=False)
        self.note = tk.StringVar()
        self.status = tk.StringVar(value='Preparando copias; los originales no se modifican…')
        self.persistent = tk.StringVar(value=','.join(revision.data['persistent']))
        self.global_epochs = tk.StringVar(value=','.join(str(i + 1) for i in revision.data['excluded_epochs']))
        self.components = tk.StringVar(value=','.join(str(i + 1) for i in revision.data['excluded_components']))
        self.repairs = tk.StringVar(value=','.join(str(i + 1) for i in revision.data['confirmed_repairs']))
        root.title('Validación EEG · Perfil experimental · Sin aceptación automática')
        root.geometry('1250x850')
        root.minsize(950, 650)
        header = ttk.Frame(root, padding=8)
        header.pack(fill='x')
        source = revision.data['source']
        ttk.Label(header, text=f"{source['participant']} · {source['session']} · {source['task']} · "
                  f"{source['split']} · épocas de 4 s").pack(side='left')
        self.button(header, 'Exportar evaluaciones', self.export).pack(side='right')
        notebook = ttk.Notebook(root)
        notebook.pack(fill='both', expand=True)
        review = ttk.Frame(notebook, padding=8)
        processing = ttk.Frame(notebook, padding=8)
        notebook.add(review, text='Revisión de señales')
        notebook.add(processing, text='Calidad, ICA y reparaciones')
        nav = ttk.Frame(review)
        nav.pack(fill='x')
        self.button(nav, '←', lambda: self.move(-1)).pack(side='left')
        self.button(nav, '→', lambda: self.move(1)).pack(side='left')
        ttk.Label(nav, text='Época').pack(side='left', padx=5)
        self.entry(nav, self.epoch, 6).pack(side='left')
        self.button(nav, 'Ir', self.draw).pack(side='left')
        cb = ttk.Combobox(nav, textvariable=self.channel, values=source['channels'], state='readonly', width=12)
        cb.pack(side='left', padx=8)
        cb.bind('<<ComboboxSelected>>', lambda e: self.draw())
        self.controls.append((cb, 'readonly'))
        toggle = ttk.Checkbutton(nav, text='Revisión final (señal limpia)', variable=self.final, command=self.draw)
        toggle.pack(side='left')
        self.controls.append((toggle, 'normal'))
        marks = ttk.Frame(review)
        marks.pack(fill='x', pady=6)
        for caption, state in [('Fisiológica', 'physiological'), ('Artefacto', 'artifact'), ('Sin revisar', 'unreviewed')]:
            self.button(marks, caption, lambda s=state: self.mark(s, False)).pack(side='left', padx=2)
        self.button(marks, 'Toda la época fisiológica', lambda: self.mark('physiological', True)).pack(side='left', padx=8)
        self.button(marks, 'Cerrar revisión ciega', self.complete).pack(side='right')
        notes = ttk.Frame(review)
        notes.pack(fill='x')
        ttk.Label(notes, text='Nota').pack(side='left')
        self.entry(notes, self.note, 65).pack(side='left', padx=8)
        self.review_status = tk.StringVar()
        ttk.Label(review, textvariable=self.review_status, wraplength=1100).pack(fill='x', pady=5)
        self.figure = Figure(figsize=(10, 5), layout='constrained')
        self.canvas = FigureCanvasTkAgg(self.figure, master=review)
        self.canvases = [self.canvas]
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        NavigationToolbar2Tk(self.canvas, review)
        ttk.Label(processing, text='Las sugerencias se revelan al cerrar la revisión ciega. '
                  'Ninguna exclusión o reparación se aplica sin confirmación.', wraplength=1100).pack(anchor='w')
        for label, variable in [('Electrodos persistentes confirmados (nombres)', self.persistent),
                                ('Épocas globalmente inutilizables (números)', self.global_epochs),
                                ('Componentes a excluir (números)', self.components),
                                ('Épocas con reparación local confirmada (números)', self.repairs)]:
            row = ttk.Frame(processing)
            row.pack(fill='x', pady=3)
            ttk.Label(row, text=label, width=54).pack(side='left')
            self.entry(row, variable, 55).pack(side='left', fill='x', expand=True)
        row = ttk.Frame(processing)
        row.pack(fill='x', pady=5)
        self.button(row, 'Ver calidad', self.quality).pack(side='left')
        self.button(row, 'Perfil de calibración', self.edit_profile).pack(side='left', padx=3)
        self.button(row, 'Confirmar y ajustar ICA', self.fit).pack(side='left', padx=3)
        self.button(row, 'Ver componentes', self.inspect_components).pack(side='left', padx=3)
        self.button(row, 'Proponer reparaciones', self.propose).pack(side='left', padx=3)
        self.button(row, 'Confirmar limpieza', self.clean).pack(side='left', padx=3)
        self.button(processing, 'Exportar informe y prueba de interpolación', self.report).pack(anchor='w')
        self.details = tk.Text(processing, wrap='word', height=22, state='disabled')
        self.details.pack(fill='both', expand=True, pady=8)
        footer = ttk.Frame(root, padding=8)
        footer.pack(fill='x')
        ttk.Label(footer, textvariable=self.status, wraplength=1050).pack(side='left')
        ttk.Button(footer, text='Cancelar cálculo', command=self.request_cancel).pack(side='right')
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.poll_id = root.after(100, self.poll)
        self.submit(lambda: ValidationSession(revision), self.loaded)

    def button(self, parent, text, command):
        b = ttk.Button(parent, text=text, command=lambda: self.guard(command))
        self.controls.append((b, 'normal'))
        return b

    def entry(self, parent, variable, width):
        e = ttk.Entry(parent, textvariable=variable, width=width)
        self.controls.append((e, 'normal'))
        return e

    def guard(self, callback):
        if self.busy:
            return
        try:
            callback()
        except Exception as exc:
            self.status.set(str(exc))
            messagebox.showerror('Validación EEG', str(exc), parent=self.root)

    def submit(self, work, done):
        self.busy = True
        self.cancel.clear()
        for widget, _ in self.controls:
            widget.configure(state='disabled')
        self.status.set('Procesando en segundo plano…')
        future = self.executor.submit(work)
        future.add_done_callback(lambda f: self.events.put((f, done)))

    def poll(self):
        try:
            future, done = self.events.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            for widget, state in self.controls:
                widget.configure(state=state)
            try:
                result = future.result()
                if self.closed:
                    if isinstance(result, ValidationSession):
                        result.close()
                else:
                    done(result)
            except Exception as exc:
                self.status.set(str(exc))
            if self.closed:
                self.finish_close()
                return
        self.poll_id = self.root.after(100, self.poll)

    def loaded(self, session):
        self.session = session
        session.cancel_event = self.cancel
        self.status.set('Revisión lista. Las decisiones se guardan automáticamente.')
        self.draw()

    def request_cancel(self):
        if self.busy:
            self.cancel.set()
            self.status.set('Cancelación solicitada; se descartará el cálculo antes de guardar.')

    def current_epoch(self):
        index = self.epoch.get() - 1
        if self.session is None or not 0 <= index < len(self.session.prepared.quality.epochs):
            raise ValueError('Número de época fuera del registro.')
        return self.session.prepared.quality.epochs[index]

    def move(self, delta):
        self.epoch.set(max(1, min(len(self.session.prepared.quality.epochs), self.epoch.get() + delta)))
        self.draw()

    def draw(self):
        if self.session is None:
            return
        try:
            e = self.current_epoch()
        except (ValueError, tk.TclError) as exc:
            self.status.set(str(exc))
            return
        c = self.channel.get()
        if self.final.get() and self.session.cleaned is None:
            self.final.set(False)
            self.status.set('Todavía no existe una señal limpia confirmada.')
        labels = self.revision.data['final_labels' if self.final.get() else 'labels']
        record = labels.get(f'{e.index}:{c}', {})
        self.note.set(record.get('note', ''))
        completed = sum(v['state'] in ('physiological', 'artifact') for v in labels.values())
        self.review_status.set(f"{c} · época {e.index + 1} · {record.get('state', 'unreviewed')} · "
                               f"{completed} evaluaciones por canal guardadas · "
                               + ('Parcial: fuera de ICA y espectros' if e.partial else 'Época completa'))
        self.figure.clear()
        temporal, spectral = self.figure.subplots(2, 1)
        raw = self.session.recording.raw
        start = (e.index // 3) * 12
        first = round(start * raw.info['sfreq'])
        last = min(first + round(12 * raw.info['sfreq']), raw.n_times)
        sources = [('Original', raw), ('Filtrada 0,2–45 Hz', self.session.prepared.analysis)]
        if self.final.get():
            comparable = self.session.prepared.analysis.copy().set_eeg_reference(
                self.revision.data['reference_channels'], verbose=False)
            sources[1] = ('Filtrada · misma referencia promedio', comparable)
            sources.append(('Limpia · referencia promedio', self.session.cleaned))
        for label, signal in sources:
            values = signal.get_data(picks=[c], start=first, stop=last)[0]
            temporal.plot(np.arange(first, last) / raw.info['sfreq'], values * 1e6, label=label, linewidth=.7)
            if not e.partial:
                values = signal.get_data(picks=[c], start=e.first, stop=e.last)[0]
                f, p = welch(values, fs=raw.info['sfreq'], nperseg=round(2 * raw.info['sfreq']))
                spectral.plot(f, 10 * np.log10(np.maximum(p * 1e12, np.finfo(float).tiny)), label=label)
        temporal.axvspan(e.start, e.stop, color='orange', alpha=.15)
        for t in np.arange(start, start + 13, 4):
            temporal.axvline(t, color='gray', alpha=.3)
        temporal.set(xlabel='Tiempo (s)', ylabel='µV', xlim=(first / raw.info['sfreq'], last / raw.info['sfreq']))
        temporal.legend(loc='upper right')
        spectral.set(xlabel='Hz', ylabel='dB (µV²/Hz)', xlim=(0, 45))
        self.canvas.draw_idle()

    def mark(self, state, all_channels):
        e = self.current_epoch()
        channels = self.revision.data['source']['channels'] if all_channels else [self.channel.get()]
        self.revision.label(e.index, channels, state, self.note.get(), self.final.get())
        self.draw()

    def complete(self):
        if messagebox.askyesno('Cerrar revisión ciega', 'Las etiquetas iniciales quedarán bloqueadas y se '
                              'revelarán las sugerencias. ¿Confirmar?', parent=self.root):
            self.revision.complete_blind()
            self.quality()

    def show(self, text):
        self.details.configure(state='normal')
        self.details.delete('1.0', 'end')
        self.details.insert('1.0', text)
        self.details.configure(state='disabled')

    def require_blind(self):
        if not self.revision.data['blind_complete']:
            raise ValueError('Completar la revisión ciega antes de revelar sugerencias.')

    def quality(self):
        self.require_blind()
        q = self.session.prepared.quality
        lines = ['Perfil experimental: ' + str(asdict(self.revision.profile)),
                 'Electrodos persistentes sugeridos: ' + ', '.join(q.persistent)]
        for e in q.epochs:
            entries = [f'{c}: {", ".join(q.reasons(e.index, c))}' for j, c in enumerate(q.channels) if q.flags[e.index, j]]
            if entries:
                lines.append(f'Época {e.index + 1}: ' + '; '.join(entries))
        lines.append('Indicadores no evaluables: ' + str(int((~q.evaluable).sum())))
        self.show('\n'.join(lines))

    def fit(self):
        self.require_blind()
        persistent = [s.strip() for s in self.persistent.get().split(',') if s.strip()]
        excluded = indices(self.global_epochs.get(), len(self.session.prepared.quality.epochs))
        if messagebox.askyesno('Confirmar', f'Canales persistentes: {persistent}\nEpochs excluidas: '
                              f'{[i + 1 for i in excluded]}\n¿Ajustar ICA con estas decisiones?', parent=self.root):
            self.submit(lambda: self.session.fit(persistent, excluded),
                        lambda _: self.status.set('ICA guardada. Inspeccionar y confirmar componentes.'))

    def inspect_components(self):
        self.require_blind()
        fitted = self.session.fitted
        if fitted is None:
            raise ValueError('Ajustar ICA primero.')
        window = tk.Toplevel(self.root)
        window.title('Componentes ICA · sugerencias, no exclusiones')
        selected = tk.IntVar(value=1)
        picker = ttk.Spinbox(window, from_=1, to=fitted.ica.n_components_, textvariable=selected, width=8)
        picker.pack()
        figure = Figure(figsize=(10, 7), layout='constrained')
        canvas = FigureCanvasTkAgg(figure, window)
        self.canvases.append(canvas)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        sources = fitted.ica.get_sources(self.session.prepared.analysis).get_data()
        def render():
            try:
                k = selected.get() - 1
                if not 0 <= k < fitted.ica.n_components_:
                    return
                figure.clear()
                axes = figure.subplots(2, 2)
                try:
                    fitted.ica.plot_components(picks=[k], axes=axes[0, 0], show=False, colorbar=False)
                except (ValueError, RuntimeError) as exc:
                    axes[0, 0].text(.05, .5, 'Mapa no disponible:\n' + str(exc), wrap=True)
                axes[0, 1].plot(np.arange(sources.shape[1]) / self.session.recording.sfreq, sources[k], linewidth=.5)
                axes[0, 1].set(xlabel='Tiempo (s)', title='Señal del componente')
                f, p = welch(sources[k], fs=self.session.recording.sfreq,
                             nperseg=round(2 * self.session.recording.sfreq))
                axes[1, 0].semilogy(f, np.maximum(p, np.finfo(float).tiny))
                axes[1, 0].set(xlim=(0, 45), xlabel='Hz', title='Espectro del componente')
                axes[1, 1].axis('off')
                axes[1, 1].text(0, .9, '\n'.join(f'{key}: {value}' for key, value in
                                               fitted.components[k].items()), va='top', wrap=True)
                canvas.draw_idle()
            except (ValueError, tk.TclError) as exc:
                self.status.set(str(exc))
        picker.configure(command=render)
        picker.bind('<Return>', lambda e: render())
        render()

    def propose(self):
        self.require_blind()
        if self.session.fitted is None:
            raise ValueError('Ajustar ICA primero.')
        components = indices(self.components.get(), self.session.fitted.ica.n_components_)
        def display(proposals):
            self.show('\n'.join(f'Época {p.epoch + 1}: {", ".join(p.channels)} · '
                                f'{p.total_affected} afectados · ' + ('reparable' if p.eligible else 'supera límite')
                                for p in proposals) or 'No se proponen reparaciones locales.')
        self.submit(lambda: self.session.repair_proposals(components), display)

    def clean(self):
        self.require_blind()
        if self.session.fitted is None:
            raise ValueError('Ajustar ICA primero.')
        components = indices(self.components.get(), self.session.fitted.ica.n_components_)
        repairs = indices(self.repairs.get(), len(self.session.prepared.quality.epochs))
        if messagebox.askyesno('Confirmar limpieza', f'Excluir componentes: {[i + 1 for i in components]}\n'
                              f'Reparar épocas: {[i + 1 for i in repairs]}\nSe invalidará la revisión final '
                              'anterior. ¿Confirmar conjuntamente?', parent=self.root):
            def done(result):
                self.final.set(True)
                self.draw()
                self.status.set('Limpieza guardada. Revisar señal final; no hay aceptación automática.')
                self.show('Referencia fija: ' + ', '.join(result.reference_channels))
            self.submit(lambda: self.session.clean(components, repairs), done)

    def edit_profile(self):
        self.require_blind()
        if self.revision.data['frozen']:
            raise ValueError('Perfil congelado: no se recalibra sobre evaluación.')
        window = tk.Toplevel(self.root)
        window.title('Perfil provisional de calibración')
        variables = {}
        for field, value in asdict(self.revision.profile).items():
            row = ttk.Frame(window)
            row.pack(fill='x', padx=8, pady=2)
            ttk.Label(row, text=field, width=26).pack(side='left')
            variables[field] = tk.StringVar(value=str(value))
            ttk.Entry(row, textvariable=variables[field]).pack(side='left')
        def save():
            profile = {k: int(v.get()) if k == 'max_repair' else float(v.get()) for k, v in variables.items()}
            RuleProfile(**profile).validate()
            self.revision.update_processing(profile=profile, excluded_components=[], confirmed_repairs=[])
            self.session.close()
            window.destroy()
            self.submit(lambda: ValidationSession(self.revision), self.loaded)
        ttk.Button(window, text='Guardar e invalidar procesamiento anterior',
                   command=lambda: self.guard(save)).pack(pady=8)

    def export(self):
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension='.json',
                                          initialfile='evaluaciones.json')
        if path:
            self.revision.export(path)
            self.status.set('Evaluaciones exportadas.')

    def report(self):
        self.require_blind()
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension='.json', initialfile='informe.json')
        if path:
            def work():
                report = self.session.report()
                report['interpolation_holdouts'] = self.session.holdouts()
                return report
            def done(report):
                if self.cancel.is_set():
                    self.status.set('Exportación cancelada.')
                    return
                export_new(path, report)
                self.status.set('Informe exportado. La aprobación fisiológica sigue siendo humana.')
            self.submit(work, done)

    def close(self):
        self.closed = True
        if self.busy:
            self.request_cancel()
            self.status.set('Cerrando al finalizar el cálculo en curso…')
        else:
            self.finish_close()

    def finish_close(self):
        # Cancel Matplotlib idle draws as well as polling before destroying Tcl commands.
        self.root.after_cancel(self.poll_id)
        for canvas in self.canvases:
            if canvas._idle_draw_id is not None and canvas.get_tk_widget().winfo_exists():
                canvas.get_tk_widget().after_cancel(canvas._idle_draw_id)
                canvas._idle_draw_id = None
        if self.session:
            self.session.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def launch(revision):
    root = tk.Tk()
    ValidationApp(root, revision)
    root.mainloop()
