"""Interfaz Tkinter; las lecturas se ejecutan en un único hilo de trabajo."""
from concurrent.futures import ThreadPoolExecutor
import queue
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from .data import discover, open_recording

BG = '#f2f5f5'
INK = '#0f2e36'
TEAL = '#0e7c86'
AMBER = '#f5a524'
MUTED = '#62767c'


class Toolbar(NavigationToolbar2Tk):
    toolitems = (
        ('Restaurar', 'Restaurar la ventana temporal', 'home', 'home'),
        ('Atrás', 'Volver a la vista anterior', 'back', 'back'),
        ('Adelante', 'Avanzar a la vista siguiente', 'forward', 'forward'),
        (None, None, None, None),
        ('Desplazar', 'Arrastrar para desplazar la vista', 'move', 'pan'),
        ('Zoom', 'Seleccionar un rectángulo para acercar', 'zoom_to_rect', 'zoom'),
    )


class App:
    def __init__(self, root, data_dir):
        self.root = root
        self.data_dir = data_dir
        self.inventory = {}
        self.studies = []
        self.recording = None
        self.worker_recording = None
        self.start = 0.0
        self.stop = 0.0
        self.generation = 0
        self.future = None
        self.events = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='eeg-reader')
        self.closed = False
        self.busy = False
        self.patient = tk.StringVar()
        self.study = tk.StringVar()
        self.channel = tk.StringVar()
        self.start_text = tk.StringVar(value='0')
        self.status = tk.StringVar(value='Buscando participantes…')
        self.window_text = tk.StringVar(value='Ventanas de 10 segundos')
        root.title('PSIB — Explorador EEG')
        root.geometry('1280x800')
        root.minsize(960, 640)
        root.configure(bg=BG)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.build()
        self.clear_plot('Elegí un paciente para comenzar')
        self.submit(lambda: discover(data_dir), self.inventory_loaded)
        self.poll_id = root.after(50, self.poll)

    def build(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', font=('DejaVu Sans', 10), background=BG, foreground=INK)
        style.configure('Card.TFrame', background='white')
        style.configure('Card.TLabel', background='white')
        style.configure('Muted.TLabel', background='white', foreground=MUTED)
        style.configure('TButton', padding=(10, 7))
        style.configure('TCombobox', padding=5)
        style.map('TCombobox', fieldbackground=[('readonly', 'white')])
        bar = tk.Frame(self.root, bg=INK, padx=20, pady=16)
        bar.pack(fill='x')
        tk.Label(bar, text='PSIB — Explorador EEG', font=('DejaVu Sans', 18, 'bold'),
                 bg=INK, fg='white').pack(side='left')
        tk.Label(bar, text='●  Señal sin procesar', bg=INK, fg='#b6e2df',
                 font=('DejaVu Sans', 10)).pack(side='right')
        tk.Frame(self.root, height=3, bg=AMBER).pack(fill='x')
        bottom = ttk.Frame(self.root, padding=(16, 10))
        bottom.pack(side='bottom', fill='x')
        self.status_label = ttk.Label(bottom, textvariable=self.status, wraplength=900)
        self.status_label.pack(fill='x')
        self.panes = tk.PanedWindow(self.root, orient='horizontal', bg=BG,
                                   sashwidth=10, borderwidth=0)
        self.panes.pack(fill='both', expand=True, padx=14, pady=14)
        sidebar = ttk.Frame(self.panes, width=320)
        main = ttk.Frame(self.panes)
        self.panes.add(sidebar, minsize=300, stretch='never')
        self.panes.add(main, minsize=590, stretch='always')
        selection = self.card(sidebar, 'SELECCIONAR REGISTRO')
        self.patient_box = self.selector(selection, '1  Paciente', self.patient, self.select_patient)
        self.study_box = self.selector(selection, '2  Estudio', self.study, self.select_study)
        self.channel_box = self.selector(selection, '3  Electrodo', self.channel, self.select_channel)
        ttk.Label(selection, text='Elegí un paciente, su estudio y luego\nun electrodo para visualizar la señal.',
                  style='Muted.TLabel', wraplength=280).pack(anchor='w', pady=(14, 4))
        notes = self.card(sidebar, 'EXPLORACIÓN TEMPORAL')
        ttk.Label(notes, text='Una señal por vez\nVentanas de 10 segundos\nCuadrícula vertical cada 5 segundos',
                  style='Card.TLabel', justify='left').pack(anchor='w', pady=8)
        ttk.Label(notes, text='Los datos originales se conservan.\nNo se aplica ningún procesamiento.',
                  style='Muted.TLabel', wraplength=280).pack(anchor='w', pady=8)
        info = self.card(main, 'INFORMACIÓN DEL ESTUDIO')
        self.info_fields = {}
        grid = ttk.Frame(info, style='Card.TFrame')
        grid.pack(fill='x', pady=8)
        for i, label in enumerate(('Paciente', 'Sesión', 'Condición', 'Ojos', 'Duración',
                                   'Muestreo', 'Canales EEG', 'Referencia')):
            row, col = divmod(i, 4)
            cell = ttk.Frame(grid, style='Card.TFrame', padding=(0, 3, 14, 3))
            cell.grid(row=row, column=col, sticky='nsew')
            grid.columnconfigure(col, weight=1)
            ttk.Label(cell, text=label, style='Muted.TLabel').pack(anchor='w')
            var = tk.StringVar(value='—')
            ttk.Label(cell, textvariable=var, style='Card.TLabel', wraplength=180).pack(anchor='w')
            self.info_fields[label] = var
        self.file_text = tk.StringVar(value='Seleccioná un estudio para ver su información.')
        file_label = ttk.Label(info, textvariable=self.file_text, style='Muted.TLabel')
        file_label.pack(fill='x')
        file_label.bind('<Configure>', lambda e: file_label.configure(wraplength=max(100, e.width)))
        chart = self.card(main, 'SEÑAL EN EL TIEMPO', expand=True)
        controls = ttk.Frame(chart, style='Card.TFrame')
        controls.pack(fill='x', pady=(8, 4))
        self.previous = ttk.Button(controls, text='← Anterior', command=lambda: self.navigate(-10))
        self.previous.pack(side='left')
        self.next = ttk.Button(controls, text='Siguiente →', command=lambda: self.navigate(10))
        self.next.pack(side='left', padx=6)
        ttk.Label(controls, text='Inicio (s)', style='Card.TLabel').pack(side='left', padx=(12, 5))
        self.start_entry = ttk.Entry(controls, textvariable=self.start_text, width=9)
        self.start_entry.pack(side='left')
        self.start_entry.bind('<Return>', lambda e: self.go())
        self.go_button = ttk.Button(controls, text='Ir', command=self.go)
        self.go_button.pack(side='left', padx=6)
        ttk.Label(chart, textvariable=self.window_text, style='Muted.TLabel').pack(anchor='w', pady=4)
        self.figure = Figure(figsize=(8, 4), dpi=100, facecolor='white', layout='constrained')
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart)
        self.toolbar = Toolbar(self.canvas, chart, pack_toolbar=False)
        self.toolbar.configure(background='white')
        self.toolbar.pack(side='bottom', fill='x')
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.update_controls()

    @staticmethod
    def card(parent, title, expand=False):
        border = tk.Frame(parent, bg='white', highlightbackground='#d9e1e2', highlightthickness=1)
        border.pack(fill='both' if expand else 'x', expand=expand, pady=(0, 10))
        inner = ttk.Frame(border, style='Card.TFrame', padding=12)
        inner.pack(fill='both', expand=True)
        ttk.Label(inner, text=title, style='Muted.TLabel',
                  font=('DejaVu Sans', 9, 'bold')).pack(anchor='w', pady=(0, 5))
        return inner

    @staticmethod
    def selector(parent, label, variable, callback):
        ttk.Label(parent, text=label, style='Card.TLabel').pack(anchor='w', pady=(12, 4))
        box = ttk.Combobox(parent, textvariable=variable, state='disabled', width=35)
        box.pack(fill='x')
        box.bind('<<ComboboxSelected>>', callback)
        return box

    def invalidate(self):
        self.generation += 1
        if self.future is not None:
            self.future.cancel()
        self.busy = False

    def submit(self, work, callback):
        token = self.generation
        self.busy = True
        self.update_controls()
        future = self.executor.submit(work)
        self.future = future
        future.add_done_callback(lambda f: self.events.put((token, callback, f)))

    def poll(self):
        while True:
            try:
                token, callback, future = self.events.get_nowait()
            except queue.Empty:
                break
            if token != self.generation or future.cancelled():
                continue
            self.busy = False
            try:
                callback(future.result())
            except Exception as exc:
                self.clear_plot('No se pudo cargar la señal')
                self.status.set(f'Error: {exc}')
            self.update_controls()
        if not self.closed:
            self.poll_id = self.root.after(50, self.poll)

    def inventory_loaded(self, inventory):
        self.inventory = inventory
        self.patient_box.configure(values=list(inventory), state='readonly' if inventory else 'disabled')
        self.status.set(f'{len(inventory)} participantes disponibles. Elegí un paciente.' if inventory
                        else 'No se encontraron participantes en la carpeta data.')

    def clear_info(self):
        for var in self.info_fields.values():
            var.set('—')
        self.file_text.set('Seleccioná un estudio para ver su información.')

    def select_patient(self, event=None):
        self.invalidate()
        self.recording = None
        self.study.set('')
        self.channel.set('')
        self.channel_box.configure(values=(), state='disabled')
        self.studies = self.inventory.get(self.patient.get(), [])
        # Include the filename only when labels would otherwise be ambiguous.
        labels = [s.label for s in self.studies]
        self.study_box.configure(values=[f'{s.label} — {s.path.name}' if labels.count(s.label) > 1
                                         else s.label for s in self.studies],
                                 state='readonly' if self.studies else 'disabled')
        self.clear_info()
        self.reset_time()
        self.clear_plot('Elegí un estudio' if self.studies else 'Este paciente no tiene estudios EEG')
        self.status.set('Elegí un estudio.' if self.studies else 'No hay estudios EEG disponibles para este paciente.')
        self.executor.submit(self.release_recording)
        self.update_controls()

    def release_recording(self):
        if self.worker_recording is not None:
            self.worker_recording.close()
            self.worker_recording = None

    def load(self, study):
        self.release_recording()
        self.worker_recording = open_recording(study)
        return self.worker_recording

    def select_study(self, event=None):
        self.invalidate()
        self.recording = None
        self.channel.set('')
        self.channel_box.configure(values=(), state='disabled')
        self.clear_info()
        self.reset_time()
        self.clear_plot('Cargando el estudio…')
        index = self.study_box.current()
        if index < 0:
            self.update_controls()
            return
        study = self.studies[index]
        self.status.set(f'Cargando {study.path.name}…')
        self.submit(lambda: self.load(study), self.recording_loaded)

    def recording_loaded(self, recording):
        self.recording = recording
        study = recording.study
        reference = recording.metadata.get('EEGReference') or 'No disponible'
        if reference == 'n/a':
            reference = 'No disponible'
        values = (study.participant, study.session, study.condition, study.eyes,
                  f'{recording.duration:g} s', f'{recording.sfreq:g} Hz',
                  str(len(recording.channels)), str(reference))
        for var, value in zip(self.info_fields.values(), values):
            var.set(value)
        self.file_text.set(study.path.name)
        self.channel_box.configure(values=recording.channels, state='readonly')
        self.clear_plot('Elegí un electrodo para ver su señal')
        self.status.set(recording.warning or 'Estudio cargado. Elegí un electrodo.')

    def reset_time(self):
        self.start = self.stop = 0.0
        self.start_text.set('0')
        self.window_text.set('Ventanas de 10 segundos')

    def select_channel(self, event=None):
        self.reset_time()
        self.show_window(0)

    def show_window(self, start):
        if self.recording is None or not self.channel.get():
            return
        self.invalidate()
        recording, channel = self.recording, self.channel.get()
        self.clear_plot('Cargando señal…')
        self.status.set(f'Leyendo {channel} desde {start:g} s…')
        self.submit(lambda: recording.read_window(channel, start), self.segment_loaded)

    def reset_toolbar(self):
        if self.toolbar.mode:
            if self.toolbar.mode.name == 'PAN':
                self.toolbar.pan()
            else:
                self.toolbar.zoom()
        self.toolbar.update()

    def clear_plot(self, message):
        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.text(.5, .5, message, transform=self.ax.transAxes, ha='center', va='center',
                     fontsize=12, color=MUTED)
        self.reset_toolbar()
        self.canvas.draw_idle()

    def segment_loaded(self, segment):
        self.start, self.stop = segment.start, segment.stop
        self.start_text.set(f'{self.start:g}')
        self.window_text.set(f'{self.start:g}–{self.stop:g} s de {self.recording.duration:g} s'
                             f'   ·   {len(segment.times):,} muestras   ·   Cuadrícula: 5 s')
        self.ax.clear()
        self.ax.set_axis_on()
        line, = self.ax.plot(segment.times, segment.values_uv, color=TEAL, linewidth=.8)
        # Preserve every sample in the rendered path, including brief extremes.
        line.get_path().should_simplify = False
        self.ax.set_xlim(segment.start, segment.stop)
        self.ax.margins(y=.08)
        self.ax.set_xlabel('Tiempo desde el inicio (s)')
        self.ax.set_ylabel('Amplitud (µV)')
        self.ax.set_title(f'{self.recording.study.participant} · {self.channel.get()}', loc='left', fontsize=11)
        self.ax.xaxis.set_major_locator(MultipleLocator(5))
        self.ax.grid(axis='x', color='#bdccce', linewidth=.8)
        self.ax.grid(axis='y', color='#edf1f1', linewidth=.5)
        for spine in self.ax.spines.values():
            spine.set_color('#d9e1e2')
        self.reset_toolbar()
        self.toolbar.push_current()
        self.canvas.draw_idle()
        self.status.set(self.recording.warning or 'Señal original · Sin filtros ni procesamiento')

    def update_controls(self):
        enabled = self.recording is not None and bool(self.channel.get()) and not self.busy
        self.previous.configure(state='normal' if enabled and self.start > 0 else 'disabled')
        self.next.configure(state='normal' if enabled and self.stop < self.recording.duration else 'disabled')
        self.start_entry.configure(state='normal' if enabled else 'disabled')
        self.go_button.configure(state='normal' if enabled else 'disabled')

    def navigate(self, delta):
        if not self.busy:
            self.show_window(max(0, self.start + delta))

    def go(self):
        if self.busy or self.recording is None or not self.channel.get():
            return
        try:
            start = float(self.start_text.get().strip().replace(',', '.'))
            if not 0 <= start < self.recording.duration:
                raise ValueError
        except ValueError:
            self.status.set(f'Ingresá un inicio entre 0 y {self.recording.duration:g} s, sin incluir el final.')
            return
        self.show_window(start)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.invalidate()
        self.root.after_cancel(self.poll_id)
        # Matplotlib schedules idle draws on Tk; cancel them before destroying it.
        for attr in ('_idle_draw_id', '_event_loop_id'):
            job = getattr(self.canvas, attr, None)
            if job:
                self.canvas.get_tk_widget().after_cancel(job)
                setattr(self.canvas, attr, None)
        self.executor.submit(self.release_recording)
        self.executor.shutdown(wait=False)
        self.root.destroy()


def main(data_dir):
    root = tk.Tk()
    App(root, data_dir)
    root.mainloop()
