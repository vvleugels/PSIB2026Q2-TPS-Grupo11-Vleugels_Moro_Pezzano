"""Interfaz Tkinter; las lecturas se ejecutan en un único hilo de trabajo."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from .ica_ui import ICAUI
from .data import discover, open_recording
from .annotations import AnnotationStore, STATES
from .review_ui import ReviewUI
from .spectral_ui import SpectralUI
from .screening_ui import ScreeningUI
from .results_ui import ResultsUI
from .theme import (INK, INK_MUTED, INK_SOFT, IRIS, IRIS_DARK, AMBER, BG, CARD, MUTED, Stepper,
                    Tooltip, apply_theme, autowrap, callout, card, mark_image, style_dialog)

PROJECT = Path(__file__).resolve().parent.parent
STEP_TITLES = ('Elegir paciente', 'Elegir estudio', 'Elegir electrodo', 'Revisar filtrado (opcional)',
               'Buscar artefactos (opcional)', 'Revisar epochs', 'Calcular espectro')
ELECTRODE_IMAGE = 'Electrode-disposition.png'


class Toolbar(NavigationToolbar2Tk):
    toolitems = (
        ('Restaurar', 'Restaurar la ventana temporal', 'home', 'home'),
        ('Atrás', 'Volver a la vista anterior', 'back', 'back'),
        ('Adelante', 'Avanzar a la vista siguiente', 'forward', 'forward'),
        (None, None, None, None),
        ('Desplazar', 'Arrastrar para desplazar la vista', 'move', 'pan'),
        ('Zoom', 'Seleccionar un rectángulo para acercar', 'zoom_to_rect', 'zoom'),
    )


class App(ReviewUI, SpectralUI, ScreeningUI, ResultsUI, ICAUI):
    def __init__(self, root, data_dir, annotations_dir=None, results_dir=None):
        self.root = root
        self.data_dir = Path(data_dir)
        self.annotations_dir = Path(annotations_dir) if annotations_dir else self.data_dir.parent / "annotations"
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
        self.inventory_ready = False
        self.sidebar_visible = True
        self.info_expanded = None  # Sigue el tamaño de ventana hasta una elección manual.
        self.sidebar_width = 350
        self.patient = tk.StringVar()
        self.study = tk.StringVar()
        self.channel = tk.StringVar()
        self.start_text = tk.StringVar(value='0')
        self.status = tk.StringVar(value='Buscando participantes…')
        self.window_text = tk.StringVar(value='Ventanas de 10 segundos')
        self.instruction = tk.StringVar(value='Buscando participantes en la carpeta data…')
        self.init_review()
        self.init_spectral()
        self.init_screening()
        # Group results live next to data/ (results/), like annotations/.
        self.init_results(Path(results_dir) if results_dir else self.data_dir.parent / 'results', PROJECT)
        root.title('PSIB — Explorador EEG')
        root.minsize(960, 640)
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.maximize()
        self.fonts = apply_theme(root)
        self.set_icon()
        self.build()
        self.clear_plot('Elegí un paciente para comenzar')
        self.submit(lambda: discover(data_dir), self.inventory_loaded)
        self.poll_id = root.after(50, self.poll)

    # ------------------------------------------------------------------- look
    def maximize(self):
        """Abrir maximizada para que el gráfico y los pasos tengan espacio."""
        try:
            if sys.platform == 'win32':
                self.root.state('zoomed')
            else:
                self.root.attributes('-zoomed', True)
        except tk.TclError:
            self.root.geometry('1280x800')

    def set_icon(self):
        """Usa el logo (o assets/icon.png) como ícono de la ventana y de la barra de tareas."""
        try:
            self.icon = ImageTk.PhotoImage(mark_image(PROJECT, 64, 'icon', 'logo'), master=self.root)
            self.root.iconphoto(True, self.icon)
        except tk.TclError:
            pass   # un ícono faltante nunca debe impedir abrir la app

    def build(self):
        self.build_appbar()
        bottom = ttk.Frame(self.root, padding=(14, 6, 14, 12))
        bottom.pack(side='bottom', fill='x')
        self.status_label = ttk.Label(bottom, textvariable=self.status, style='Bold.TLabel')
        self.status_label.pack(fill='x')
        autowrap(self.status_label)
        self.panes = tk.PanedWindow(self.root, orient='horizontal', background=BG,
                                    sashwidth=12, sashrelief='flat', borderwidth=0)
        self.panes.pack(fill='both', expand=True, padx=14, pady=(8, 0))
        self.build_sidebar()
        self.main = ttk.Frame(self.panes)
        self.panes.add(self.sidebar, minsize=300, stretch='never')
        self.panes.add(self.main, minsize=590, stretch='always')
        self.build_main(self.main)
        self.add_tooltips()
        self.root.bind('<Configure>', self.responsive_layout, add='+')
        for sequence in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            self.root.bind_all(sequence, self.on_wheel, add='+')
        self.update_controls()

    def build_appbar(self):
        bar = tk.Frame(self.root, background=INK)
        bar.pack(fill='x')
        self.logo = ImageTk.PhotoImage(mark_image(PROJECT, 34, 'logo', 'icon'), master=self.root)
        tk.Label(bar, image=self.logo, background=INK).pack(side='left', padx=(16, 10), pady=8)
        tk.Label(bar, text='PSIB', font=self.fonts.title, foreground='white',
                 background=INK).pack(side='left')
        tk.Label(bar, text='Explorador EEG · sueño normal y privación de sueño', font=self.fonts.base,
                 foreground=INK_MUTED, background=INK).pack(side='left', padx=14, pady=(5, 0))
        tk.Label(bar, textvariable=self.signal_label, background=INK, foreground=INK_MUTED,
                 font=self.fonts.base).pack(side='right', padx=8)
        tk.Label(bar, text='Señal:', background=INK, foreground=INK_MUTED,
                 font=self.fonts.small).pack(side='right')
        tk.Frame(self.root, background=AMBER, height=3).pack(fill='x')

    def build_sidebar(self):
        """Panel izquierdo con selección, pasos y revisión; se desplaza si no entra en la pantalla."""
        self.sidebar = tk.Frame(self.panes, background=BG)
        self.side_canvas = tk.Canvas(self.sidebar, background=BG, highlightthickness=0, width=336)
        scroll = ttk.Scrollbar(self.sidebar, orient='vertical', command=self.side_canvas.yview)
        self.side_canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.side_canvas.pack(side='left', fill='both', expand=True)
        body = ttk.Frame(self.side_canvas, padding=(0, 0, 8, 8))
        body_id = self.side_canvas.create_window(0, 0, window=body, anchor='nw')
        self.side_canvas.bind('<Configure>',
                              lambda e: self.side_canvas.itemconfigure(body_id, width=e.width))
        body.bind('<Configure>',
                  lambda e: self.side_canvas.configure(scrollregion=self.side_canvas.bbox('all')))

        self.sidebar_rail = ttk.Frame(self.panes, width=38)
        self.sidebar_restore = ttk.Button(self.sidebar_rail, text='▶', width=3,
                                          command=self.toggle_sidebar)
        self.sidebar_restore.pack(anchor='n', pady=4)
        Tooltip(self.sidebar_restore, 'Mostrar panel lateral.')

        selection = self.card(body, 'Registro')
        self.sidebar_button = ttk.Button(selection, text='◀', width=3,
                                         command=self.toggle_sidebar)
        self.sidebar_button.place(relx=1, x=-8, y=0, anchor='ne')
        self.patient_box = self.selector(selection, 'Paciente', self.patient, self.select_patient)
        self.study_box = self.selector(selection, 'Estudio  (ses-1 normal · ses-2 privación)',
                                       self.study, self.select_study)
        self.channel_box = self.selector(selection, 'Electrodo', self.channel, self.select_channel,
                                         info=self.show_electrodes)

        filtering = self.card(body, 'Filtrado')
        row = ttk.Frame(filtering, style='Card.TFrame')
        row.pack(fill='x', pady=(4, 0))
        ttk.Label(row, text='Señal:', style='CardMuted.TLabel').pack(side='left')
        ttk.Label(row, textvariable=self.signal_label, style='CardBold.TLabel').pack(side='left', padx=(6, 0))
        self.config_button = ttk.Button(filtering, text='⚙ Configurar filtrado',
                                        command=self.configuration_dialog)
        self.config_button.pack(fill='x', pady=(6, 0))

        self.build_ica(body)

        steps = self.card(body, 'Pasos')
        self.stepper = Stepper(steps, self.fonts, len(STEP_TITLES), self.run_step)
        self.stepper.pack(fill='x', pady=(4, 0))
        self.hint = ttk.Label(steps, justify='left', style='Card.TLabel')
        self.hint.pack(fill='x', pady=8)
        autowrap(self.hint)
        self.next_button = ttk.Button(steps, text='Ir al siguiente paso ▶', style='Next.TButton',
                                      command=self.next_step)
        self.next_button.pack(fill='x')

        self.build_batch_card(body)
        self.build_review_sidebar(body, selection)

    def build_main(self, main):
        # Franja ámbar: qué hacer ahora.
        callout(main, self.instruction).pack(fill='x', pady=(0, 8))

        outer, info = card(main, 'Información del estudio')
        self.info_card = outer
        self.compact_text = tk.StringVar(value='Seleccioná un estudio para ver su información.')
        self.compact_info = ttk.Frame(main, style='Card.TFrame', padding=8)
        self.info_restore = ttk.Button(self.compact_info, text='▼', width=3,
                                       command=self.toggle_info)
        self.info_restore.pack(side='right', padx=(8, 0))
        self.compact_label = ttk.Label(self.compact_info, textvariable=self.compact_text,
                                       style='Card.TLabel', wraplength=550)
        self.compact_label.pack(side='left', fill='x', expand=True)
        Tooltip(self.info_restore, 'Expandir información del estudio.')
        self.info_fields = {}
        grid = ttk.Frame(info, style='Card.TFrame')
        grid.pack(fill='x', pady=(6, 4))
        for i, label in enumerate(('Paciente', 'Sesión', 'Condición', 'Ojos', 'Duración',
                                   'Muestreo', 'Canales EEG', 'Referencia')):
            row, col = divmod(i, 4)
            cell = ttk.Frame(grid, style='Card.TFrame', padding=(0, 3, 14, 3))
            cell.grid(row=row, column=col, sticky='nsew')
            grid.columnconfigure(col, weight=1)
            ttk.Label(cell, text=label, style='CardMuted.TLabel').pack(anchor='w')
            var = tk.StringVar(value='—')
            ttk.Label(cell, textvariable=var, style='CardBold.TLabel', wraplength=180).pack(anchor='w')
            self.info_fields[label] = var
        self.file_text = tk.StringVar(value='Seleccioná un estudio para ver su información.')
        footer = ttk.Frame(info, style='Card.TFrame')
        footer.pack(fill='x')
        self.info_button = ttk.Button(footer, text='▲', width=3, command=self.toggle_info)
        self.info_button.pack(side='right', padx=(8, 0))
        Tooltip(self.info_button, 'Plegar información del estudio.')
        file_label = ttk.Label(footer, textvariable=self.file_text, style='CardMuted.TLabel')
        file_label.pack(side='left', fill='x', expand=True)
        autowrap(file_label)

        self.view_tabs = ttk.Notebook(main)
        self.view_tabs.pack(fill='both', expand=True)
        temporal = ttk.Frame(self.view_tabs, style='Card.TFrame', padding=10)
        spectral = ttk.Frame(self.view_tabs, style='Card.TFrame', padding=10)
        self.view_tabs.add(temporal, text='Señal temporal')
        self.view_tabs.add(spectral, text='Análisis espectral')
        # «Resultados del grupo» replaces the read-only GroupUI tab; Tomás's figures and
        # tables from results/group20-channel-2026/analysis are listed inside it.
        group = ttk.Frame(self.view_tabs, style='Card.TFrame', padding=10)
        self.view_tabs.add(group, text='Resultados del grupo')
        self.group_tab = group
        self.view_tabs.bind('<<NotebookTabChanged>>', self.tab_changed)
        self.chart_card = self.view_tabs
        chart = temporal

        controls = ttk.Frame(chart, style='Card.TFrame')
        controls.pack(fill='x', pady=(0, 6))
        self.previous = ttk.Button(controls, text='◀ Anterior', style='Tool.TButton',
                                   command=lambda: self.navigate(-10))
        self.previous.pack(side='left')
        self.next = ttk.Button(controls, text='Siguiente ▶', style='Tool.TButton',
                               command=lambda: self.navigate(10))
        self.next.pack(side='left', padx=(4, 0))
        ttk.Label(controls, text='Inicio (s)', style='Card.TLabel').pack(side='left', padx=(14, 5))
        self.start_entry = ttk.Entry(controls, textvariable=self.start_text, width=9)
        self.start_entry.pack(side='left')
        self.start_entry.bind('<Return>', lambda e: self.go())
        self.go_button = ttk.Button(controls, text='Ir', style='Tool.TButton', command=self.go)
        self.go_button.pack(side='left', padx=4)
        ttk.Label(controls, textvariable=self.window_text, style='CardMuted.TLabel').pack(side='right')

        self.build_review_editor(chart)
        self.plot_container = ttk.Frame(chart, style='Card.TFrame')
        self.plot_view = tk.Canvas(self.plot_container, background='white', highlightthickness=0)
        plot_scroll = ttk.Scrollbar(self.plot_container, command=self.plot_view.yview)
        plot_scroll.pack(side='right', fill='y')
        self.plot_view.configure(yscrollcommand=plot_scroll.set)
        self.plot_view.pack(fill='both', expand=True)
        self.figure = Figure(figsize=(8, 4), dpi=100, facecolor='white', layout='constrained')
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_view)
        self.toolbar = Toolbar(self.canvas, chart, pack_toolbar=False)
        self.toolbar.configure(background='white')
        self.toolbar.pack(side='bottom', fill='x')
        self.plot_container.pack(fill='both', expand=True)
        self.plot_window = self.plot_view.create_window(0, 0, window=self.canvas.get_tk_widget(), anchor='nw')
        self.plot_view.bind('<Configure>', self.resize_plot)
        self.canvas.mpl_connect('button_press_event', self.plot_clicked)
        self.build_spectral(spectral, Toolbar)
        self.build_results(group)

    def add_tooltips(self):
        tips = [
            (self.patient_box, 'Participante del estudio (carpeta sub-XX dentro de data).'),
            (self.study_box, 'ses-1 es sueño normal y ses-2 es privación del sueño; '
                             'cada una tiene registros con ojos abiertos y cerrados.'),
            (self.channel_box, 'Electrodo cuya señal se muestra. Cambiarlo conserva el tiempo '
                               'y la epoch seleccionada.'),
            (self.multi_toggle, 'Muestra de 1 a 8 electrodos a la vez, en paneles con tiempo compartido.'),
            (self.next_button, 'Abre la herramienta del primer paso pendiente.'),
            (self.config_button, 'Activa el filtrado pasabanda FIR y ajusta sus cortes '
                                 '(por defecto apagado, 0,2–45 Hz).'),
            (self.electrode_info, 'Ver la disposición de los electrodos sobre la cabeza.'),
            (self.sidebar_button, 'Oculta el panel lateral para dar más espacio al gráfico.'),
            (self.previous, 'Ventana de 10 s anterior.'),
            (self.next, 'Ventana de 10 s siguiente.'),
            (self.start_entry, 'Segundo en el que empieza la ventana. Acepta punto o coma decimal; '
                               'Enter para ir.'),
            (self.screening_button, 'Mide la amplitud pico a pico de cada epoch. Las epochs sin avisos '
                                    'se aceptan y guardan automáticamente; las decisiones manuales se conservan.'),
            (self.epoch_tree, 'Elegí una epoch para llevar el gráfico a ella. También podés hacer clic '
                              'sobre su intervalo en el gráfico.'),
            (self.calculate_button, 'Calcula la PSD de Welch y la potencia por bandas con las epochs aceptadas.'),
            (self.spectral_settings_button, 'Duración del segmento Welch, solapamiento y límites de las bandas.'),
            (self.export_button, 'Guarda tablas (y opcionalmente el gráfico) en la carpeta que elijas.'),
        ]
        for widget, text in tips:
            Tooltip(widget, text)

    def toggle_sidebar(self):
        if self.sidebar_visible:
            self.sidebar_width = self.sidebar.winfo_width()
            self.panes.forget(self.sidebar)
            self.panes.add(self.sidebar_rail, before=self.main, minsize=38, width=38, stretch='never')
        else:
            self.panes.forget(self.sidebar_rail)
            self.panes.add(self.sidebar, before=self.main, minsize=300,
                           width=self.sidebar_width, stretch='never')
        self.sidebar_visible = not self.sidebar_visible

    def on_wheel(self, event):
        """La rueda desplaza el panel lateral cuando el puntero está sobre él."""
        try:
            widget = self.root.winfo_containing(event.x_root, event.y_root)
        except (KeyError, tk.TclError):
            return
        if self.figure_wheel(event, widget):   # visor de figuras de «Resultados del grupo»
            return
        while widget is not None and widget is not self.side_canvas:
            if isinstance(widget, (tk.Listbox, ttk.Treeview, tk.Text)):
                return   # esos controles ya se desplazan solos
            widget = widget.master
        if widget is None:
            return
        steps = -1 if event.num == 4 or (event.num != 5 and event.delta > 0) else 1
        self.side_canvas.yview_scroll(steps * 3, 'units')

    def toggle_info(self):
        self.info_expanded = not bool(self.info_card.winfo_manager())
        self.layout_info()

    def responsive_layout(self, event):
        if event.widget != self.root:
            return
        self.layout_info()

    def on_group_tab(self):
        return hasattr(self, 'group_tab') and self.view_tabs.select() == str(self.group_tab)

    def tab_changed(self, event=None):
        self.layout_info()
        self.refresh_guide()

    def layout_info(self):
        if self.on_group_tab():
            # The study information is irrelevant on «Resultados del grupo»: give that space to it.
            self.info_card.pack_forget()
            self.compact_info.pack_forget()
            return
        expanded = (self.root.winfo_height() >= 850 if self.info_expanded is None
                    else self.info_expanded)
        if not expanded:
            self.info_card.pack_forget()
            self.compact_info.pack(fill='x', pady=(0, 8), before=self.chart_card)
            self.compact_label.configure(wraplength=max(300, self.chart_card.winfo_width() - 20))
        else:
            self.compact_info.pack_forget()
            self.info_card.pack(fill='x', pady=(0, 8), before=self.chart_card)

    @staticmethod
    def card(parent, title, expand=False):
        outer, inner = card(parent, title)
        outer.pack(fill='both' if expand else 'x', expand=expand, pady=(0, 8))
        return inner

    def selector(self, parent, label, variable, callback, info=None):
        row = ttk.Frame(parent, style='Card.TFrame')
        row.pack(fill='x', pady=(8, 2))
        ttk.Label(row, text=label, style='Card.TLabel').pack(side='left')
        if info:
            icon = tk.Label(row, text='ⓘ', font=self.fonts.bold, foreground=IRIS, background=CARD,
                            cursor='hand2')
            icon.pack(side='left', padx=(6, 0))
            icon.bind('<Button-1>', lambda event: info())
            icon.bind('<Enter>', lambda event: icon.configure(foreground=IRIS_DARK))
            icon.bind('<Leave>', lambda event: icon.configure(foreground=IRIS))
            self.electrode_info = icon
        box = ttk.Combobox(parent, textvariable=variable, state='disabled', width=35)
        box.pack(fill='x')
        box.bind('<<ComboboxSelected>>', callback)
        return box

    # ------------------------------------------------------------ guided steps
    def pending_epochs(self):
        if not self.store:
            return None
        return [i for i, record in enumerate(self.store.records) if record['state'] == STATES[0]]

    def steps(self):
        """Filas de la lista de pasos: (título, ayuda, estado) con estado done / todo / skip."""
        pending = self.pending_epochs()
        reviewed = pending is not None and not pending
        shown = self.displayed is not None

        def row(index, hint, done):
            return STEP_TITLES[index], hint, 'done' if done else 'todo'

        return [
            row(0, 'Elegí un paciente en la tarjeta «Registro».', bool(self.patient.get())),
            row(1, 'Elegí un estudio: ses-1 es sueño normal y ses-2 privación del sueño.',
                self.recording is not None),
            row(2, 'Elegí un electrodo para ver su señal. Activá «Varios electrodos» para comparar hasta 8.',
                shown),
            row(3, 'Presioná «⚙ Configurar filtrado» y elegí si filtrar la señal (0,2–45 Hz por defecto). '
                   'Presioná «Aplicar» aunque lo dejes apagado.', self.filter_reviewed),
            row(4, 'Presioná «Buscar posibles artefactos»: las epochs sin avisos se aceptan solas. '
                   'Podés saltear este paso y revisar todo a mano.',
                self.screening_result is not None or reviewed),
            row(5, ('Elegí cada epoch pendiente y marcala con A (aceptada), R (con artefactos) '
                    'o U (sin revisar).' + (f' Quedan {len(pending)}.' if pending else '')),
                reviewed),
            row(6, 'Abrí «Análisis espectral» y presioná «Calcular».', self.spectral_result is not None),
        ]

    def refresh_guide(self):
        if not hasattr(self, 'stepper'):
            return
        steps = self.steps()
        nxt = next((i for i, step in enumerate(steps) if step[2] == 'todo'), None)
        self.stepper.update_steps([(title, state) for title, hint, state in steps], nxt)
        self.hint.configure(text=steps[nxt][1] if nxt is not None else
                            'Todo listo. Podés exportar el análisis espectral o seguir revisando.')
        self.next_button.configure(state='normal' if nxt is not None and not self.busy else 'disabled')
        self.instruction.set(self.instruction_text(nxt))

    def instruction_text(self, nxt):
        """Una frase con lo que hay que hacer ahora."""
        if not self.inventory_ready:
            return 'Buscando participantes en la carpeta data…'
        if not self.inventory:
            return 'No hay participantes en la carpeta data. Copiá allí el dataset ds004902 y reiniciá.'
        if self.on_group_tab():
            return self.results_instruction_text()
        if self.busy:
            return 'Trabajando… esperá a que termine la operación en curso.'
        if nxt == 0:
            return 'Empezá eligiendo un paciente en el panel izquierdo. El paso ▶ resaltado en ámbar es siempre el siguiente.'
        if nxt == 1:
            return 'Elegí un estudio del paciente: ses-1 es sueño normal y ses-2 es privación del sueño.'
        if nxt == 2:
            return 'Elegí un electrodo para ver su señal en ventanas de 10 s.'
        if nxt == 3:
            return ('Opcional: revisá el filtrado con «⚙ Configurar filtrado». Por defecto la señal se '
                    'muestra sin filtrar; presioná «Aplicar» para confirmar tu elección.')
        if nxt == 4:
            return ('Opcional: buscá posibles artefactos para aceptar automáticamente las epochs limpias, '
                    'o revisá cada epoch a mano.')
        if nxt == 5:
            return ('Revisá las epochs: elegí una en la lista o hacé clic sobre su intervalo en el gráfico, '
                    'y marcala con A (aceptada), R (con artefactos) o U (sin revisar).')
        if nxt == 6:
            return 'Todas las epochs están revisadas. Abrí «Análisis espectral» y presioná «Calcular».'
        return 'Análisis listo. Exportá los resultados o elegí otro estudio para compararlo.'

    def next_step(self):
        steps = self.steps()
        for index, step in enumerate(steps):
            if step[2] == 'todo':
                self.run_step(index)
                return

    def run_step(self, index):
        """Abre la herramienta de un paso (no hace nada si todavía no está disponible)."""
        if self.busy:
            return
        if index < 3:
            box = (self.patient_box, self.study_box, self.channel_box)[index]
            if str(box.cget('state')) == 'disabled':
                return
            self.view_tabs.select(0)
            box.focus_set()
            try:
                self.root.tk.call('ttk::combobox::Post', box)
            except tk.TclError:
                pass
        elif index == 3:
            if self.recording is not None:
                self.configuration_dialog()
        elif index == 4:
            self.view_tabs.select(0)
            self.screening_button.invoke()
        elif index == 5:
            pending = self.pending_epochs()
            if pending and self.displayed is not None:
                self.view_tabs.select(0)
                later = [i for i in pending if self.selected_epoch is None or i > self.selected_epoch]
                self.select_epoch((later or pending)[0], navigate=True)
        elif index == 6:
            self.view_tabs.select(1)
            self.calculate_button.invoke()

    def show_electrodes(self):
        """Ventana con la disposición de los electrodos (Electrode-disposition.png)."""
        if getattr(self, 'electrode_window', None) is not None and self.electrode_window.winfo_exists():
            self.electrode_window.lift()
            return
        path = next((p for p in (PROJECT / 'assets' / ELECTRODE_IMAGE, PROJECT / ELECTRODE_IMAGE)
                     if p.is_file()), None)
        if path is None:
            self.status.set(f'No se encontró {ELECTRODE_IMAGE} en la carpeta del proyecto.')
            return
        try:
            with Image.open(path) as image:
                image = image.convert('RGB')
        except OSError as exc:
            self.status.set(f'No se pudo abrir {path.name}: {exc}')
            return
        limit = (self.root.winfo_screenwidth() * .8, self.root.winfo_screenheight() * .75)
        scale = min(1.0, limit[0] / image.width, limit[1] / image.height)
        if scale < 1:
            image = image.resize((round(image.width * scale), round(image.height * scale)),
                                 Image.Resampling.LANCZOS)
        window = tk.Toplevel(self.root)
        window.title('Disposición de los electrodos')
        window.transient(self.root)
        window.resizable(False, False)
        style_dialog(window)
        self.electrode_window = window
        self.electrode_photo = ImageTk.PhotoImage(image, master=window)
        body = ttk.Frame(window, padding=12)
        body.pack(fill='both')
        current = ', '.join(self.visible_channels())
        ttk.Label(body, text='Disposición de los electrodos' + (f' · Seleccionado: {current}' if current else ''),
                  style='Bold.TLabel').pack(anchor='w', pady=(0, 8))
        tk.Label(body, image=self.electrode_photo, background='white', borderwidth=0).pack()
        ttk.Button(body, text='Cerrar', command=window.destroy).pack(anchor='e', pady=(10, 0))
        window.bind('<Escape>', lambda event: window.destroy())
        window.focus_set()

    def invalidate(self):
        if self.screening_pending:
            self.screening_pending = False
            self.screening_summary.set('Barrido cancelado. Volvé a buscar.')
        if self.spectral_pending:
            self.spectral_pending = False
            self.spectral_message.set('Cálculo cancelado. Volvé a calcular.')
        self.generation += 1
        if self.future is not None:
            self.future.cancel()
        self.busy = False

    def submit(self, work, callback, on_error=None):
        token = self.generation
        self.busy = True
        self.update_controls()
        future = self.executor.submit(work)
        self.future = future
        future.add_done_callback(lambda f: self.events.put((token, callback, f, on_error)))

    def poll(self):
        while True:
            try:
                token, callback, future, on_error = self.events.get_nowait()
            except queue.Empty:
                break
            if token != self.generation or future.cancelled():
                continue
            self.busy = False
            try:
                callback(future.result())
            except Exception as exc:
                if on_error:
                    on_error(exc)
                else:
                    self.clear_plot('No se pudo cargar la señal')
                    self.status.set(f'Error: {exc}')
            self.update_controls()
        self.poll_pipeline()
        if not self.closed:
            self.poll_id = self.root.after(50, self.poll)

    def inventory_loaded(self, inventory):
        self.inventory = inventory
        self.inventory_ready = True
        self.patient_box.configure(values=list(inventory), state='readonly' if inventory else 'disabled')
        self.status.set(f'{len(inventory)} participantes disponibles. Elegí un paciente.' if inventory
                        else 'No se encontraron participantes en la carpeta data.')

    def clear_info(self):
        self.compact_text.set('Seleccioná un estudio para ver su información.')
        for var in self.info_fields.values():
            var.set('—')
        self.file_text.set('Seleccioná un estudio para ver su información.')

    def select_patient(self, event=None):
        if not self.may_leave():
            self.patient.set(self.active_patient)
            return
        self.cancel_ica(close=True)
        self.active_patient = self.patient.get()
        self.active_study_index = -1
        self.reset_review()
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
        self.clear_batch_card()
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
        from . import ica
        import json
        pointer = ica.revision_root(self.annotations_dir, study) / 'latest.json'
        if pointer.exists():
            try:
                saved = json.loads(pointer.read_text())
                if saved.get('enabled', False):
                    ica.apply_revision(self.worker_recording, self.data_dir, self.annotations_dir, saved['revision'])
            except (OSError, ValueError, KeyError) as exc:
                self.worker_recording.warning += f' ICA no restaurada: {exc}'
        self.worker_recording.annotations = AnnotationStore(
            self.annotations_dir, self.worker_recording, self.data_dir)
        # Batch outcome for this study, read here so the first lookup (~0.4 s) stays off the UI thread.
        self.worker_recording.batch = self.results.study_summary(study.participant, study.session, study.task)
        return self.worker_recording

    def select_study(self, event=None):
        if not self.may_leave():
            if self.active_study_index >= 0:
                self.study_box.current(self.active_study_index)
            return
        self.cancel_ica(close=True)
        self.active_study_index = self.study_box.current()
        self.reset_review()
        self.invalidate()
        self.recording = None
        self.channel.set('')
        self.channel_box.configure(values=(), state='disabled')
        self.clear_info()
        self.clear_batch_card('Cargando…')
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
        self.compact_text.set(f'{study.participant} · {study.label}\n'
                              f'{recording.duration:g} s · {recording.sfreq:g} Hz · '
                              f'{len(recording.channels)} EEG · Referencia: {reference}\n'
                              f'{study.path.name}')
        self.channel_box.configure(values=recording.channels, state='readonly')
        self.clear_plot('Elegí un electrodo para ver su señal')
        self.status.set(recording.warning or 'Estudio cargado. Elegí un electrodo.')
        self.study_batch = getattr(recording, 'batch', None)
        self.update_batch_card()
        self.review_loaded(recording)

    def reset_time(self):
        self.start = self.stop = 0.0
        self.start_text.set('0')
        self.window_text.set('Ventanas de 10 segundos')

    def select_channel(self, event=None):
        self.invalidate_spectrum()
        if self.multi.get() and self.recording and self.channel.get():
            self.channel_list.selection_clear(0, 'end')
            self.channel_list.selection_set(self.recording.channels.index(self.channel.get()))
            self.last_multi = self.channel_list.curselection()
        self.update_batch_card()
        self.show_window(self.start)

    def show_window(self, start):
        if not self.flush_autosave():
            return
        if self.recording is None:
            return
        channels = self.visible_channels()
        self.invalidate()
        if not channels:
            self.clear_plot('Elegí al menos un electrodo')
            self.update_controls()
            return
        recording, config = self.recording, self.config
        # Keep the current plot while the next window loads when the electrodes are the same:
        # draw_segment then only updates the lines (faster ◀/▶, no blank flash).
        if self.displayed is None or self.displayed.channels != tuple(channels):
            self.clear_plot('Cargando señal…')
        self.status.set(f'Leyendo {", ".join(channels)} desde {start:g} s…')
        self.submit(lambda: recording.read_channels(channels, start, config), self.segment_loaded)

    def reset_toolbar(self):
        if self.toolbar.mode:
            if self.toolbar.mode.name == 'PAN':
                self.toolbar.pan()
            else:
                self.toolbar.zoom()
        self.toolbar.update()

    def clear_plot(self, message):
        self.displayed = None
        self.epoch_artists = []
        self.segment_lines = []
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.resize_plot()
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
        if self.store and (self.selected_epoch is None or
                           self.store.epochs[self.selected_epoch].stop <= self.start or
                           self.store.epochs[self.selected_epoch].start >= self.stop):
            self.selected_epoch = next(e.index for e in self.store.epochs
                                       if e.stop > self.start)
            self.epoch_tree.selection_set(str(self.selected_epoch))
            self.epoch_tree.see(str(self.selected_epoch))
            self.load_editor()
        self.draw_segment(segment)
        self.status.set(self.recording.warning or f'{self.config.label} · Epochs de 5 s · Sin exclusión de muestras')

    def update_controls(self):
        self.refresh_ica()
        enabled = self.recording is not None and bool(self.visible_channels()) and not self.busy
        self.previous.configure(state='normal' if enabled and self.start > 0 else 'disabled')
        self.next.configure(state='normal' if enabled and self.stop < self.recording.duration else 'disabled')
        self.start_entry.configure(state='normal' if enabled else 'disabled')
        self.go_button.configure(state='normal' if enabled else 'disabled')
        self.config_button.configure(state='normal' if self.recording and not self.busy else 'disabled')
        can_mark = (enabled and self.store and not self.store.blocked
                    and self.selected_epoch is not None and self.displayed is not None)
        for button in self.mark_buttons + [self.note_entry]:
            button.configure(state='normal' if can_mark else 'disabled')
        self.retry_save_button.configure(state='normal' if self.store and self.store.dirty and not self.busy else 'disabled')
        self.update_reason()
        self.update_spectral_controls()
        self.update_screening_controls()
        self.refresh_guide()

    def navigate(self, delta):
        if not self.busy:
            self.show_window(max(0, self.start + delta))

    def go(self):
        if self.busy or self.recording is None or not self.visible_channels():
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
        if self.runner.running:
            if not messagebox.askyesno('Proceso en curso',
                                       'Hay un paso de «Resultados del grupo» en ejecución. ¿Cancelarlo y salir? '
                                       'Lo ya calculado por el lote se conserva.', parent=self.root):
                return
            self.runner.cancel()
        if not self.may_leave():
            return
        self.cancel_ica(close=True)
        self.closed = True
        self.invalidate()
        self.root.after_cancel(self.poll_id)
        # Matplotlib schedules idle draws on Tk; cancel them before destroying it.
        for canvas in (self.canvas, self.spectral_canvas):
            for attr in ('_idle_draw_id', '_event_loop_id'):
                job = getattr(canvas, attr, None)
                if job:
                    canvas.get_tk_widget().after_cancel(job)
                    setattr(canvas, attr, None)
        self.executor.submit(self.release_recording)
        self.executor.shutdown(wait=False)
        self.root.destroy()


def main(data_dir, root=None):
    """Run the app; gui.py passes a root that already shows a loading message."""
    if root is None:
        if sys.platform == 'win32':
            import ctypes   # identidad propia en la barra de tareas, separada de python.exe
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('psib.eeg.explorer')
        root = tk.Tk()
    for child in root.winfo_children():   # remove the loading screen
        child.destroy()
    app = App(root, data_dir)
    try:
        root.mainloop()
    finally:
        app.executor.shutdown(wait=True)
