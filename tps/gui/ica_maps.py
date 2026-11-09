"""Scrollable ICA map selection; geometry fallbacks are display-only and explicit."""
import math
import tkinter as tk
from tkinter import ttk

import mne
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def map_info(model):
    """Return a copy: never attach estimated sensor positions to the fitted model."""
    info = mne.pick_info(model.info, [model.info.ch_names.index(c) for c in model.ch_names])
    positions = np.array([ch['loc'][:3] for ch in info['chs']])
    if (np.isfinite(positions).all() and np.all(np.linalg.norm(positions, axis=1) > 0)
            and len(np.unique(positions, axis=0)) == len(positions)):
        return info, 'Posiciones guardadas en el modelo'
    available = mne.channels.get_builtin_montages()
    name = 'colin27_1020' if 'colin27_1020' in available else 'standard_1020'
    montage = mne.channels.make_standard_montage(name)
    missing = set(info.ch_names) - set(montage.ch_names)
    if missing:
        raise ValueError('Sin posiciones estándar para: ' + ', '.join(sorted(missing)))
    info.set_montage(montage, on_missing='raise')
    return info, 'Posiciones estándar 10–20 APROXIMADAS · solo para visualización'


class ICAMapGrid(tk.Toplevel):
    def __init__(self, parent, model, exclusions, inspect):
        super().__init__(parent)
        self.title('ICA · Mapas de componentes')
        self.geometry('1050x760')
        self.minsize(680, 420)
        self.transient(parent)
        self.model = model
        self.exclusions = exclusions
        self.inspect = inspect
        self.trace = None
        self.closed = False
        self.protocol('WM_DELETE_WINDOW', self.destroy)
        self.bind('<Destroy>', self.cleanup, add='+')
        top = ttk.Frame(self, padding=10)
        top.pack(fill='x')
        ttk.Label(top, text='Clic: conservar / excluir · Clic derecho: inspeccionar el componente',
                  font=('', 11, 'bold')).pack(anchor='w')
        ttk.Label(top, text='Verde = conservar · Rojo = excluir. Selección pendiente; confirmar en la ventana ICA.').pack(anchor='w')
        self.message = tk.StringVar()
        ttk.Label(top, textvariable=self.message, wraplength=980).pack(anchor='w')
        frame = ttk.Frame(self)
        frame.pack(fill='both', expand=True)
        self.scroll = tk.Canvas(frame, highlightthickness=0, background='white')
        vertical = ttk.Scrollbar(frame, orient='vertical', command=self.scroll.yview)
        self.scroll.configure(yscrollcommand=vertical.set)
        vertical.pack(side='right', fill='y')
        self.scroll.pack(fill='both', expand=True)
        columns = min(6, model.n_components_)
        rows = math.ceil(model.n_components_ / columns)
        self.figure = Figure(figsize=(10, max(2.1, rows*2.1)), dpi=100, facecolor='white')
        self.figure.subplots_adjust(left=.025, right=.975, top=1-.2/(rows*2.1), bottom=.05/(rows*2.1), hspace=.45, wspace=.25)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.scroll)
        self.window_id = self.scroll.create_window(0, 0, window=self.canvas.get_tk_widget(), anchor='nw')
        self.scroll.bind('<Configure>', self.resize)
        self.canvas.get_tk_widget().bind('<Configure>', lambda _: self.scroll.configure(scrollregion=self.scroll.bbox('all')))
        for sequence in ('<MouseWheel>', '<Button-4>', '<Button-5>'):
            self.canvas.get_tk_widget().bind(sequence, self.wheel)
        self.axes = {}
        weights = model.get_components()
        try:
            info, caption = map_info(model)
        except ValueError as exc:
            info, caption = None, str(exc)
        ttk.Label(top, text=caption, foreground='#865500', wraplength=980).pack(anchor='w')
        ttk.Label(top, text='Cada mapa se normaliza a ±1: rojo/azul = signo del peso, no artefacto ni µV.').pack(anchor='w')
        for k in range(model.n_components_):
            ax = self.figure.add_subplot(rows, columns, k+1)
            self.axes[ax] = k
            if info is not None:
                values = weights[:, k]
                maximum = np.max(np.abs(values))
                values = values / maximum if maximum else values
                try:
                    mne.viz.plot_topomap(values, info, axes=ax, show=False, cmap='RdBu_r',
                                         vlim=(-1, 1), sensors=True, contours=6)
                except (ValueError, RuntimeError) as exc:
                    ax.clear(); ax.text(.5, .5, 'Mapa no disponible', ha='center', va='center')
                    self.message.set(str(exc))
            else:
                ax.text(.5, .5, 'Sin geometría', ha='center', va='center')
            ax.set_axis_off()
        self.canvas.mpl_connect('button_press_event', self.click)
        self.trace = exclusions.trace_add('write', self.update_selection)
        self.update_selection()
        ttk.Button(self, text='Volver a la revisión', command=self.destroy).pack(pady=8)
        self.grab_set()

    def selected(self):
        values = {int(x.strip())-1 for x in self.exclusions.get().split(',') if x.strip()}
        if any(k < 0 or k >= self.model.n_components_ for k in values):
            raise ValueError('Componente fuera de rango.')
        return values

    def update_selection(self, *_):
        try:
            selected = self.selected()
        except ValueError:
            self.message.set('Corregí la lista de componentes: usá números válidos separados por comas.')
            return
        self.message.set(f'{len(selected)} de {self.model.n_components_} componentes seleccionados para excluir.')
        for ax, k in self.axes.items():
            excluded = k in selected
            ax.set_title(f'{k+1} · ' + ('EXCLUIR' if excluded else 'Conservar'), fontsize=9,
                         color='#8a2020' if excluded else '#18512c',
                         bbox=dict(facecolor='#ffd0ce' if excluded else '#d6efdb', edgecolor='none', pad=3))
        self.canvas.draw_idle()

    def click(self, event):
        if event.inaxes not in self.axes:
            return
        k = self.axes[event.inaxes]
        if event.button == 3:
            self.destroy()
            self.inspect(k)
        elif event.button == 1:
            try:
                selected = self.selected()
            except ValueError:
                return
            selected.symmetric_difference_update({k})
            self.exclusions.set(','.join(str(i+1) for i in sorted(selected)))

    def resize(self, event):
        self.scroll.itemconfigure(self.window_id, width=event.width)

    def wheel(self, event):
        steps = -1 if event.num == 4 or (event.num != 5 and event.delta > 0) else 1
        self.scroll.yview_scroll(steps*3, 'units')
        return 'break'

    def cleanup(self, event):
        if event.widget is not self or self.closed:
            return
        self.closed = True
        if self.trace is not None:
            self.exclusions.trace_remove('write', self.trace)
        if getattr(self.canvas, '_idle_draw_id', None):
            self.canvas.get_tk_widget().after_cancel(self.canvas._idle_draw_id)
            self.canvas._idle_draw_id = None
        if self.master.winfo_exists():
            self.master.grab_set()
