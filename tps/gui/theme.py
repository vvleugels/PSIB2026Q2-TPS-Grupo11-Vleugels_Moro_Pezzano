"""Identidad visual del explorador EEG: paleta, fuentes, estilos ttk, logo, pasos y ayudas.

Sigue el estilo de la GUI de la guía PSIB: el teal estructura la interfaz y el
ámbar se reserva para «lo que hay que hacer ahora»."""
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from types import SimpleNamespace

from PIL import Image, ImageDraw

INK = '#0f2e36'
INK_MUTED = '#9db8be'
INK_SOFT = '#1c4752'
IRIS = '#0e7c86'
IRIS_DARK = '#0a5f67'
IRIS_LIGHT = '#d7eef0'
AMBER = '#f5a524'
AMBER_DARK = '#dd9013'
AMBER_LIGHT = '#fff4dc'
BG = '#f2f5f5'
CARD = '#ffffff'
BORDER = '#d9e1e2'
TEXT = '#17262b'
MUTED = '#62767c'
ERROR = '#a12b25'

# Trazo EEG del logo, en fracciones del ancho/alto del dibujo.
_WAVE = ((.18, .52), (.30, .52), (.36, .40), (.42, .64), (.48, .26), (.55, .74),
         (.61, .44), (.67, .56), (.72, .50), (.82, .50))


def make_mark(size):
    """Logo del explorador: un trazo EEG dentro de un disco teal con anillo ámbar (RGBA)."""
    s = size * 4
    image = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    c = s / 2

    def circle(radius, **kwargs):
        draw.ellipse([c - radius, c - radius, c + radius, c + radius], **kwargs)

    circle(s * 0.48, fill=IRIS)
    circle(s * 0.48, outline=AMBER, width=max(2, int(s * 0.04)))
    circle(s * 0.36, fill=INK)
    points = [(x * s, y * s) for x, y in _WAVE]
    draw.line(points, fill='white', width=max(2, int(s * 0.045)), joint='curve')
    return image.resize((size, size), Image.Resampling.LANCZOS)


def mark_image(root, size, *names):
    """Usa assets/<nombre>.png si el proyecto lo tiene; si no, dibuja el logo por defecto."""
    for name in names:
        path = root / 'assets' / f'{name}.png'
        if path.is_file():
            try:
                with Image.open(path) as image:
                    return image.convert('RGBA').resize((size, size), Image.Resampling.LANCZOS)
            except OSError:
                continue
    return make_mark(size)


def apply_theme(window):
    """Configura fuentes y estilos ttk del tema claro; devuelve las fuentes de la app."""
    families = set(tkfont.families(window))
    family = next((f for f in ('Inter', 'Segoe UI', 'Ubuntu', 'Noto Sans', 'Cantarell', 'DejaVu Sans')
                   if f in families), None)
    for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont'):
        font = tkfont.nametofont(name)
        font.configure(size=10)
        if family:
            font.configure(family=family)
    tkfont.nametofont('TkFixedFont').configure(size=10)

    base = tkfont.nametofont('TkDefaultFont')
    fonts = SimpleNamespace(base=base, bold=base.copy(), title=base.copy(), small=base.copy(),
                            note=base.copy())
    fonts.bold.configure(weight='bold')
    fonts.title.configure(size=16, weight='bold')
    fonts.small.configure(size=9, weight='bold')
    fonts.note.configure(size=9)

    style = ttk.Style(window)
    style.theme_use('clam')
    window.configure(background=BG)
    style.configure('.', background=BG, foreground=TEXT, bordercolor=BORDER, focuscolor=BG)
    style.configure('TLabel', background=BG)
    style.configure('Muted.TLabel', foreground=MUTED)
    style.configure('Bold.TLabel', font=fonts.bold)

    # Tarjetas: superficies blancas con sus propias variantes de etiqueta y casilla.
    style.configure('Card.TFrame', background=CARD)
    style.configure('Card.TLabel', background=CARD)
    style.configure('CardBold.TLabel', background=CARD, font=fonts.bold)
    style.configure('CardMuted.TLabel', background=CARD, foreground=MUTED)
    style.configure('CardError.TLabel', background=CARD, foreground=ERROR)
    style.configure('CardTitle.TLabel', background=CARD, foreground=MUTED, font=fonts.small)
    style.configure('Info.TFrame', background=IRIS_LIGHT)
    style.configure('Info.TLabel', background=IRIS_LIGHT, foreground=INK)
    style.configure('InfoTitle.TLabel', background=IRIS_LIGHT, foreground=IRIS_DARK, font=fonts.bold)
    style.configure('Callout.TLabel', background=AMBER_LIGHT, foreground=INK, padding=(12, 8))
    for kind in ('TCheckbutton', 'TRadiobutton'):
        style.configure(f'Card.{kind}', background=CARD)
        style.map(f'Card.{kind}', background=[('active', CARD)],
                  indicatorbackground=[('selected', IRIS), ('!selected', 'white')],
                  indicatorforeground=[('selected', 'white')],
                  foreground=[('disabled', '#a2b0b3')])

    style.configure('TEntry', padding=5, fieldbackground='white')
    style.map('TEntry', fieldbackground=[('readonly', '#eef2f3'), ('disabled', '#eef2f3')],
              bordercolor=[('focus', IRIS)])
    style.configure('TCombobox', padding=4, fieldbackground='white', arrowcolor=MUTED)
    style.map('TCombobox', fieldbackground=[('readonly', 'white'), ('disabled', '#eef2f3')],
              bordercolor=[('focus', IRIS)])

    style.configure('TButton', padding=(12, 6), background='#e6edee', borderwidth=1)
    style.map('TButton', background=[('disabled', '#f0f4f4'), ('active', '#d5e0e2')],
              foreground=[('disabled', '#a2b0b3')])
    style.configure('Tool.TButton', padding=(8, 5))
    style.configure('Accent.TButton', padding=(18, 8), background=IRIS, foreground='white',
                    font=fonts.bold, bordercolor=IRIS)
    style.map('Accent.TButton', background=[('disabled', '#a9cfd3'), ('active', IRIS_DARK)],
              foreground=[('disabled', 'white')])
    style.configure('Next.TButton', background=AMBER, foreground=INK, font=fonts.bold,
                    bordercolor=AMBER)
    style.map('Next.TButton', background=[('disabled', '#f1e3c4'), ('active', AMBER_DARK)],
              foreground=[('disabled', '#9a8c6a')])
    style.configure('Link.TButton', background=BG, foreground=IRIS, borderwidth=0, padding=(2, 4))
    style.map('Link.TButton', background=[('active', BG)], foreground=[('active', IRIS_DARK)])
    style.configure('CardLink.TButton', background=CARD, foreground=IRIS, borderwidth=0, padding=(0, 2))
    style.map('CardLink.TButton', background=[('active', CARD)], foreground=[('active', IRIS_DARK)])

    style.configure('TNotebook', background=BG, borderwidth=0)
    style.configure('TNotebook.Tab', padding=(18, 8), background='#e3eaeb', foreground=MUTED)
    style.map('TNotebook.Tab', background=[('selected', CARD)],
              foreground=[('selected', IRIS_DARK)])
    style.configure('Treeview', rowheight=25, background='white', fieldbackground='white',
                    bordercolor=BORDER)
    style.configure('Treeview.Heading', font=fonts.bold, padding=(6, 5), background='#e6edee')
    style.map('Treeview', background=[('selected', IRIS)], foreground=[('selected', 'white')])
    for orient in ('Vertical', 'Horizontal'):
        style.configure(f'{orient}.TScrollbar', background='#c3d0d2', troughcolor=BG,
                        bordercolor=BG, arrowcolor=MUTED)
    return fonts


def card(parent, title, collapsible=False):
    """Panel blanco con borde y un título pequeño; devuelve (marco exterior, marco del contenido).

    collapsible: agrega un triángulo en la esquina del título (▾ abierto, ▸ plegado) que
    oculta o muestra el contenido; el título queda siempre visible."""
    outer = tk.Frame(parent, background=CARD, highlightbackground=BORDER,
                     highlightcolor=BORDER, highlightthickness=1)
    inner = ttk.Frame(outer, style='Card.TFrame', padding=(10, 6, 10, 8))
    inner.pack(fill='both', expand=True)
    header = ttk.Frame(inner, style='Card.TFrame')
    header.pack(fill='x')
    title_label = ttk.Label(header, text=title.upper(), style='CardTitle.TLabel')
    title_label.pack(side='left', anchor='w')
    body = ttk.Frame(inner, style='Card.TFrame')
    body.pack(fill='both', expand=True)
    if collapsible:
        arrow = tk.Label(header, text='▾', background=CARD, foreground=MUTED, cursor='hand2',
                         font=('TkDefaultFont', 11, 'bold'), padx=4)
        arrow.pack(side='right')

        def toggle(event=None):
            if body.winfo_manager():
                body.pack_forget()
                arrow.configure(text='▸')
                inner.configure(padding=(10, 6, 10, 6))
                if outer.winfo_manager() == 'pack':   # a plegado no debe ocupar el espacio libre
                    outer.pack_configure(expand=False, fill='x')
            else:
                body.pack(fill='both', expand=True)
                arrow.configure(text='▾')
                inner.configure(padding=(10, 6, 10, 8))
                if outer.winfo_manager() == 'pack' and getattr(outer, 'expands', False):
                    outer.pack_configure(expand=True, fill='both')

        for widget in (arrow, title_label):   # el título también pliega
            widget.bind('<Button-1>', toggle)
        arrow.bind('<Enter>', lambda e: arrow.configure(foreground=IRIS_DARK))
        arrow.bind('<Leave>', lambda e: arrow.configure(foreground=MUTED))
        outer.toggle = toggle
        outer.collapsed = lambda: not body.winfo_manager()
    return outer, body


def callout(parent, variable, color=AMBER, light=AMBER_LIGHT):
    """Franja con barra de color a la izquierda y texto que se ajusta al ancho."""
    frame = tk.Frame(parent, background=light)
    tk.Frame(frame, background=color, width=4).pack(side='left', fill='y')
    message = ttk.Label(frame, textvariable=variable, style='Callout.TLabel', anchor='w',
                        justify='left')
    message.pack(side='left', fill='x', expand=True)
    message.bind('<Configure>', lambda e: message.configure(wraplength=max(e.width - 30, 200)))
    return frame


def autowrap(label):
    """Ajusta el texto de una etiqueta al ancho que recibe."""
    label.bind('<Configure>', lambda event: label.configure(wraplength=max(event.width - 4, 100)))


def style_dialog(dialog):
    """Fondo del tema para ventanas secundarias."""
    dialog.configure(background=BG)


class Tooltip:
    """Ayuda breve que aparece al dejar el mouse sobre un control."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.window = None
        self.job = None
        widget.bind('<Enter>', self.schedule, add='+')
        widget.bind('<Leave>', self.hide, add='+')
        widget.bind('<ButtonPress>', self.hide, add='+')

    def schedule(self, event=None):
        self.hide()
        self.job = self.widget.after(500, self.show)

    def show(self):
        self.job = None
        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
            self.window = tk.Toplevel(self.widget)
        except tk.TclError:
            return
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f'+{x}+{y}')
        tk.Label(self.window, text=self.text, justify='left', wraplength=320,
                 background='#fffbe6', foreground=TEXT, relief='solid',
                 borderwidth=1, padx=8, pady=5).pack()

    def hide(self, event=None):
        if self.job:
            self.widget.after_cancel(self.job)
            self.job = None
        if self.window:
            self.window.destroy()
            self.window = None


class Stepper(tk.Canvas):
    """Lista vertical de pasos numerados. Hecho = check teal, siguiente = ámbar, pendiente = contorno.

    Las filas se achican si la ventana es baja, para que todos los pasos sigan visibles."""

    ROW = 33
    RADIUS = 11

    def __init__(self, parent, fonts, count, on_click, width=236):
        super().__init__(parent, width=width, height=count * self.ROW, background=CARD,
                         highlightthickness=0, cursor='hand2')
        self.fonts = fonts
        self.count = count
        self.on_click = on_click
        self.steps = []
        self.nxt = None
        self.bind('<Button-1>', self.click)
        self.bind('<Configure>', lambda event: self.draw())

    @property
    def row(self):
        return max(24, min(self.ROW, self.winfo_height() / self.count))

    def click(self, event):
        index = int(event.y // self.row)
        if 0 <= index < self.count:
            self.on_click(index)

    def update_steps(self, steps, nxt):
        """steps: lista de (título, estado) con estado done / todo / skip; nxt: índice resaltado."""
        if (steps, nxt) == (self.steps, self.nxt):
            return
        self.steps = steps
        self.nxt = nxt
        self.draw()

    def draw(self):
        self.delete('all')
        row, nxt = self.row, self.nxt
        width = self.winfo_width()
        x, r = 20, min(self.RADIUS, row / 2 - 3)
        for i, (title, state) in enumerate(self.steps):
            y = i * row + row / 2
            if i == nxt:
                self.create_rectangle(0, i * row + 2, width, (i + 1) * row - 2,
                                      fill=AMBER_LIGHT, outline='')
            if i < len(self.steps) - 1:
                self.create_line(x, y + r, x, y + row - r, width=2,
                                 fill=IRIS if state == 'done' else BORDER)
        for i, (title, state) in enumerate(self.steps):
            y = i * row + row / 2
            box = (x - r, y - r, x + r, y + r)
            if state == 'done':
                self.create_oval(*box, fill=IRIS, outline=IRIS)
                glyph, glyph_color = '✔', 'white'
            elif i == nxt:
                self.create_oval(*box, fill=AMBER, outline=AMBER)
                glyph, glyph_color = str(i + 1), INK
            elif state == 'skip':
                self.create_oval(*box, fill='#eef2f3', outline='#d5dcdd')
                glyph, glyph_color = '–', MUTED
            else:
                self.create_oval(*box, fill=CARD, outline='#b4c3c6', width=2)
                glyph, glyph_color = str(i + 1), MUTED
            self.create_text(x, y, text=glyph, fill=glyph_color, font=self.fonts.small)
            weight = self.fonts.bold if i == nxt else self.fonts.base
            color = TEXT if state == 'done' or i == nxt else MUTED
            self.create_text(x + r + 12, y, text=title, anchor='w', fill=color, font=weight)
