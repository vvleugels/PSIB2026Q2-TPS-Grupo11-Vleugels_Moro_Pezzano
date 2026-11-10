"""«Canales malos» card: automatic detection (tps/core/badchannels.py) and manual marks.

Manual marks are saved in the study's annotations (AnnotationStore.bad_channels) and the
group batch excludes them; automatic results are recomputed on demand and not saved, since
the batch recomputes them with its own settings.
"""
import tkinter as tk
from tkinter import ttk

from tps import config
from tps.core.badchannels import CRITERIA, BadChannelConfig, find_bad_channels
from tps.gui.theme import Tooltip, autowrap


class BadChannelsUI:
    def init_bad_channels(self):
        self.auto_bad = {}            # {channel: {'criterion', 'reason'}} of the last detection
        self.auto_bad_filter = None   # filter label used by that detection
        self.bad_reason = tk.StringVar()
        self.bad_summary = tk.StringVar(value='Elegí un estudio.')

    def build_bad_channels(self, parent):
        card = self.card(parent, 'Canales malos')
        ttk.Label(card, textvariable=self.bad_summary, style='CardMuted.TLabel').pack(fill='x', pady=(4, 4))
        autowrap(card.winfo_children()[-1])
        frame = ttk.Frame(card, style='Card.TFrame')
        frame.pack(fill='x')
        self.bad_tree = ttk.Treeview(frame, columns=('source', 'reason'), height=4, selectmode='browse')
        self.bad_tree.heading('#0', text='Canal')
        self.bad_tree.column('#0', width=56, stretch=False)
        self.bad_tree.heading('source', text='Origen')
        self.bad_tree.column('source', width=58, stretch=False)
        self.bad_tree.heading('reason', text='Motivo')
        self.bad_tree.column('reason', width=150)
        self.bad_tree.tag_configure('manual', foreground='#a12b25')
        scroll = ttk.Scrollbar(frame, command=self.bad_tree.yview)
        scroll.pack(side='right', fill='y')
        self.bad_tree.configure(yscrollcommand=scroll.set)
        self.bad_tree.pack(fill='x', expand=True)
        self.bad_tree.bind('<<TreeviewSelect>>', self.bad_selected)
        self.bad_detect_button = ttk.Button(card, text='Detectar canales malos', command=self.detect_bad_channels)
        self.bad_detect_button.pack(fill='x', pady=(6, 0))
        row = ttk.Frame(card, style='Card.TFrame')
        row.pack(fill='x', pady=(6, 0))
        ttk.Label(row, text='Motivo', style='Card.TLabel').pack(side='left')
        self.bad_reason_entry = ttk.Entry(row, textvariable=self.bad_reason, width=12)
        self.bad_reason_entry.pack(side='left', fill='x', expand=True, padx=(6, 0))
        self.bad_mark_button = ttk.Button(card, text='Marcar / desmarcar electrodo actual',
                                          command=lambda: self.toggle_bad_channel(self.channel.get()))
        self.bad_mark_button.pack(fill='x', pady=(4, 0))
        Tooltip(self.bad_detect_button,
                'Busca en todo el registro: señal plana, amplitud anómala, baja correlación con los demás '
                'canales y ruido de alta frecuencia (criterios de EEGLAB/PREP, umbrales no validados). '
                'Usa el filtrado del explorador.')
        Tooltip(self.bad_mark_button,
                'Marca el electrodo como malo en todo el registro (se guarda en las anotaciones y el '
                'análisis de grupo lo excluye). En «Todos apilados» también se marca con un clic en su nombre.')

    # ----------------------------------------------------------------- state
    def bad_channel_set(self) -> set:
        manual = set(self.store.bad_channels) if self.store else set()
        return manual | set(self.auto_bad)

    def clear_bad_channels(self):
        self.auto_bad, self.auto_bad_filter = {}, None
        self.refresh_bad_channels()

    def refresh_bad_channels(self):
        if not hasattr(self, 'bad_tree'):
            return
        self.bad_tree.delete(*self.bad_tree.get_children())
        manual = self.store.bad_channels if self.store else {}
        rows = [(c, 'Manual', v['reason'] or 'marcado a mano', 'manual') for c, v in manual.items()]
        rows += [(c, 'Auto', CRITERIA.get(v['criterion'], v['criterion']), 'auto')
                 for c, v in self.auto_bad.items() if c not in manual]
        order = {c: i for i, c in enumerate(self.recording.channels)} if self.recording else {}
        for channel, source, reason, tag in sorted(rows, key=lambda r: order.get(r[0], 0)):
            self.bad_tree.insert('', 'end', iid=channel, text=channel, values=(source, reason), tags=(tag,))
        if self.recording is None:
            self.bad_summary.set('Elegí un estudio.')
        else:
            parts = [f'{len(manual)} marcado{"s" if len(manual) != 1 else ""} a mano']
            if self.auto_bad_filter is None:
                parts.append('detección automática sin ejecutar')
            else:
                parts.append(f'{len(self.auto_bad)} detectado{"s" if len(self.auto_bad) != 1 else ""} '
                             f'({self.auto_bad_filter})')
            self.bad_summary.set(' · '.join(parts) + '.')
        self.update_bad_controls()
        if getattr(self, 'stacked', False):
            self.style_stacked()

    def update_bad_controls(self):
        if not hasattr(self, 'bad_detect_button'):
            return
        ready = self.recording is not None and not self.busy
        self.bad_detect_button.configure(state='normal' if ready else 'disabled')
        can_mark = ready and self.store is not None and not self.store.blocked and bool(self.channel.get())
        self.bad_mark_button.configure(state='normal' if can_mark else 'disabled')

    def bad_selected(self, event=None):
        """Selecting a row shows that electrode (single-channel mode) and fills its reason."""
        selected = self.bad_tree.selection()
        if not selected or self.recording is None:
            return
        channel = selected[0]
        if self.store and channel in self.store.bad_channels:
            self.bad_reason.set(self.store.bad_channels[channel]['reason'])
        elif channel in self.auto_bad:
            self.status.set(f"{channel}: {self.auto_bad[channel]['reason']}")
        if channel != self.channel.get() and not getattr(self, 'stacked', False) and not self.multi.get():
            self.channel.set(channel)
            self.select_channel()

    # --------------------------------------------------------------- actions
    def toggle_bad_channel(self, channel):
        if not channel or self.store is None or self.recording is None:
            return
        if self.store.blocked:
            self.status.set(self.store.blocked)
            return
        try:
            if channel in self.store.bad_channels:
                self.store.unmark_bad_channel(channel)
                message = f'{channel} ya no está marcado como malo.'
            else:
                self.store.mark_bad_channel(channel, self.bad_reason.get().strip()[:200])
                message = f'{channel} marcado como malo: el análisis de grupo lo excluirá.'
        except ValueError as exc:
            self.status.set(str(exc))
            return
        if self.save_annotations():
            self.status.set(message)
        self.refresh_bad_channels()

    def detect_bad_channels(self):
        if self.busy or self.recording is None:
            return
        recording, filtering = self.recording, self.config
        criteria = config.BAD_CHANNELS or BadChannelConfig()
        self.bad_summary.set('Buscando canales malos en todo el registro…')

        def done(found):
            if recording is not self.recording:
                return
            self.auto_bad, self.auto_bad_filter = found, filtering.label
            self.refresh_bad_channels()
            self.status.set(f'{len(found)} canal{"es" if len(found) != 1 else ""} malo'
                            f'{"s" if len(found) != 1 else ""} detectado{"s" if len(found) != 1 else ""}.'
                            ' El análisis de grupo también los excluye; marcá a mano otros que veas mal.')
            self.update_controls()

        self.submit(lambda: find_bad_channels(recording, filtering, criteria), done,
                    on_error=lambda exc: self.bad_summary.set(f'No se pudo detectar: {exc}'))
