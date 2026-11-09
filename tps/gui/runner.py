"""Run the batch / statistics / figures steps as subprocesses, without blocking the GUI.

The steps are the same commands documented in the README (python -m batch.run_batch,
analysis.stats, analysis.figures). Running them in a child process keeps the explorer
responsive and lets the user cancel without leaving half-initialised state in the GUI.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading

STEPS = ('batch', 'stats', 'figures')
STEP_LABELS = {'batch': 'Potencias por lote', 'stats': 'Estadística', 'figures': 'Figuras'}
PROGRESS = re.compile(r'\[(\d+)/(\d+)\]')


@dataclass
class Options:
    """User choices for the steps (defaults = the README commands)."""
    eyes: str | None = None          # None = both, 'open' or 'closed'
    jobs: int = max(1, (os.cpu_count() or 2) // 2)
    force: bool = False
    permutations: int = 5000
    ica_mode: str = 'original'
    data_dir: Path | None = None
    annotations_dir: Path | None = None


def command(step: str, options: Options, results_dir: Path) -> list[str]:
    """argv for one step, run with the same Python as the GUI."""
    if step == 'batch':
        argv = [sys.executable, '-m', 'batch.run_batch', '--out-dir', str(results_dir),
                '--jobs', str(max(1, options.jobs))]
        argv += ['--ica-mode', options.ica_mode]
        if options.data_dir is not None:
            argv += ['--data-dir', str(options.data_dir)]
        if options.annotations_dir is not None:
            argv += ['--annotations-dir', str(options.annotations_dir)]
        if options.eyes:
            argv += ['--eyes', options.eyes]
        if options.force:
            argv.append('--force')
        return argv
    if step == 'stats':
        return [sys.executable, '-m', 'analysis.stats', '--results-dir', str(results_dir),
                '--permutations', str(options.permutations)]
    if step == 'figures':
        return [sys.executable, '-m', 'analysis.figures', '--results-dir', str(results_dir)]
    raise ValueError(f'Paso desconocido: {step}')


def parse_progress(line: str) -> float | None:
    """Percentage from a batch progress line '[k/N] ...', or None for other lines."""
    match = PROGRESS.search(line)
    if not match:
        return None
    done, total = int(match[1]), int(match[2])
    return 100.0 * done / total if total else None


def missing_dependencies(step: str) -> list[str]:
    """Packages the step needs that are not installed (statistics and figures need pandas)."""
    needed = ['pandas'] if step in ('stats', 'figures') else []
    return [name for name in needed if importlib.util.find_spec(name) is None]


class Runner:
    """Runs steps one after another in a child process; events go to a queue read by the GUI.

    Events: ('start', step), ('line', step, text), ('progress', step, percent or None),
    ('done', step, returncode), ('finished', ok, cancelled).
    """

    def __init__(self, project: Path, results_dir: Path):
        self.project = Path(project)
        self.results_dir = Path(results_dir)
        self.events: queue.Queue = queue.Queue()
        self.process: subprocess.Popen | None = None
        self.cancelled = False
        self.thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, steps: list[str], options: Options) -> None:
        if self.running:
            raise RuntimeError('Ya hay un proceso en ejecución.')
        self.cancelled = False
        self.thread = threading.Thread(target=self._run, args=(list(steps), options), daemon=True)
        self.thread.start()

    def _run(self, steps: list[str], options: Options) -> None:
        ok = True
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'   # µ, −, … in log lines on Windows consoles
        env['MPLBACKEND'] = 'Agg'
        env['PYTHONUNBUFFERED'] = '1'      # stream log lines live instead of at the end
        for step in steps:
            if self.cancelled:
                break
            self.events.put(('start', step))
            try:
                self.process = subprocess.Popen(
                    command(step, options, self.results_dir), cwd=self.project, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                    errors='replace', bufsize=1,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            except OSError as exc:
                self.events.put(('line', step, f'No se pudo iniciar: {exc}'))
                self.events.put(('done', step, -1))
                ok = False
                break
            if self.cancelled:   # cancel pressed while the process was starting
                self._kill(self.process)
            for line in self.process.stdout:
                line = line.rstrip()
                self.events.put(('line', step, line))
                percent = parse_progress(line) if step == 'batch' else None
                if percent is not None:
                    self.events.put(('progress', step, percent))
            code = self.process.wait()
            self.process.stdout.close()
            self.events.put(('done', step, code))
            if code != 0:
                ok = False
                break
        self.process = None
        self.events.put(('finished', ok and not self.cancelled, self.cancelled))

    def cancel(self) -> None:
        """Stop the running step and do not start the next ones (child processes too)."""
        self.cancelled = True
        process = self.process
        if process is not None:
            self._kill(process)

    @staticmethod
    def _kill(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        if sys.platform == 'win32':
            # /T also ends the batch's worker processes (--jobs).
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        else:
            process.terminate()
