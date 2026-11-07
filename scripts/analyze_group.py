"""Command-line runner for group analysis of ds004902."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eeg_viewer.group_analysis import run_group_analysis
from eeg_viewer.processing import FilterConfig
from eeg_viewer.screening import ScreeningConfig
from eeg_viewer.spectral import SpectralConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description='Features EEG grupales para ds004902.')
    parser.add_argument('--data', type=Path, default=ROOT / 'data', help='Raíz BIDS (default: data/).')
    parser.add_argument('--output', type=Path, help='Carpeta nueva de salida; por defecto results/group-analysis-<UTC>.')
    parser.add_argument('--max-participants', type=int, help='Limita el número de participantes de la lista ordenada.')
    parser.add_argument('--participants', nargs='+', help='IDs a procesar, p. ej. sub-11 sub-12 sub-13.')
    parser.add_argument('--high-uv', type=float, default=150.0, help='Umbral pico a pico alto en µV (estricto >).')
    parser.add_argument('--flat-uv', type=float, default=1.0, help='Umbral pico a pico plano en µV (estricto <).')
    args = parser.parse_args(argv)
    try:
        result = run_group_analysis(
            data_dir=args.data, output_dir=args.output,
            max_participants=args.max_participants, participants=args.participants,
            screening_config=ScreeningConfig(high_uv=args.high_uv, flat_uv=args.flat_uv),
            filter_config=FilterConfig(enabled=True, low=0.2, high=45.0),
            spectral_config=SpectralConfig(),
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f"Salida: {result['output_dir']}")
    print(f"Filas de features: {len(result['features'])}")
    print(f"Registros resumidos: {len(result['summaries'])}")
    print(f"Exclusiones: {len(result['exclusions'])}")
    for row in result['summaries']:
        discarded = int(row['epochs_total'] or 0) - int(row['epochs_used'] or 0)
        print(f"{row['participant']} {row['session']} {row['eyes']}: "
              f"usadas={row['epochs_used']} descartadas={discarded} ({row['status']})")


if __name__ == '__main__':
    main()
