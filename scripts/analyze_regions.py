"""Summarize regional powers and paired NS/SD statistics from features.csv."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eeg_viewer.region_analysis import run_region_analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description='Estadística regional desde features.csv.')
    parser.add_argument('--features', type=Path,
                        default=ROOT / 'results' / 'group20-channel-2026' / 'features.csv')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results' / 'group20-channel-2026' / 'analysis')
    args = parser.parse_args(argv)
    try:
        result = run_region_analysis(args.features, args.output)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f"Entrada: {args.features}")
    print(f"Participantes: {len({row['participant'] for row in result['regional']})}")
    print(f"Filas de features: {result['feature_rows']}; filas regionales: {result['regional_rows']}; "
          f"canales: {result['channels']}")
    print(f"Resultados: {args.output}")


if __name__ == '__main__':
    main()
