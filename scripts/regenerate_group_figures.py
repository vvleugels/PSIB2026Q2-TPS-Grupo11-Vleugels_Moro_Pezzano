"""Regenerate absolute and relative group figures from the saved features CSV."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eeg_viewer.group_figures import regenerate_group_figures


def main(argv=None):
    parser = argparse.ArgumentParser(description='Regenerar figuras guardadas del análisis grupal.')
    parser.add_argument('--features', type=Path,
                        default=ROOT / 'results' / 'group20-channel-2026' / 'features.csv')
    parser.add_argument('--analysis', type=Path,
                        default=ROOT / 'results' / 'group20-channel-2026' / 'analysis')
    args = parser.parse_args(argv)
    try:
        paths = regenerate_group_figures(args.features, args.analysis)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f'Figuras generadas: {len(paths)}')
    for path in paths:
        print(path)


if __name__ == '__main__':
    main()
