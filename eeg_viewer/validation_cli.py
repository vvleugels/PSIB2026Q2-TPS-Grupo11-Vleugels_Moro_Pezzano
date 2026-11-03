"""Reproducible pilot setup, review, profile freezing and evidence export."""
import argparse
import json
from pathlib import Path

from .validation_session import ValidationSession
from .validation_store import Revision, create_sample, export_new, freeze_profile

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    parser = argparse.ArgumentParser(description='Validación EEG experimental, separada de gui.py')
    commands = parser.add_subparsers(dest='command', required=True)
    sample = commands.add_parser('sample', help='Verificar y guardar la muestra estratificada de 12 registros')
    sample.add_argument('--data', type=Path, default=ROOT / 'data')
    sample.add_argument('--output', type=Path, default=ROOT / 'annotations/validation/sample.json')
    new = commands.add_parser('new', help='Crear revisión de 4 s sin heredar etiquetas anteriores')
    new.add_argument('--sample', type=Path, default=ROOT / 'annotations/validation/sample.json')
    new.add_argument('--annotations', type=Path, default=ROOT / 'annotations')
    new.add_argument('--record', required=True, help='Nombre del archivo .set, con o sin extensión')
    new.add_argument('--frozen', type=Path, help='Obligatorio para registros de evaluación')
    new.add_argument('--open', action='store_true', help='Abrir el visor inmediatamente después de crear la revisión')
    review = commands.add_parser('review', help='Abrir el visor local de validación')
    review.add_argument('revision', type=Path, help='Directorio que contiene revision.json')
    freeze = commands.add_parser('freeze', help='Congelar perfil tras revisar las seis calibraciones')
    freeze.add_argument('--sample', type=Path, default=ROOT / 'annotations/validation/sample.json')
    freeze.add_argument('--revisions', type=Path, nargs=6, required=True)
    freeze.add_argument('--output', type=Path, required=True)
    report = commands.add_parser('report', help='Exportar evidencia sin aprobar automáticamente el perfil')
    report.add_argument('revision', type=Path)
    report.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'sample':
            value = create_sample(args.data, args.output)
            print(f"Muestra guardada: {args.output}\n{len(value['records'])} registros verificados.")
            for r in value['records']:
                print(r['split'], Path(r['path']).stem,
                      f"{len(r['channels'])} canales; {r['n_times'] / r['sfreq']:g} s; "
                      f"{len(r['missing_positions'])} posiciones ausentes")
        elif args.command == 'new':
            value = json.loads(args.sample.read_text())
            matches = [r for r in value['records'] if Path(r['path']).stem == Path(args.record).stem]
            if len(matches) != 1:
                raise ValueError('El registro debe identificar exactamente un elemento de la muestra.')
            revision = Revision.create(args.annotations, matches[0], args.sample, frozen=args.frozen)
            print(revision.directory.absolute())
            if args.open:
                from .validation_ui import launch
                launch(revision)
        elif args.command == 'review':
            from .validation_ui import launch
            launch(Revision(args.revision))
        elif args.command == 'freeze':
            freeze_profile(args.sample, args.revisions, args.output)
            print(f'Perfil congelado para evaluación: {args.output}. No aprobado para integración.')
        elif args.command == 'report':
            session = ValidationSession(Revision(args.revision))
            try:
                value = session.report()
                value['interpolation_holdouts'] = session.holdouts()
                export_new(args.output, value)
            finally:
                session.close()
            print(f'Informe exportado: {args.output}')
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f'Error: {exc}\n')
