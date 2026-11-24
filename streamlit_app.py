"""Cloud entry point supporting both original and browser-uploaded layouts."""
from pathlib import Path
import runpy


def prepare_data(root):
    root = Path(root)
    mappings = [
        (root / name, root / 'data' / name)
        for name in ('chunks.jsonl', 'chunks_20.jsonl',
                     'chunks_50.jsonl', 'chunks_100.jsonl')
    ]
    for variant in ('small', 'medium', 'large'):
        for name in ('bm25.pkl', 'dense.ids.json', 'dense.npy', 'meta.json'):
            mappings.append((root / variant / name,
                             root / 'data' / 'indexes' / variant / name))
    missing = []
    for source, target in mappings:
        if target.exists():
            continue
        if not source.is_file():
            missing.append(str(target.relative_to(root)))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.symlink_to(source.resolve())
        except FileExistsError:
            if not target.exists():
                raise
    if missing:
        raise FileNotFoundError('Missing uploaded files: ' + ', '.join(missing))


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    prepare_data(root)
    runpy.run_path(str(root / 'src' / 'app' / 'streamlit_app.py'),
                   run_name='__main__')
