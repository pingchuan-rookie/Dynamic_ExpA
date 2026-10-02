"""Shared persistent output locations, independent of datasets and the working directory."""

import argparse
import os
from pathlib import Path


def run_site(environ=None):
    environ = os.environ if environ is None else environ
    site = environ.get('RUN_SITE') or 'local'
    if site not in ('local', 'lucia'):
        raise ValueError('RUN_SITE must be local or lucia')
    return site


def artifact_root(project=None, environ=None):
    environ = os.environ if environ is None else environ
    project = Path(project).resolve() if project is not None else Path(__file__).resolve().parents[2]
    site = run_site(environ)
    explicit = environ.get('ARTIFACT_ROOT')
    if explicit:
        root = Path(explicit).expanduser()
        if not root.is_absolute():
            raise ValueError('ARTIFACT_ROOT must be absolute')
        return root.resolve()
    if site == 'lucia':
        data_root = environ.get('DATA_ROOT')
        if not data_root or not Path(data_root).expanduser().is_absolute():
            raise ValueError('Cluster runs require an absolute DATA_ROOT or ARTIFACT_ROOT')
        return (Path(data_root).expanduser() / 'artifacts').resolve()
    return project / 'artifacts'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path)
    parser.add_argument('--kind', choices=('root', 'ckpt', 'outputs'), default='root')
    args = parser.parse_args()
    try:
        root = artifact_root(args.project)
    except ValueError as exc:
        parser.error(str(exc))
    print(root if args.kind == 'root' else root / args.kind)


if __name__ == '__main__':
    main()
