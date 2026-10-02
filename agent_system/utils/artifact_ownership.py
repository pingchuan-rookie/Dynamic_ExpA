"""Best-effort ownership restoration limited to explicitly selected run directories."""
import os
from pathlib import Path
import stat
import sys


def restore_run_ownership(paths, environ):
    """Never follow links or mask the run's exit status, including on unsupported filesystems."""
    if not environ.get('HOST_UID') or not environ.get('HOST_GID'):
        return
    try:
        uid, gid = int(environ['HOST_UID']), int(environ['HOST_GID'])
        if uid < 0 or gid < 0:
            raise ValueError('HOST_UID and HOST_GID must be nonnegative')
    except (TypeError, ValueError) as exc:
        print(f'[ownership] skipped: {exc}', file=sys.stderr)
        return

    def visit(fd):
        metadata = os.fstat(fd)
        if stat.S_ISDIR(metadata.st_mode):
            for name in os.listdir(fd):
                try:
                    info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                    elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                        flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
                    else:
                        continue
                    child = os.open(name, flags, dir_fd=fd)
                    try:
                        visit(child)
                    finally:
                        os.close(child)
                except OSError as exc:
                    print(f'[ownership] skipped {name!r}: {exc}', file=sys.stderr)
        elif not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            return  # Hard-linked files may also belong to a historical/shared run.
        os.fchown(fd, uid, gid)

    for raw in paths:
        path = Path(raw).expanduser()
        if not path.is_absolute() or path == Path('/'):
            print(f'[ownership] skipped non-run path: {path}', file=sys.stderr)
            continue
        fd = None
        try:
            # Open each ancestor without following links, not only the last component.
            fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
            for part in path.parts[1:]:
                if part == '..':
                    raise ValueError('parent traversal is not a run path')
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child
            visit(fd)
        except (OSError, ValueError) as exc:
            print(f'[ownership] skipped {path}: {exc}', file=sys.stderr)
        finally:
            if fd is not None:
                os.close(fd)
