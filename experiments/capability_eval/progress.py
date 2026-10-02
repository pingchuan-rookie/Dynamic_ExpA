"""Benchmark-labelled tqdm snapshots for non-interactive, concurrent job logs."""
import json
from pathlib import Path
import sys
import threading
import time


class BenchmarkProgress:
    def __init__(self, benchmark, work_dir, *, interval=10, stream=None):
        self.benchmark = benchmark
        self.path = Path(work_dir) / 'progress.json'
        self.interval = interval
        self.stream = stream if stream is not None else sys.stderr
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.thread = None

    def emit(self, phase):
        from tqdm import tqdm

        prefix = f'[capability_eval][{self.benchmark}]'
        try:
            state = json.loads(self.path.read_text())
            done, total = state['processed_count'], state['total_count']
            if type(done) is not int or type(total) is not int or not 0 <= done <= total or total <= 0:
                raise ValueError('Invalid progress counters')
        except (OSError, ValueError, KeyError, TypeError):
            line = f'{prefix} {phase}: preparing/loading dataset; sample count pending'
        else:
            elapsed = max(0, time.monotonic() - self.started)
            bar = tqdm.format_meter(done, total, elapsed, prefix=prefix, ascii=True,
                                    unit='sample', ncols=120,
                                    bar_format='{desc} {percentage:6.2f}%|{bar:20}| {n_fmt}/{total_fmt} '
                                               '[{elapsed}<{remaining}, {rate_fmt}]')
            line = f'{bar} phase={phase} tracker={state.get("status", "unknown")}'
        # One flushed line per snapshot: no carriage returns or terminal cursor moves.
        print(line, file=self.stream, flush=True)

    def watch(self):
        while not self.stop.wait(self.interval):
            self.emit('evaluating')

    def __enter__(self):
        self.emit('evaluating')
        self.thread = threading.Thread(target=self.watch, name=f'progress-{self.benchmark}', daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join()
        # Reaching all samples does not imply the reports have passed validation.
        self.emit('failed' if exc_type else 'samples_finished')
        return False
