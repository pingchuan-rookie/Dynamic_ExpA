"""Own the Alignment training process and persist its configuration, console log and exit status."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def training(values):
    from agent_system.utils.artifact_ownership import restore_run_ownership
    env = {**os.environ, **values}
    try:
        return _training(values)
    finally:
        try:
            restore_run_ownership([values['OUT'], values['CHECKPOINT_DIR']], env)
        except Exception as exc:
            print(f'[ownership] cleanup failed: {exc}', file=sys.stderr)


def _training(values):
    out = Path(values['OUT'])
    out.mkdir(parents=True, exist_ok=True)
    Path(values['CHECKPOINT_DIR']).mkdir(parents=True, exist_ok=True)
    for key in ('WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR', 'WANDB_DATA_DIR'):
        Path(values[key]).mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **values, 'PYTHON': os.environ.get('PYTHON', sys.executable), 'PYTHONUNBUFFERED': '1'}
    command = ['bash', str(Path(__file__).with_name('actenc_alignment_training.sh'))]
    (out / 'resolved_config.json').write_text(json.dumps(values, indent=2) + '\n')
    # Only selected non-secret values are recorded, never the inherited environment.
    (out / 'command.json').write_text(json.dumps({'command': command, 'python': env['PYTHON']}, indent=2) + '\n')
    pointer = env.get('ALIGNMENT_RUN_DIR_FILE')
    if pointer:
        Path(pointer).write_text(str(out) + '\n')
    process = None
    class Interrupted(Exception):
        def __init__(self, signum):
            self.signum = signum

    def forward(signum, frame):
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signum)
        raise Interrupted(signum)
    previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
    code = 1
    try:
        with (out / 'trainer.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, start_new_session=True)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
            code = process.wait()
            if code == 0 and values.get('ALIGNMENT_POST_UPLOAD') == '1':
                analysis = Path(values['PROJECT']) / 'experiments/dyad_training/action_encoder_alignment/analysis'
                for script in ('actenc_alignment_upload_wandb.py', 'actenc_alignment_verify_wandb.py'):
                    process = subprocess.Popen(
                        [env['PYTHON'], str(analysis / script), '--runs-dir', str(out.parent), '--run', out.name],
                        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, start_new_session=True)
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end='', flush=True)
                    code = process.wait()
                    if code:
                        break
    except Interrupted as exc:
        code = 128 + exc.signum
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        (out / 'exit_status.json').write_text(json.dumps({'returncode': code}) + '\n')
    return code if code >= 0 else 128 - code
