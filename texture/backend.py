"""Lazy, process-locked setup for the optional Linux painting backend."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read_ready(root):
    try:
        config = json.loads((root/'texture_backend.json').read_text())
        if all(Path(config[k]).exists() for k in ('python','source','models')):
            return config
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def ensure_backend(root, interrupted=lambda: None):
    root = Path(root)
    config = read_ready(root)
    if config: return config
    if sys.platform != 'linux':
        raise RuntimeError('Automatic Hunyuan painting setup currently requires Linux with an NVIDIA CUDA GPU.')
    import fcntl
    # Lock is released by the OS even if ComfyUI exits during installation.
    with (root/'texture_setup.lock').open('a') as lock:
        while True:
            interrupted()
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                time.sleep(.5)
        config = read_ready(root)
        if config: return config
        destination = Path(os.environ.get('HUNYUAN_TEXTURE_BACKEND_DIR', str(root.parent.parent/'models/hunyuan-texture-backend'))).resolve()
        log_path = root/'texture_setup.log'
        print('[Hunyuan Texture] First-run setup: installing the isolated backend, compiling CUDA extensions, and downloading painting weights. This may take several minutes.', flush=True)
        print('[Hunyuan Texture] Backend: '+str(destination)+' | log: '+str(log_path), flush=True)
        env = os.environ.copy(); env.pop('PYTHONPATH', None); env['PYTHONUNBUFFERED']='1'
        # Setup must be online even if inference is configured for offline loading.
        env.pop('HF_HUB_OFFLINE',None); env.pop('TRANSFORMERS_OFFLINE',None)
        with log_path.open('w') as log:
            process = subprocess.Popen([sys.executable,'-u',str(root/'scripts/setup_texture.py'),'--backend-dir',str(destination)],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:
                with log_path.open() as progress:
                    while process.poll() is None:
                        interrupted()
                        for line in progress.readlines(): print('[Hunyuan Texture setup] '+line.rstrip(),flush=True)
                        time.sleep(.5)
                    for line in progress.readlines(): print('[Hunyuan Texture setup] '+line.rstrip(),flush=True)
            finally:
                if process.poll() is None:
                    import signal
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL); process.wait()
        if process.returncode:
            tail=log_path.read_text(errors='replace')[-4000:]
            raise RuntimeError('Automatic texture setup failed. Full log: '+str(log_path)+'\n'+tail)
        config = read_ready(root)
        if not config: raise RuntimeError('Setup finished without a valid backend configuration. See '+str(log_path))
        print('[Hunyuan Texture] Backend ready. Starting painting.',flush=True)
        return config
