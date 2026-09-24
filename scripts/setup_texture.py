"""Explicit, optional Linux/CUDA setup. Run with the Python used by ComfyUI."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE_REVISION = 'f8db63096c8282cb27354314d896feba5ba6ff8a'
MODEL_REVISION = '9cd649ba6913f7a852e3286bad86bfa9a2d83dcf'
ROOT = Path(__file__).resolve().parents[1]


def run(args, cwd=None):
    env = os.environ.copy()
    # RunPod may enable an optional downloader absent from this environment.
    env['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    subprocess.run([str(x) for x in args], cwd=cwd, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-dir', default='/workspace/hunyuan-texture-backend')
    args = parser.parse_args()
    if sys.platform != 'linux': raise SystemExit('This experimental backend currently targets Linux CUDA (RunPod).')
    import torch
    if not torch.cuda.is_available(): raise SystemExit('Run using your CUDA-enabled ComfyUI Python on the GPU pod.')
    for executable in ('git', 'nvcc', 'g++'):
        if not shutil.which(executable): raise SystemExit('Missing '+executable+'. Use a CUDA development image with a C++ compiler.')
    backend = Path(args.backend_dir).resolve(); backend.mkdir(parents=True, exist_ok=True)
    environment = backend/'venv'; python = environment/'bin/python'; source = backend/'Hunyuan3D-2'; models = backend/'models'
    if not python.exists(): run([sys.executable, '-m', 'venv', '--system-site-packages', environment])
    # A separate venv shares the working CUDA torch, while overrides stay outside ComfyUI.
    run([python, '-m', 'pip', 'install', 'pip>=24', 'setuptools<81', 'wheel', 'ninja', 'pybind11',
         'numpy<2', 'Pillow', 'scipy', 'diffusers==0.32.2', 'transformers==4.48.3',
         'huggingface-hub==0.28.1', 'accelerate==1.3.0', 'einops', 'omegaconf',
         'opencv-python-headless==4.10.0.84', 'trimesh==4.6.4', 'xatlas==0.0.9', 'pygltflib==1.16.3'])
    if not source.exists(): run(['git','clone','https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git',source])
    current = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if current != SOURCE_REVISION:
        if subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip():
            raise SystemExit('Upstream source has local edits. Use a new --backend-dir to avoid overwriting them.')
        run(['git','checkout',SOURCE_REVISION],cwd=source)
    for part in ('custom_rasterizer', 'differentiable_renderer'):
        run([python,'-m','pip','install','--no-build-isolation','--no-deps','.'],cwd=source/'hy3dgen/texgen'/part)
    # Fail before large downloads if any runtime import is unavailable.
    run([python,'-c','import torch; import sys; sys.path.insert(0,'+repr(str(source))+'); import custom_rasterizer, mesh_processor; from hy3dgen.texgen import Hunyuan3DPaintPipeline; print("Backend imports passed")'])
    code = ('from huggingface_hub import snapshot_download; snapshot_download('
            'repo_id="tencent/Hunyuan3D-2", revision='+repr(MODEL_REVISION)+', local_dir='+repr(str(models))+', '
            'allow_patterns=["hunyuan3d-delight-v2-0/*","hunyuan3d-paint-v2-0/*"])')
    run([python,'-c',code])
    config = dict(python=str(python), source=str(source), models=str(models), source_revision=SOURCE_REVISION, model_revision=MODEL_REVISION)
    temp = ROOT/'texture_backend.json.tmp'; temp.write_text(json.dumps(config,indent=2)); os.replace(temp,ROOT/'texture_backend.json')
    print('Texture backend configured. Automatic setup will now continue to painting.')

if __name__ == '__main__': main()
