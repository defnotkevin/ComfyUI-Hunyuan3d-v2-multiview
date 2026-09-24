"""Run upstream painting in the optional CUDA environment; never imported by ComfyUI."""
import json
from pathlib import Path
import sys
import time


def main(job):
    sys.path.insert(0, job['source'])
    import torch
    import trimesh
    from PIL import Image
    from hy3dgen.texgen import Hunyuan3DPaintPipeline, Hunyuan3DTexGenConfig
    if not torch.cuda.is_available(): raise RuntimeError('Painting requires an NVIDIA CUDA GPU.')
    models = Path(job['models'])
    config = Hunyuan3DTexGenConfig(str(models/'hunyuan3d-delight-v2-0'), str(models/'hunyuan3d-paint-v2-0'), 'hunyuan3d-paint-v2-0')
    config.texture_size = job['resolution']
    started = time.monotonic()
    print('[Texture timing] Loading painting models...', flush=True)
    try:
        from .dynamic_loader import create_painter
    except ImportError:
        from dynamic_loader import create_painter
    pipeline = create_painter(Hunyuan3DPaintPipeline, config)
    print(f'[Texture timing] Model load: {time.monotonic()-started:.1f}s', flush=True)
    if any(k in job for k in ('remove_lighting','paint_steps','save_diagnostics')):
        try:
            from .fidelity import configure_fidelity
        except ImportError:
            from fidelity import configure_fidelity
        configure_fidelity(pipeline,job)
    mesh = trimesh.load(job['mesh'], force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces): raise ValueError('Input must contain a nonempty triangle mesh.')
    bounds = mesh.bounds.copy()
    images = [Image.open(p).convert('RGBA') for p in job['references']]
    if job.get('save_diagnostics',False):
        diagnostic=Path(job['output'])/'diagnostics';diagnostic.mkdir(parents=True,exist_ok=True)
        for index,image in enumerate(images):image.save(diagnostic/f'input_reference_{index:02d}.png')
        mesh.export(diagnostic/'unpainted.glb')
        (diagnostic/'settings.json').write_text(json.dumps({k:job.get(k) for k in ('resolution','paint_steps','remove_lighting','source_revision','model_revision')},indent=2))
    started = time.monotonic()
    print('[Texture timing] Painting (includes UV unwrap and baking)...', flush=True)
    painted = pipeline(mesh, image=images)
    if job.get('texture_mode','Generated') == 'Front reference projection (experimental)':
        try:
            from .projection import apply_front_projection
        except ImportError:
            from projection import apply_front_projection
        painted = apply_front_projection(pipeline,images[0],job)
    print(f'[Texture timing] Painting: {time.monotonic()-started:.1f}s', flush=True)
    import numpy as np
    if not np.allclose(bounds, painted.bounds, atol=1e-5): raise ValueError('Painting unexpectedly changed geometry bounds.')
    # Publish standard UV/base-color PBR rather than viewer-specific vertex colors.
    texture = painted.visual.material.image
    if job.get('save_diagnostics',False):texture.save(Path(job['output'])/'diagnostics/base_color.png')
    try:
        from .materials import apply_maps
    except ImportError:
        from materials import apply_maps
    apply_maps(painted, job.get('maps'), job.get('roughness',1.0), job.get('metallic',0.0), job.get('normal_convention','OpenGL (+Y)'), job['output'])
    output = Path(job['output'])
    painted.export(output/'textured.glb')
    # Portable fallback for DCC applications without a GLB importer.
    painted.visual.material = trimesh.visual.material.SimpleMaterial(image=texture, diffuse=[255,255,255,255])
    obj, resources = trimesh.exchange.obj.export_obj(painted, return_texture=True)
    (output/'textured.obj').write_text(obj)
    for name, data in resources.items():
        p = output / name
        if p.parent != output: raise ValueError('Unexpected texture export path.')
        p.write_bytes(data.encode() if isinstance(data, str) else data)
    print('Saved textured GLB and OBJ/MTL/texture files:', output, flush=True)

if __name__ == '__main__': main(json.loads(Path(sys.argv[1]).read_text()))
