"""Optional painting integration. Heavy dependencies stay in a separate process."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
import folder_paths

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ('front', 'left', 'back', 'right')


def reference_png(image, mask, convention, destination):
    import numpy as np
    import torch
    from PIL import Image
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3 or not torch.isfinite(image).all():
        raise ValueError('Each view must be one finite RGB IMAGE, not a batch.')
    rgb = image[0].detach().cpu().float().clamp(0, 1).numpy()
    h, w = rgb.shape[:2]
    alpha = np.ones((h, w), dtype=np.float32)
    if mask is not None:
        m = mask.detach().cpu().float()
        if m.ndim == 3 and m.shape[0] == 1: m = m[0]
        if m.ndim != 2 or not torch.isfinite(m).all():
            raise ValueError('Each mask must be one finite MASK.')
        if tuple(m.shape) != (h, w):
            if convention == 'white removes' and not torch.count_nonzero(m): m = torch.zeros((h, w))
            else: raise ValueError('Mask size must match its image.')
        alpha = m.clamp(0, 1).numpy()
        if convention == 'white removes': alpha = 1 - alpha
    if not (alpha > 0).any(): raise ValueError('Mask removes the entire reference image.')
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    Image.fromarray((rgba * 255).round().astype(np.uint8), 'RGBA').save(destination)


def map_png(image, destination):
    import torch
    from PIL import Image
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3 or not torch.isfinite(image).all():
        raise ValueError('Each material map must be one finite RGB IMAGE, not a batch.')
    data = image[0].detach().cpu().float().clamp(0, 1).numpy()
    Image.fromarray((data * 255).round().astype('uint8'), 'RGB').save(destination)


def material_inputs():
    return {'roughness': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
            'metallic': ('FLOAT', {'default': 0.0, 'min': 0.0, 'max': 1.0, 'step': 0.01}),
            'normal_convention': (['OpenGL (+Y)', 'DirectX (-Y)'],),
            **{name+'_map': ('IMAGE',) for name in ('roughness','metallic','normal')}}


def save_maps(optional, directory):
    maps = {}
    for name in ('roughness','metallic','normal'):
        image = optional.get(name+'_map')
        if image is not None:
            path = Path(directory)/(name+'.png'); map_png(image,path); maps[name]=str(path)
    return maps


def publish_asset(data, output_root):
    ident = hashlib.sha256(data).hexdigest()
    directory = Path(output_root) / 'world_viewer' / 'assets'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (ident + '.glb')
    if not target.exists():
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as f:
            temporary = Path(f.name); f.write(data)
        os.replace(temporary, target)
    return {'asset': ident, 'name': 'Textured Hunyuan object'}


class HunyuanAddTexture:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'mesh': ('MESH',), 'front_image': ('IMAGE',),
            'texture_resolution': ([1024, 2048], {'default': 2048}),
            'mask_convention': (['white removes', 'white keeps'],),
        }, 'optional': {
            'front_mask': ('MASK',), **material_inputs(),
            'remove_lighting': ('BOOLEAN', {'default': True}),
            'paint_steps': ('INT', {'default': 30, 'min': 1, 'max': 100}),
            'save_diagnostics': ('BOOLEAN', {'default': True}),
            'texture_mode': (['Generated', 'Front reference projection (experimental)'],),
            'projection_strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0, 'step': 0.05}),
            'projection_scale': ('FLOAT', {'default': 1.0, 'min': 0.5, 'max': 1.5, 'step': 0.01}),
            'projection_offset_x': ('FLOAT', {'default': 0.0, 'min': -0.25, 'max': 0.25, 'step': 0.005}),
            'projection_offset_y': ('FLOAT', {'default': 0.0, 'min': -0.25, 'max': 0.25, 'step': 0.005}),
            **{key: (kind,) for v in VIEWS[1:] for key, kind in [(v+'_image','IMAGE'),(v+'_mask','MASK')]}
        }}
    RETURN_TYPES = ('WORLD_ASSET', 'FILE_3D_GLB', 'STRING')
    RETURN_NAMES = ('world_asset', 'textured_glb', 'glb_path')
    FUNCTION = 'paint'
    CATEGORY = '3d/native hunyuan'

    def paint(self, mesh, front_image, texture_resolution=2048, mask_convention='white removes', **optional):
        import comfy.model_management as mm
        from .backend import ensure_backend
        mm.unload_all_models(); mm.soft_empty_cache()
        config = ensure_backend(ROOT, mm.throw_exception_if_processing_interrupted)
        if mesh.vertices.shape[0] != 1: raise ValueError('One mesh per texture run is supported.')
        from comfy_extras.nodes_save_3d import mesh_item_to_glb_bytes
        data = mesh_item_to_glb_bytes(mesh, 0)
        if not data: raise ValueError('Mesh is empty.')
        import comfy.model_management as mm
        from comfy_api.latest import Types
        output = Path(folder_paths.get_output_directory()) / 'hunyuan_textured' / uuid.uuid4().hex
        output.mkdir(parents=True)
        with tempfile.TemporaryDirectory(prefix='hunyuan-paint-') as tmp:
            temp = Path(tmp); (temp/'mesh.glb').write_bytes(data)
            refs = []
            for view in VIEWS:
                image = front_image if view == 'front' else optional.get(view+'_image')
                mask = optional.get(view+'_mask')
                if image is None:
                    if mask is not None: raise ValueError(view+' mask requires its image.')
                    continue
                path = temp / (view+'.png'); reference_png(image, mask, mask_convention, path); refs.append(str(path))
            if optional.get('texture_mode','Generated') != 'Generated':
                from PIL import Image
                from .projection import validate_reference
                validate_reference(Image.open(refs[0]))
            maps = save_maps(optional, temp)
            job = dict(config, remove_lighting=optional.get('remove_lighting',True), paint_steps=optional.get('paint_steps',30), save_diagnostics=optional.get('save_diagnostics',True), maps=maps, roughness=optional.get('roughness',1.0), metallic=optional.get('metallic',0.0), normal_convention=optional.get('normal_convention','OpenGL (+Y)'), mesh=str(temp/'mesh.glb'), references=refs, resolution=int(texture_resolution), output=str(output))
            for key, default in [('texture_mode','Generated'),('projection_strength',1.0),('projection_scale',1.0),('projection_offset_x',0.0),('projection_offset_y',0.0)]:
                job[key] = optional.get(key,default)
            (temp/'job.json').write_text(json.dumps(job))
            mm.unload_all_models(); mm.soft_empty_cache()
            env = os.environ.copy(); env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            env.pop('PYTHONPATH', None)
            # Isolate Python custom-code caches between workers, not model weights.
            env['HF_MODULES_CACHE'] = str(temp/'hf_modules')
            with (output/'paint.log').open('w') as log:
                process = subprocess.Popen([config['python'], str(ROOT/'texture/worker.py'), str(temp/'job.json')], stdout=log, stderr=subprocess.STDOUT, env=env)
                try:
                    while process.poll() is None:
                        mm.throw_exception_if_processing_interrupted(); time.sleep(.25)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try: process.wait(timeout=10)
                        except subprocess.TimeoutExpired: process.kill(); process.wait()
            if process.returncode:
                tail = (output/'paint.log').read_text(errors='replace')[-3500:]
                raise RuntimeError('Hunyuan painting failed. Log: '+str(output/'paint.log')+'\n'+tail)
        path = output/'textured.glb'; textured = path.read_bytes()
        asset = publish_asset(textured, folder_paths.get_output_directory())
        return (asset, Types.File3D(io.BytesIO(textured), file_format='glb'), str(path))


class HunyuanApplyPBRMaps:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'world_asset': ('WORLD_ASSET',)}, 'optional': material_inputs()}
    RETURN_TYPES = HunyuanAddTexture.RETURN_TYPES
    RETURN_NAMES = HunyuanAddTexture.RETURN_NAMES
    FUNCTION = 'apply'
    CATEGORY = '3d/native hunyuan'

    def apply(self, world_asset, **optional):
        import re
        import trimesh
        from comfy_api.latest import Types
        from .materials import apply_maps
        ident = world_asset.get('asset','')
        if not re.fullmatch('[0-9a-f]{64}', ident): raise ValueError('Invalid World Asset ID.')
        source = Path(folder_paths.get_output_directory())/'world_viewer/assets'/(ident+'.glb')
        scene = trimesh.load(source, force='scene', process=False)
        if len(scene.geometry) != 1: raise ValueError('Apply Maps currently supports a single mesh/material asset.')
        mesh = next(iter(scene.geometry.values()))
        output = Path(folder_paths.get_output_directory())/'hunyuan_textured'/uuid.uuid4().hex
        output.mkdir(parents=True)
        maps = save_maps(optional,output)
        apply_maps(mesh,maps,optional.get('roughness',1.0),optional.get('metallic',0.0),optional.get('normal_convention','OpenGL (+Y)'),output)
        path = output/'textured.glb'; scene.export(path); data = path.read_bytes()
        asset = publish_asset(data,folder_paths.get_output_directory()); asset['name']=world_asset.get('name',asset['name'])
        return (asset,Types.File3D(io.BytesIO(data),file_format='glb'),str(path))
