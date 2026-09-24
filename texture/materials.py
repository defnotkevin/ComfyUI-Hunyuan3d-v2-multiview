"""UV-space PBR maps, using glTF linear-data channel conventions."""
import math
from pathlib import Path


def apply_maps(mesh, maps=None, roughness=1.0, metallic=0.0, normal_convention='OpenGL (+Y)', output=None):
    import numpy as np
    from PIL import Image
    from trimesh.visual.material import PBRMaterial
    maps = maps or {}
    for name, value in [('roughness', roughness), ('metallic', metallic)]:
        if not math.isfinite(value) or not 0 <= value <= 1: raise ValueError(name+' must be between 0 and 1.')
    if normal_convention not in ('OpenGL (+Y)', 'DirectX (-Y)'): raise ValueError('Unknown normal convention.')
    uv = getattr(mesh.visual, 'uv', None)
    if maps and (uv is None or len(uv) != len(mesh.vertices) or not np.isfinite(uv).all()):
        raise ValueError('Material maps require valid UV coordinates on this mesh.')
    old = mesh.visual.material
    material = old.copy() if isinstance(old, PBRMaterial) else PBRMaterial(baseColorTexture=getattr(old, 'image', None))
    rough = Image.open(maps['roughness']).convert('RGB').getchannel('R') if maps.get('roughness') else None
    metal = Image.open(maps['metallic']).convert('RGB').getchannel('R') if maps.get('metallic') else None
    if rough and metal and rough.size != metal.size:
        raise ValueError('Roughness and metallic maps must have the same dimensions; resize explicitly.')
    packed = None
    if rough or metal:
        size = (rough or metal).size
        rough = rough or Image.new('L', size, round(roughness*255))
        metal = metal or Image.new('L', size, round(metallic*255))
        packed = Image.merge('RGB', (Image.new('L', size, 255), rough, metal))
        material.metallicRoughnessTexture = packed
        material.roughnessFactor = 1.0; material.metallicFactor = 1.0
    else:
        material.metallicRoughnessTexture = None
        material.roughnessFactor = roughness; material.metallicFactor = metallic
    normal = None
    if maps.get('normal'):
        normal = np.array(Image.open(maps['normal']).convert('RGB'), dtype=np.uint8)
        if normal_convention == 'DirectX (-Y)': normal[..., 1] = 255 - normal[..., 1]
        normal = Image.fromarray(normal, 'RGB'); material.normalTexture = normal
    mesh.visual.material = material
    if output:
        import json
        out = Path(output); out.mkdir(parents=True, exist_ok=True)
        for name, image in [('roughness', rough), ('metallic', metal), ('metallic_roughness', packed), ('normal_opengl', normal)]:
            if image is not None: image.save(out/(name+'.png'))
        (out/'pbr_material.json').write_text(json.dumps(dict(roughness_factor=material.roughnessFactor, metallic_factor=material.metallicFactor,
            packed_channels='R unused; G roughness; B metallic', normal_convention='OpenGL (+Y)',
            maps={k:v+'.png' for k,v in [('roughness','roughness'),('metallic','metallic'),('normal','normal_opengl')] if (out/(v+'.png')).exists()}), indent=2))
    return mesh
