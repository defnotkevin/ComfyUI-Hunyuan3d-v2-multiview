import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import struct
import numpy as np
from PIL import Image
import trimesh

spec=importlib.util.spec_from_file_location('materials',Path(__file__).parents[1]/'texture/materials.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def mesh():
    obj=trimesh.creation.box()
    obj.visual=trimesh.visual.TextureVisuals(uv=np.zeros((len(obj.vertices),2)),material=trimesh.visual.material.SimpleMaterial(image=Image.new('RGB',(4,4),'red')))
    return obj

class MaterialTests(unittest.TestCase):
    def test_fallback_factors(self):
        material=m.apply_maps(mesh(),roughness=.4,metallic=.7).visual.material
        self.assertAlmostEqual(material.roughnessFactor,.4);self.assertAlmostEqual(material.metallicFactor,.7)
        self.assertIsNone(material.metallicRoughnessTexture)
    def test_pack_flip_export_and_uv_preservation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            Image.new('RGB',(4,4),(64,64,64)).save(p/'r.png')
            Image.new('RGB',(4,4),(192,192,192)).save(p/'m.png')
            Image.new('RGB',(4,4),(128,80,240)).save(p/'n.png')
            obj=mesh();uv=obj.visual.uv.copy();vertices=obj.vertices.copy()
            m.apply_maps(obj,{'roughness':p/'r.png','metallic':p/'m.png','normal':p/'n.png'},normal_convention='DirectX (-Y)',output=p)
            self.assertEqual(obj.visual.material.metallicRoughnessTexture.getpixel((0,0)),(255,64,192))
            self.assertEqual(obj.visual.material.normalTexture.getpixel((0,0)),(128,175,240))
            self.assertTrue(np.array_equal(uv,obj.visual.uv));self.assertTrue(np.array_equal(vertices,obj.vertices))
            data=obj.export(file_type='glb');length=struct.unpack_from('<I',data,12)[0];doc=json.loads(data[20:20+length]);mat=doc['materials'][0]
            self.assertIn('normalTexture',mat);self.assertIn('metallicRoughnessTexture',mat['pbrMetallicRoughness'])
            self.assertTrue(all('bufferView' in x for x in doc['images']))
    def test_single_map_uses_other_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'r.png';Image.new('RGB',(4,4),(90,90,90)).save(p)
            mat=m.apply_maps(mesh(),{'roughness':p},metallic=.2).visual.material
            self.assertEqual(mat.metallicRoughnessTexture.getpixel((0,0)),(255,90,51))
    def test_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);Image.new('RGB',(4,4)).save(p/'r.png');Image.new('RGB',(8,8)).save(p/'m.png')
            with self.assertRaisesRegex(ValueError,'same dimensions'):m.apply_maps(mesh(),{'roughness':p/'r.png','metallic':p/'m.png'})
    def test_invalid_factor(self):
        with self.assertRaises(ValueError):m.apply_maps(mesh(),roughness=float('nan'))

if __name__=='__main__':unittest.main()
