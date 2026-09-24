"""CPU integration checks. The neural painter is replaced with a textured fixture."""
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class TextureTests(unittest.TestCase):
    def setUp(self):
        self.stub=patch.dict(sys.modules,{'folder_paths':types.ModuleType('folder_paths')});self.stub.start()
        self.nodes=load('texture_nodes',ROOT/'texture/nodes.py')
    def tearDown(self): self.stub.stop()
    def test_ports(self):
        node=self.nodes.HunyuanAddTexture
        self.assertEqual(node.RETURN_TYPES,('WORLD_ASSET','FILE_3D_GLB','STRING'))
        self.assertIn('texture_mode',node.INPUT_TYPES()['optional'])
        self.assertIn('front_mask',node.INPUT_TYPES()['optional'])
    def test_world_asset_store_compatible(self):
        with tempfile.TemporaryDirectory() as temp:
            a=self.nodes.publish_asset(b'fixture',temp); b=self.nodes.publish_asset(b'fixture',temp)
            self.assertEqual(a,b)
            self.assertEqual((Path(temp)/'world_viewer/assets'/(a['asset']+'.glb')).read_bytes(),b'fixture')
    def test_worker_exports_uv_texture_and_keeps_bounds(self):
        import numpy as np
        import trimesh
        from PIL import Image
        worker=load('texture_worker',ROOT/'texture/worker.py')
        sys.path.insert(0,str(ROOT/'texture'))
        seen=[]
        class Config:
            def __init__(self,*args): pass
        class Painter:
            def __init__(self,config): self.config=config
            def __call__(self,mesh,image):
                seen.append(len(image))
                uv=np.zeros((len(mesh.vertices),2));uv[:,0]=np.linspace(0,1,len(mesh.vertices))
                mesh.visual=trimesh.visual.TextureVisuals(uv=uv,material=trimesh.visual.material.SimpleMaterial(image=Image.new('RGB',(8,8),'red')))
                return mesh
        backend=types.ModuleType('hy3dgen.texgen');backend.Hunyuan3DPaintPipeline=Painter;backend.Hunyuan3DTexGenConfig=Config
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:True))
        dynamic=types.SimpleNamespace(get_class_in_module=lambda *a:None)
        diffusers_utils=types.ModuleType('diffusers.utils');diffusers_utils.dynamic_modules_utils=dynamic
        with tempfile.TemporaryDirectory() as temp,patch.dict(sys.modules,{'diffusers':types.ModuleType('diffusers'),'diffusers.utils':diffusers_utils,'torch':torch,'hy3dgen':types.ModuleType('hy3dgen'),'hy3dgen.texgen':backend}):
            p=Path(temp);mesh=trimesh.creation.box();mesh.export(p/'in.glb')
            Image.new('RGBA',(8,8),'blue').save(p/'ref.png')
            for count in (1,4):
                worker.main(dict(source=temp,models=temp,mesh=str(p/'in.glb'),references=[str(p/'ref.png')]*count,resolution=1024,output=temp))
                result=trimesh.load(p/'textured.glb',force='mesh',process=False)
                self.assertTrue(np.allclose(result.bounds,mesh.bounds));self.assertIsNotNone(result.visual.uv)
                data=(p/'textured.glb').read_bytes();length=struct.unpack_from('<I',data,12)[0];doc=json.loads(data[20:20+length])
                self.assertIn('baseColorTexture',doc['materials'][0]['pbrMetallicRoughness'])
                self.assertIn('bufferView',doc['images'][0]);self.assertTrue((p/'textured.obj').exists())
                self.assertTrue(list(p.glob('*.mtl')));self.assertTrue(list(p.glob('*.png')))
            self.assertEqual(seen,[1,4])

if __name__=='__main__': unittest.main()
