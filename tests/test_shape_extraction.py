"""Verify native graph expansion without loading torch or checkpoints."""
import ast
import os
from pathlib import Path
import types
import unittest

class Graph:
    def __init__(self):self.nodes={}
    def node(self,kind,**kwargs):
        ident=str(len(self.nodes));self.nodes[ident]={'class_type':kind,'inputs':kwargs}
        return types.SimpleNamespace(out=lambda slot:[ident,slot])
    def finalize(self):return self.nodes

class ExtractionTests(unittest.TestCase):
    def setUp(self):
        tree=ast.parse((Path(__file__).parents[1]/'__init__.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NativeHunyuanImagesToMesh')
        env=dict(os=os,GraphBuilder=Graph,SINGLE='2.1 single-view',MULTI='2mv multi-view',VIEWS=('front','left','back','right'),folder_paths=types.SimpleNamespace(get_filename_list=lambda _:[]))
        exec(compile(ast.Module(body=[node],type_ignores=[]),'shape','exec'),env)
        self.cls=env['NativeHunyuanImagesToMesh']
        self.args=dict(mode='2mv multi-view',checkpoint='hunyuan3d-dit-v2-mv_fp16.safetensors',front_image='front',left_image='left',seed=0,steps=30,cfg=5,latent_resolution=4096,octree_resolution=256,num_chunks=8000,threshold=.6,mask_convention='white removes',framing='unchanged',background=1)
    def test_legacy_default_and_shared_volume(self):
        result=self.cls().generate(**self.args);graph=result['expand']
        mesh_link,voxel_link=result['result']
        self.assertEqual(graph[mesh_link[0]]['inputs']['algorithm'],'surface net')
        self.assertEqual(graph[mesh_link[0]]['inputs']['voxel'],voxel_link)
        self.assertEqual(graph[voxel_link[0]]['class_type'],'VAEDecodeHunyuan3D')
        self.assertEqual(self.cls.RETURN_TYPES,('MESH','VOXEL'))
    def test_only_extractor_changes(self):
        a=self.cls().generate(**self.args)['expand']
        b=self.cls().generate(**self.args,mesh_algorithm='basic')['expand']
        differences=[k for k in a if a[k]!=b[k]]
        self.assertEqual(len(differences),1)
        self.assertEqual(b[differences[0]]['class_type'],'VoxelToMesh')
        self.assertEqual(b[differences[0]]['inputs']['algorithm'],'basic')
    def test_invalid_algorithm_rejected(self):
        with self.assertRaisesRegex(ValueError,'extraction algorithm'):
            self.cls().generate(**self.args,mesh_algorithm='unknown')
