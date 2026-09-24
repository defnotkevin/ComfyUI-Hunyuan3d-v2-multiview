import importlib.util
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
spec=importlib.util.spec_from_file_location('projection',Path(__file__).parents[1]/'texture/projection.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ProjectionTests(unittest.TestCase):
    def reference(self):
        a=np.zeros((100,100,4),dtype=np.uint8)
        a[20:80,35:65]=[255,0,0,255]
        return Image.fromarray(a)
    def test_uniform_alignment_preserves_proportions(self):
        target=np.zeros((200,200),bool);target[40:160,70:130]=True
        image,iou=m.align_reference(self.reference(),target)
        self.assertGreater(iou,.99)
        self.assertEqual(m.bounds(np.asarray(image)[...,3]>=128),(70,40,130,160))
    def test_pose_mismatch_is_not_stretched_to_fit(self):
        target=np.zeros((200,200),bool);target[40:160,20:180]=True
        _,iou=m.align_reference(self.reference(),target)
        self.assertLess(iou,.65)
    def test_opaque_and_empty_masks_rejected(self):
        for alpha in (0,255):
            with self.assertRaisesRegex(ValueError,'foreground mask'):
                m.validate_reference(Image.new('RGBA',(20,20),(255,255,255,alpha)))
    def test_unseen_grazing_and_transparent_pixels_keep_generated(self):
        base=np.full((1,3,3),.25)
        projected=np.array([[[1,0,0,1],[1,0,0,1],[0,0,0,0]]],dtype=float)
        cosine=np.array([[[0],[.4],[1]]])
        result,weight=m.blend_projection(base,projected,cosine,1)
        np.testing.assert_allclose(result,base)
        self.assertFalse(weight.any())
    def test_premultiplied_color_and_strength(self):
        base=np.zeros((1,1,3))
        projected=np.array([[[.5,0,0,.5]]])
        result,weight=m.blend_projection(base,projected,np.ones((1,1,1)),.5)
        np.testing.assert_allclose(result,[[[.25,0,0]]])
        np.testing.assert_allclose(weight,.25)
    def test_controls_and_empty_target(self):
        with self.assertRaises(ValueError):m.align_reference(self.reference(),np.zeros((20,20),bool))
        with self.assertRaises(ValueError):m.align_reference(self.reference(),np.ones((20,20),bool),scale=float('nan'))
