import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from PIL import Image
spec=importlib.util.spec_from_file_location('fidelity',Path(__file__).parents[1]/'texture/fidelity.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class FidelityTests(unittest.TestCase):
    def test_sampling_overrides_only_steps(self):
        seen={}
        def call(*args,**kwargs):seen.update(kwargs);return 'result'
        self.assertEqual(m.SamplingOverride(call,40)(num_inference_steps=30,width=512),'result')
        self.assertEqual(seen,{'num_inference_steps':40,'width':512})
    def test_bypass_and_diagnostic_capture(self):
        class Multi:
            pipeline=lambda *a,**k:None
            def __call__(self,*a,**k):return [Image.new('RGB',(4,4),'red')]
        def forbidden(image):raise AssertionError('delight should be bypassed')
        p=types.SimpleNamespace(models={'delight_model':forbidden,'multiview_model':Multi()})
        with tempfile.TemporaryDirectory() as d:
            m.configure_fidelity(p,dict(output=d,remove_lighting=False,save_diagnostics=True))
            result=p.models['delight_model'](Image.new('RGBA',(4,4),(0,0,0,0)))
            self.assertEqual(result.getpixel((0,0)),(255,255,255))
            p.models['multiview_model']()
            self.assertEqual(len(list((Path(d)/'diagnostics').glob('*.png'))),2)
    def test_original_delight_used(self):
        calls=[]
        p=types.SimpleNamespace(models={'delight_model':lambda image:calls.append(image) or image,'multiview_model':types.SimpleNamespace(pipeline=None)})
        m.configure_fidelity(p,{})
        image=Image.new('RGB',(2,2));self.assertIs(p.models['delight_model'](image),image);self.assertEqual(len(calls),1)
