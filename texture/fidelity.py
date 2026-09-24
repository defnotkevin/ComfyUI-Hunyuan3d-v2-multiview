"""Instrument the pinned painter without editing upstream source or camera geometry."""
from pathlib import Path
import time


class CaptureCall:
    def __init__(self, wrapped, directory=None, prefix='image'):
        self.wrapped=wrapped; self.directory=directory; self.prefix=prefix; self.index=0
    def __call__(self,*args,**kwargs):
        started=time.monotonic();result=self.wrapped(*args,**kwargs)
        print(f'[Texture timing] {self.prefix}: {time.monotonic()-started:.1f}s',flush=True)
        if self.directory:
            for image in result if isinstance(result,list) else [result]:
                image.save(self.directory/f'{self.prefix}_{self.index:02d}.png');self.index+=1
        return result


class SamplingOverride:
    def __init__(self,pipeline,steps):self.pipeline=pipeline;self.steps=steps
    def __getattr__(self,name):return getattr(self.pipeline,name)
    def __call__(self,*args,**kwargs):
        kwargs['num_inference_steps']=self.steps
        return self.pipeline(*args,**kwargs)


def configure_fidelity(pipeline, job):
    steps=int(job.get('paint_steps',30))
    if not 1 <= steps <= 100:raise ValueError('paint_steps must be between 1 and 100.')
    directory=None
    if job.get('save_diagnostics',False):
        directory=Path(job['output'])/'diagnostics';directory.mkdir(parents=True,exist_ok=True)
    delight=pipeline.models['delight_model']
    if not job.get('remove_lighting',True):
        # Convert transparent references to white-background RGB, as the standard
        # delight output would; keep their recentered framing unchanged.
        def delight(image):
            from PIL import Image
            rgba=image.convert('RGBA');base=Image.new('RGBA',rgba.size,(255,255,255,255))
            return Image.alpha_composite(base,rgba).convert('RGB')
    pipeline.models['delight_model']=CaptureCall(delight,directory,'prepared_reference')
    multiview=pipeline.models['multiview_model']
    multiview.pipeline=SamplingOverride(multiview.pipeline,steps)
    pipeline.models['multiview_model']=CaptureCall(multiview,directory,'generated_view')
