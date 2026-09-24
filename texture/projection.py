"""Experimental front-view silhouette alignment and visibility-aware UV projection.

No landmark registration is claimed: pose and camera must already approximately match.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter


def bounds(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError('Projection requires a nonempty foreground silhouette.')
    return int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)


def validate_reference(image):
    alpha = np.asarray(image.convert('RGBA'))[..., 3]
    if not (alpha < 128).any() or not (alpha >= 128).any():
        raise ValueError('Front reference projection requires a foreground mask. Connect front_mask with the correct mask convention; a fully opaque image cannot be aligned safely.')


def align_reference(image, silhouette, scale=1.0, offset_x=0.0, offset_y=0.0):
    validate_reference(image)
    if not all(np.isfinite(v) for v in (scale, offset_x, offset_y)) or not 0.5 <= scale <= 1.5 or max(abs(offset_x),abs(offset_y)) > .25:
        raise ValueError('Invalid projection alignment controls.')
    rgba = image.convert('RGBA')
    source = np.asarray(rgba)[..., 3] >= 128
    sx0, sy0, sx1, sy1 = bounds(source)
    tx0, ty0, tx1, ty1 = bounds(silhouette)
    # Preserve proportions; fit height, then center. Width differences remain
    # visible in the IoU score instead of silently stretching facial features.
    ratio = (ty1-ty0)/(sy1-sy0)*scale
    width, height = max(1, round((sx1-sx0)*ratio)), max(1, round((sy1-sy0)*ratio))
    crop = rgba.crop((sx0,sy0,sx1,sy1)).resize((width,height), Image.Resampling.LANCZOS)
    h,w = silhouette.shape
    left = round((tx0+tx1-width)/2+offset_x*w)
    top = round((ty0+ty1-height)/2+offset_y*h)
    aligned = Image.new('RGBA',(w,h))
    aligned.paste(crop,(left,top))
    a = np.asarray(aligned)[...,3] >= 128
    iou = float((a & silhouette).sum()/max(1,(a | silhouette).sum()))
    return aligned, iou


def blend_projection(base, projected, cosine, strength):
    # Projected RGBA carries premultiplied color, avoiding background bleed.
    alpha = np.clip(projected[...,3:4],0,1)
    color = projected[...,:3]/np.maximum(alpha,1e-8)
    confidence = np.clip((cosine-.5)/.35,0,1)*alpha*strength
    return np.clip(base*(1-confidence)+color*confidence,0,1), confidence


def apply_front_projection(pipeline, reference, job):
    import torch
    render = pipeline.render
    size = int(pipeline.config.render_size)
    with torch.no_grad():
        _, clip = render.get_pos_from_mvp(0,0,None,None)
        raster,_ = render.raster_rasterize(clip,render.pos_idx,resolution=[size,size])
        silhouette = (raster[0,...,-1]>0).cpu().numpy()
        aligned, iou = align_reference(reference,silhouette,
            job.get('projection_scale',1.0),job.get('projection_offset_x',0.0),job.get('projection_offset_y',0.0))
        diagnostic = Path(job['output'])/'diagnostics'
        diagnostic.mkdir(parents=True,exist_ok=True)
        aligned.save(diagnostic/'projection_aligned_front.png')
        overlay = np.asarray(aligned.convert('RGB')).copy()
        overlay[silhouette] = (overlay[silhouette]*.6+np.array([0,180,255])*.4).astype('uint8')
        Image.fromarray(overlay).save(diagnostic/'projection_alignment_overlay.png')
        report = {'mode':'front silhouette fit','silhouette_iou':iou,'minimum_iou':.65,
                  'note':'Silhouette overlap does not verify facial landmarks or perspective.'}
        (diagnostic/'projection.json').write_text(json.dumps(report,indent=2))
        if iou < .65:
            raise ValueError(f'Front projection alignment is poor (silhouette IoU {iou:.2f} < 0.65). Inspect {diagnostic}/projection_alignment_overlay.png; correct the mask, pose or alignment controls, or select Generated mode.')
        rgba = np.asarray(aligned,dtype=np.float32)/255
        # Erode/feather the reference boundary; renderer separately rejects
        # occluded surfaces, depth boundaries and grazing angles.
        a = aligned.getchannel('A').filter(ImageFilter.MinFilter(9)).filter(ImageFilter.GaussianBlur(2))
        rgba[...,3] *= np.asarray(a,dtype=np.float32)/255
        rgba[...,3] *= silhouette
        rgba[...,:3] *= rgba[...,3:4]
        projected,cosine,_ = render.back_project(rgba,0,0)
        base = render.get_texture()
        merged, confidence = blend_projection(base,projected.cpu().numpy(),cosine.cpu().numpy(),job.get('projection_strength',1.0))
        render.set_texture(torch.tensor(merged,device=render.device,dtype=torch.float32))
        Image.fromarray((confidence[...,0]*255).astype('uint8')).save(diagnostic/'projection_uv_coverage.png')
        print(f'[Projection] Front silhouette IoU {iou:.3f}; reference projection applied.',flush=True)
        return render.save_mesh()
