"""Native graph wrapper targeting ComfyUI a8686f2b. No extra dependencies."""
import os
import folder_paths
from comfy_execution.graph_utils import GraphBuilder

SINGLE = '2.1 single-view'
MULTI = '2mv multi-view'
VIEWS = ('front', 'left', 'back', 'right')

class NativeHunyuanPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image': ('IMAGE',), 'mask_convention': (['white removes', 'white keeps'],),
                             'framing': (['unchanged', 'pad square'],),
                             'background': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0})},
                'optional': {'mask': ('MASK',)}}
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'prepare'
    CATEGORY = '3d/native hunyuan/utilities'

    def prepare(self, image, mask_convention, framing, background, mask=None):
        import torch
        import torch.nn.functional as F
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
            raise ValueError('Each view must contain exactly one RGB image; image batches are not views.')
        if not torch.isfinite(image).all():
            raise ValueError('Image contains non-finite values.')
        h, w = image.shape[1:3]
        out = image
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim != 3 or mask.shape[0] != 1 or not torch.isfinite(mask).all():
                raise ValueError('Each mask must be one finite [1,H,W] mask.')
            # Load Image emits a 64x64 zero mask for images without alpha.
            if tuple(mask.shape[1:]) != (h, w):
                if mask_convention == 'white removes' and torch.count_nonzero(mask) == 0:
                    mask = torch.zeros((1, h, w), device=image.device, dtype=image.dtype)
                else:
                    raise ValueError('Mask dimensions must match its image. Resize the mask explicitly.')
            mask = mask.to(device=image.device, dtype=image.dtype).clamp(0, 1)
            alpha = 1 - mask if mask_convention == 'white removes' else mask
            if not torch.any(alpha > 0):
                raise ValueError('Mask removes the entire image. Check mask convention.')
            out = image * alpha[..., None] + background * (1 - alpha[..., None])
        if framing == 'pad square':
            side = max(h, w)
            out = F.pad(out.movedim(-1, 1), ((side-w)//2, side-w-(side-w)//2,
                        (side-h)//2, side-h-(side-h)//2), value=background).movedim(1, -1)
        return (out,)

class NativeHunyuanImagesToMesh:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {'front_mask': ('MASK',), 'mesh_algorithm': (['surface net', 'basic'], {'default': 'surface net'})}
        for view in VIEWS[1:]:
            optional[view + '_image'] = ('IMAGE',)
            optional[view + '_mask'] = ('MASK',)
        return {'required': {
            'mode': ([SINGLE, MULTI],),
            'checkpoint': (folder_paths.get_filename_list('checkpoints'),),
            'front_image': ('IMAGE',),
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff, 'control_after_generate': True}),
            'steps': ('INT', {'default': 30, 'min': 1, 'max': 1000}),
            'cfg': ('FLOAT', {'default': 5.0, 'min': 0.0, 'max': 30.0}),
            'latent_resolution': ('INT', {'default': 4096, 'min': 1, 'max': 8192}),
            'octree_resolution': ('INT', {'default': 256, 'min': 16, 'max': 512}),
            'num_chunks': ('INT', {'default': 8000, 'min': 1000, 'max': 500000}),
            'threshold': ('FLOAT', {'default': 0.6, 'min': -1.0, 'max': 1.0, 'step': 0.01}),
            'mask_convention': (['white removes', 'white keeps'],),
            'framing': (['unchanged', 'pad square'],),
            'background': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.0}),
        }, 'optional': optional}
    RETURN_TYPES = ('MESH', 'VOXEL')
    RETURN_NAMES = ('mesh', 'voxel')
    FUNCTION = 'generate'
    CATEGORY = '3d/native hunyuan'
    DESCRIPTION = 'Native 2.1 single-view / standard 2mv multi-view shape generation. No textures. Four visible view slots; connect 1–4 in 2mv mode.'

    def generate(self, mode, checkpoint, front_image, seed, steps, cfg,
                 latent_resolution, octree_resolution, num_chunks, threshold,
                 mask_convention, framing, background, **optional):
        algorithm = optional.get('mesh_algorithm', 'surface net')
        if algorithm not in ('surface net', 'basic'):
            raise ValueError('Unknown mesh extraction algorithm.')
        if mode not in (SINGLE, MULTI):
            raise ValueError('Unknown generation mode.')
        # Strict names are deliberate: do not imply arbitrary checkpoints are compatible.
        expected = 'hunyuan_3d_v2.1.safetensors' if mode == SINGLE else 'hunyuan3d-dit-v2-mv.safetensors'
        allowed = {expected} if mode == SINGLE else {expected, 'hunyuan3d-dit-v2-mv_fp16.safetensors'}
        if os.path.basename(checkpoint) not in allowed:
            raise ValueError(f'{mode} requires the official standard checkpoint named {expected}. Fast/turbo variants are not supported.')
        images = {'front': front_image, **{v: optional.get(v+'_image') for v in VIEWS[1:]}}
        if front_image is None:
            raise ValueError('Connect a front image.')
        for view in VIEWS:
            if optional.get(view+'_mask') is not None and images[view] is None:
                raise ValueError(f'{view}: mask connected without an image.')
        if mode == SINGLE and any(images[v] is not None for v in VIEWS[1:]):
            raise ValueError('2.1 is single-view. Disconnect side/back images or select 2mv and its checkpoint.')
        g = GraphBuilder()
        loader = g.node('ImageOnlyCheckpointLoader', ckpt_name=checkpoint)
        model = g.node('ModelSamplingAuraFlow', model=loader.out(0), shift=1.0, sampling='flow')
        encoded = {}
        for view in VIEWS:
            if images[view] is None:
                continue
            args = dict(image=images[view], mask_convention=mask_convention, framing=framing, background=background)
            if optional.get(view+'_mask') is not None:
                args['mask'] = optional[view+'_mask']
            prepared = g.node('NativeHunyuanPrepare', **args)
            encoded[view] = g.node('CLIPVisionEncode', clip_vision=loader.out(1), image=prepared.out(0), crop=('center' if mode == SINGLE else 'none')).out(0)
        if mode == SINGLE:
            cond = g.node('Hunyuan3Dv2Conditioning', clip_vision_output=encoded['front'])
        else:
            cond = g.node('Hunyuan3Dv2ConditioningMultiView', **encoded)
        latent = g.node('EmptyLatentHunyuan3Dv2', resolution=latent_resolution, batch_size=1)
        samples = g.node('KSampler', model=model.out(0), seed=seed, steps=steps, cfg=cfg,
                         sampler_name='euler', scheduler='normal', positive=cond.out(0),
                         negative=cond.out(1), latent_image=latent.out(0), denoise=1.0)
        voxels = g.node('VAEDecodeHunyuan3D', samples=samples.out(0), vae=loader.out(2),
                        num_chunks=num_chunks, octree_resolution=octree_resolution)
        mesh = g.node('VoxelToMesh', voxel=voxels.out(0), algorithm=algorithm, threshold=threshold)
        return {'result': (mesh.out(0), voxels.out(0)), 'expand': g.finalize()}

NODE_CLASS_MAPPINGS = {'NativeHunyuanImagesToMesh': NativeHunyuanImagesToMesh,
                       'NativeHunyuanPrepare': NativeHunyuanPrepare}
NODE_DISPLAY_NAME_MAPPINGS = {'NativeHunyuanImagesToMesh': 'Hunyuan Images → Mesh (Native)',
                              'NativeHunyuanPrepare': 'Hunyuan Prepare Image + Mask'}

# Optional backend imports are lazy: installing this node does not install painting dependencies.
from .texture.nodes import HunyuanAddTexture
NODE_CLASS_MAPPINGS['Hunyuan3dv2AddTexture'] = HunyuanAddTexture
NODE_DISPLAY_NAME_MAPPINGS['Hunyuan3dv2AddTexture'] = 'Hunyuan3d-v2-Add-Texture'

from .texture.nodes import HunyuanApplyPBRMaps
NODE_CLASS_MAPPINGS['Hunyuan3dv2ApplyPBRMaps'] = HunyuanApplyPBRMaps
NODE_DISPLAY_NAME_MAPPINGS['Hunyuan3dv2ApplyPBRMaps'] = 'Hunyuan3d-v2-Apply-PBR-Maps'
