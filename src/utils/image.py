import numpy as np
from .spectrum import denorm_spectrum
import torch
from torch.nn.functional import interpolate

def image_to_spectrum(imgs, infos):
    outputs = {}
    for info, img in zip(infos, imgs):
        inp, tgt, vmin, vmax, x = info
        tag = (inp, tgt)
        if tag not in outputs.keys():
            outputs[tag] = {'x':[], 'y':[]}
        outputs[tag]['x'].append(x)
        outputs[tag]['y'].append(denorm_spectrum(img.reshape(-1), vmin, vmax))
        
    output_mats = {}
    for tag, v in outputs.items():
        x = np.array(v['x'])
        y = np.array(v['y'])
        i = np.argsort(x)
        output_mats[tag] = (x[i], y[i])

    return output_mats

def augment_image(img_inp, img_tgt, img_bic, img_nn=None, img_flow=None, flip_h=True, flip_v=True):
    if flip_h and np.random.rand() < 0.5:
        img_inp = torch.flip(img_inp, dims=[-1])
        img_tgt = torch.flip(img_tgt, dims=[-1])
        img_bic = torch.flip(img_bic, dims=[-1])
        if img_nn is not None:
            img_nn = torch.flip(img_nn, dims=[-1])
        if img_flow is not None:
            img_flow = torch.flip(img_flow, dims=[-1])
    if flip_v and np.random.rand() < 0.5:
        img_inp = torch.flip(img_inp, dims=[-2])
        img_tgt = torch.flip(img_tgt, dims=[-2])
        img_bic = torch.flip(img_bic, dims=[-2])
        if img_nn is not None:
            img_nn = torch.flip(img_nn, dims=[-2])
        if img_flow is not None:
            img_flow = torch.flip(img_flow, dims=[-2])

    return img_inp, img_tgt, img_bic, img_nn, img_flow

def convert_to_image(inp, tgt, upscale_factor, channels=3):  
    # generate channel axis (batch, channels, R, R)
    if inp is not None:
        R1 = inp.shape[-1]
#        inp = stack_spectrum(inp, channels=channels, channel_stride=channel_stride)
    if tgt is not None:
        R2 = tgt.shape[-1]
#        tgt = stack_spectrum(tgt, channels=channels, channel_stride=channel_stride)
        
    # make bicubic and convert to tensor [0,1]
    if inp is None:
        R1 = R2 // upscale_factor
        tgt_imgs = torch.from_numpy(tgt).float().unsqueeze(1)
        inp_imgs = interpolate(tgt_imgs, scale_factor=1/upscale_factor, mode='bicubic')
        bic_imgs = interpolate(inp_imgs, scale_factor=upscale_factor, mode='bicubic')
    elif tgt is None:
        R2 = R1 * upscale_factor
        inp_imgs = torch.from_numpy(inp).float().unsqueeze(1)
        bic_imgs = interpolate(inp_imgs, scale_factor=upscale_factor, mode='bicubic')
        tgt_imgs = bic_imgs
    else:
        inp_imgs = torch.from_numpy(inp).float().unsqueeze(1)
        bic_imgs = interpolate(inp_imgs, scale_factor=upscale_factor, mode='bicubic')
        tgt_imgs = torch.from_numpy(tgt).float().unsqueeze(1)
    
    # check
    if R1 * upscale_factor != R2:
        print('Warning: upscale factor mismatch.', upscale_factor, R1, R2)
        return None
    if channels == 3:
        inp_imgs = torch.concat([
            inp_imgs,
            interpolate(interpolate(inp_imgs, scale_factor=upscale_factor, mode='bilinear'), scale_factor=1/upscale_factor, mode='bicubic'),
            interpolate(interpolate(inp_imgs, scale_factor=upscale_factor, mode='nearest'), scale_factor=1/upscale_factor, mode='bicubic'),
        ], dim=1)
    return inp_imgs, tgt_imgs, bic_imgs