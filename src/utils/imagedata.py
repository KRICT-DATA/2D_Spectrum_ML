from torch.utils.data import Dataset
from .image import augment_image, convert_to_image
from .spectrum import get_mask, transpose_spectrum, norm_spectrum
import torch
import numpy as np

class SpectrumImageDataset(Dataset):
    def __init__(self, upscale_factor, data_augmentation=False, 
                 channels=3, window_size=0, mode='histogram'):
        
        super(SpectrumImageDataset, self).__init__()
        self.augmentation = data_augmentation
        self.upscale_factor = upscale_factor
        self.channels = channels
        self.mode = mode
        self.window_size = window_size

        self.xs = {}
        self.infos = []
        self.inputs = []
        self.targets = []
        self.bicubics = []
        self.subset_index = []

    def from_data(self, x, ys_inp, xs_ref=None, variance_masking=False, is_target=False):
        self.xs = {}
        self.infos = []
        self.inputs = []
        self.targets = []
        self.bicubics = []

        inp = ys_inp.squeeze()
        n = x.shape[0]
        shape = inp.shape
        inp = transpose_spectrum(inp, shape.index(n))
        if len(shape) == 4:
            inp = inp[..., 0]

        vmin, vmax, inp = norm_spectrum(inp, ref=inp, mode=self.mode)

        xs = x.copy()
        mask = np.ones(n, dtype=bool)
        if xs_ref is not None:
            mask = get_mask(x, xs_ref)
            n = mask.shape[0]
            xs = xs_ref.copy()
            inp_ = np.zeros((n, *shape[1:]), dtype=np.float32)
            inp_[mask] = inp
            inp = inp_.copy()

        if variance_masking:
            vars = np.var(inp.reshape(n, -1), axis=1)
            hist, bins = np.histogram(vars, bins=2000)
            bins = (bins[1:] + bins[:-1]) * 0.5
            mask = vars > bins[np.argmax(hist)]
            n = np.sum(mask)
            mask[np.argsort(vars)[-n-128+n%128:-n]] = True

        n, h, w = inp.shape
        if self.window_size != 0:
            h_pad = (h // self.window_size + 1) * self.window_size - h
            w_pad = (w // self.window_size + 1) * self.window_size - w
            inp = np.concatenate([inp, np.flip(inp, 1)], 1)[:, :h+h_pad, :]
            inp = np.concatenate([inp, np.flip(inp, 2)], 2)[:, :, :w+w_pad]
            
        _x = convert_to_image(inp, None, upscale_factor=self.upscale_factor, channels=self.channels)

        imgs_inp, imgs_tgt, imgs_bic = _x
        for i in np.where(mask)[0]:
            self.infos.append([None, None, vmin, vmax, xs[i], h*self.upscale_factor, w*self.upscale_factor])
            self.inputs.append(imgs_inp[i])
            self.targets.append(imgs_tgt[i])
            self.bicubics.append(imgs_bic[i])

    def __len__(self):
        return len(self.infos)
    
    def __getitem__(self, index):
        input = self.inputs[index]
        target = self.targets[index]
        bicubic = self.bicubics[index]
        info = self.infos[index]
        
        if self.augmentation:
            input, target, bicubic = augment_image(input, target, bicubic)[:3]
        return input, target, bicubic, info

def collate_fn(data):
    inps = []
    tgts = []
    bics = []
    infos = []
    for (inp, tgt, bic, info) in data:
        inps.append(inp)
        tgts.append(tgt)
        bics.append(bic)
        infos.append(info)
    return torch.stack(inps, dim=0), torch.stack(tgts, dim=0), torch.stack(bics, dim=0), infos
