import numpy as np

def read_spectrum(fn, transpose_y=True):
    mat = np.loadtxt(fn).T
    x = mat[0]
    ys = mat[1:]
    r = np.sqrt(ys.shape[0]).astype(int)
    ys = ys.reshape(r,r,-1)
    if transpose_y:
        # channel at first dim
        ys = ys.transpose(2, 0, 1)
    return x, ys

def save_spectrum(x, ys, fn, fmt='%.2f'):
    x = x.reshape(-1, 1)
    n = x.shape[0]
    i = ys.shape.index(n)
    if i == 0:
        ys = ys.reshape(n, -1)
    else:
        ys = ys.reshape(-1, n).T
    mat = np.hstack([x, ys])
    np.savetxt(fn, mat, fmt=fmt)

def norm_spectrum(ys: np.ndarray, 
                  ref: np.ndarray = None, 
                  vmin:float = None, 
                  vmax:float = None, 
                  n: int = 2000, 
                  min_count: int = 3, 
                  mode:str = 'histogram',
                  num_bit = 256,
                  extend_bit = True,
                  ):
    if ref is None:
        ref = ys.copy()
    if mode == 'histogram':
        flat = ref.reshape(-1)
        count, bins = np.histogram(flat, bins=n)
        bins = (bins[1:] + bins[:-1]) * 0.5
        vmin = bins[count > min_count][0]
        vmax = bins[count > min_count][-1]
    elif mode == 'minmax':
        vmin = vmin if vmin is not None else np.min(ref)
        vmax = vmax if vmax is not None else np.max(ref)
    else:
        raise ValueError('Not supported normalization', mode)

    if extend_bit and num_bit != 0:
        vmax = np.max([vmax, vmin + num_bit - 1])
    y_norm = np.clip((ys - vmin) / (vmax - vmin), 0, 1)
    if num_bit != 0:
        y_norm = (y_norm * (num_bit - 1)).astype(int).astype(float) / 255.0
    return vmin, vmax, y_norm

def denorm_spectrum(ys, vmin, vmax):
    return ys.copy() * (vmax - vmin) + vmin

def get_mask(x1, x2):
    x1 = x1.reshape(-1)
    x2 = x2.reshape(-1)
    if x1.shape[0] > x2.shape[0]:
        v1 = x1.reshape(-1, 1)
        v2 = x2.reshape(1, -1)
    else:
        v1 = x2.reshape(-1, 1)
        v2 = x1.reshape(1, -1)
    mask = np.zeros(v1.shape[0]).astype(bool)
    mask[np.argmin(np.abs(v1 - v2), axis=0)] = True
    return mask

def transpose_spectrum(ys, axis=0):
    ys = ys.squeeze()
    shape = ys.shape
    if len(shape) == 2:
        if axis != 0:
            ys = ys.T
            shape = ys.shape
        r = np.sqrt(shape[1]).astype(int)
        ys = ys.reshape(-1, r, r)
    elif len(shape) == 3:
        if axis == 2 or shape[0] == shape[1]:
            ys.transpose(2,0,1)
    elif len(shape) == 4:
        i = np.argmax(shape)
        c = np.argmin(shape)
        mask = np.ones(4, dtype=bool)
        mask[i] = False
        mask[c] = False
        r = np.where(mask)[0].tolist()
        ys = ys.transpose(i, *r, c)
    else:
        raise ValueError('Invalid data shape', ys.shape)
    return ys