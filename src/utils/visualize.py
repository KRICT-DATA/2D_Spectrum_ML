import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .spectrum import norm_spectrum
 
def super_resolution_summary(x, y, x_inp=None, y_inp=None, 
                             locs=[624, 634, 644, 654], intensity_range=None, 
                             cmap=mpl.cm.inferno, figsize=(4,4), fontsize=14,
                             scalebar='horizontal'):
    xs = [x]
    ys = [y]
    vmin, vmax, _ = norm_spectrum(y, num_bit=0)
    if intensity_range is not None:
        vmin, vmax = intensity_range
    if vmax - vmin > 5000:
        vmin = int(vmin / 1000) * 1000
        vmax = int(vmax / 1000) * 1000 + (1000 if vmax % 1000 != 0 else 0)

    lbls = ['SR']
    if x_inp is not None and y_inp is not None:
        xs = [x_inp, x]
        ys = [y_inp, y]
        lbls = ['Orig.', 'SR']

    f, axs = plt.subplots(len(xs), len(locs), figsize=(figsize[0]*len(locs), figsize[1]*len(xs)))
    axs = axs.reshape(len(xs), len(locs))
    for i, (_x, _y, lbl) in enumerate(zip(xs, ys, lbls)):
        _, n1, n2 = _y.shape
        axs[i, 0].set_ylabel(f'{lbl}: {n1}x{n2}', fontsize=fontsize)

        for j, loc in enumerate(locs):
            k = np.argmin(np.abs(_x - loc))
            if i == 0:
                axs[0,j].set_title(f'{_x[k]:.2f}nm', fontsize=fontsize)
            im = axs[i,j].imshow(_y[k], vmin=vmin, vmax=vmax, cmap=cmap)
    for ax in axs.reshape(-1):
        ax.set_yticks([])
        ax.set_xticks([])

    f.subplots_adjust(wspace=0.05, hspace=0.02)
    
    if isinstance(scalebar, str) and scalebar.lower().startswith('v'):
        f1, ax1 = plt.subplots(1,1,figsize=(0.3,4))
        plt.colorbar(im, cax=ax1)
        ax1.set_xticks([])
        ax1.set_ylabel('Intensity', fontsize=fontsize)
    else:
        f1, ax1 = plt.subplots(1,1,figsize=(4,0.3))
        plt.colorbar(im, cax=ax1, orientation='horizontal')
        ax1.set_yticks([])
        ax1.set_xlabel('Intensity', fontsize=fontsize)
    return f, f1

def clustering_summary(tsne_vector, labels, 
                       figsize=(12,5.5), gridspec_kw={'width_ratios':[1,1,0.05]},
                       cmap=mpl.cm.viridis, fontsize=15):
    cmap.set_under([0.7, 0.7, 0.7])
    num_clusters = np.max(labels) + 1
    bounds = np.linspace(0,num_clusters,num_clusters+1)
    f, axs = plt.subplots(1,3,figsize=figsize, gridspec_kw=gridspec_kw)
    im = axs[0].scatter(*tsne_vector.T, c=labels.reshape(-1)[labels.reshape(-1) != -1], cmap=cmap)
    im = axs[1].imshow(labels, vmin=0, vmax=num_clusters-1, cmap=cmap)
    cb = plt.colorbar(im, cax=axs[2], ticks=bounds, boundaries=bounds-0.5, extend='min')
    cb.set_label(label='Cluster', size=fontsize-1.5)
    cb.ax.set_yticklabels((bounds+1).astype(int))
    axs[0].set_title('$t$-SNE', fontsize=fontsize)
    axs[1].set_title('Sample', fontsize=fontsize)
    f.subplots_adjust(wspace=0.05)
    for ax in axs[:2]: 
        ax.set_xticks([])
        ax.set_yticks([])
    return f

def clustering_details(x, ys, tsne_vector, labels, num_example,
                       figsize=(12,4), gridspec_kw={'width_ratios':[1.5,1,1]},
                       cmap=mpl.cm.viridis, fontsize=15, random_state=100):
    num_clusters = np.max(labels) + 1
    cmap.set_under([0.7, 0.7, 0.7])
    np.random.seed(random_state)
    n1, n2 = labels.shape
    l = labels.reshape(-1)
    ys_ = ys.reshape(x.shape[0], -1).T
    f, axs = plt.subplots(num_clusters, 3, figsize=(figsize[0], figsize[1]*num_clusters), gridspec_kw=gridspec_kw)
    l_img = np.ones((n1*n2, 3)) * 0.7
    l_img[l != -1] = [0.6, 0.6, 0.6]
    vmin, vmax, _ = norm_spectrum(ys, num_bit=0)
    for i, ax in enumerate(axs):
        ax[0].set_title(f'Cluster: {i+1} / Spectrum', fontsize=fontsize)
        ax[1].set_title(f'Cluster: {i+1} / $t$-SNE', fontsize=fontsize)
        ax[2].set_title(f'Cluster: {i+1} / Sample', fontsize=fontsize)

        lidxs = np.where(l == i)[0]
        img = l_img.copy()
        img[lidxs] = cmap(i/(num_clusters-1))[:3]
        
        idxs = np.arange(len(lidxs))
        np.random.shuffle(idxs)
        idxs = sorted(idxs[:num_example], key=lambda x: ys[:, lidxs[x]//n1, lidxs[x]%n1].std())
        ax[1].scatter(*tsne_vector[l[l != -1] != i].T, color=[0.6, 0.6, 0.6])
        ax[1].scatter(*tsne_vector[l[l != -1] == i].T, color=cmap(i/(num_clusters-1)))
        ax[2].imshow(img.reshape(n1,n2,-1))
        for ax_ in ax[1:]:
            ax_.set_xticks([])
            ax_.set_yticks([])
        for j, idx in enumerate(idxs):
            ax[0].plot(x, ys_[lidxs[idx]] + j * vmax * 0.3)
            ax[0].set_ylabel('Intensity (a.u.)', fontsize=fontsize)
            ax[0].set_xlabel('Wavelength (nm)', fontsize=fontsize)
            ax[1].scatter(*tsne_vector[l[l != -1] == i][idx], s=100, edgecolor=[0,0,0], marker='D')
            ax[2].scatter(lidxs[idx]%n1, lidxs[idx]//n1, color=mpl.cm.tab10(j), s=100, edgecolor=[0,0,0], marker='D')
        ax[0].set_ylim([0, (0.7 + 0.3*num_example) * vmax])
        for j in range(num_example+2):
            ax[0].axhline(vmin + j * vmax * 0.3, ls='--', color=[0,0,0], lw=0.5)
    f.subplots_adjust(wspace=0.03)
    f.subplots_adjust(hspace=0.4)
    return f 