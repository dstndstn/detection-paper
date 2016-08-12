from __future__ import print_function
#import matplotlib
#matplotlib.use('Agg')
import pylab as plt
import numpy as np

from astrometry.util.plotutils import *
from scipy.ndimage.filters import *

# def detection_rate(N, bands, true_sed, psf_sigmas, nsigma, seds):
#     H,W = 1000,1000
#     sig1 = 1.
#     psfnorms = 1. / (2. * np.sqrt(pi) * psf_sigmas)
#     true_sed /= np.sum(true_sed)
#     # compute S/N of unit flux, scale sig1s to achive target S/N.
#     fluxes = true_sed
#     nsigmas = fluxes * psfnorms / sig1
#     nstot = np.sqrt(np.sum(nsigmas**2))
#     scale = nsigma / nstot
#     sig1s = np.ones(len(bands)) / scale
# 
#     imgs = []
#     for flux,psf_sigma,sig1 in zip(fluxes, psf_sigmas,sig1s):
#         imgs.ap


def falsepos_rate(psf_sigmas, sig1s, thresh, seds, ps):
    H,W = 1000,1000
    psf_norms = 1. / (2. * np.sqrt(np.pi) * psf_sigmas)

    imgs = []
    for psf_sigma,sig1 in zip(psf_sigmas,sig1s):
        img = np.random.normal(scale=sig1, size=(H,W)).astype(np.float32)
        imgs.append(img)

    detmaps = []
    for img,psf_sigma,psf_norm in zip(imgs, psf_sigmas, psf_norms):
        detim = gaussian_filter(img, psf_sigma) / psf_norm**2
        detmaps.append(detim)
    detsigs = sig1s / psf_norms

    print('1 / detsigs =', 1./detsigs)
    
    nhots = []
    allhots = []

    if ps is not None:
        plt.clf()
        
    chisq = 0.
    for detmap,detsig1 in zip(detmaps,detsigs):
        chisq = chisq + (detmap / detsig1)**2
    hot = (chisq > thresh**2)
    print('N hot pixels (chisq):', np.sum(hot))
    nhots.append(np.sum(hot))
    if ps is not None:
        plt.hist(chisq.ravel(), range=(-5, 15), bins=100,
                 histtype='step', label='Chisq')
        allhots.append(('Chisq', hot))

    anyhot = None
    for sedname,sed in seds:
        sedmap = 0.
        sediv  = 0.
        for iband in range(len(sed)):
            if sed[iband] == 0:
                continue
            detiv = 1./detsigs[iband]**2
            sedmap = sedmap + detmaps[iband] * detiv / sed[iband]
            sediv  = sediv  + detiv / sed[iband]**2
        sedmap /= sediv
        sedsn = sedmap * np.sqrt(sediv)
        hot = (sedsn > thresh)
        print('N hot pixels (%s):' % sedname, np.sum(hot))
        nhots.append(np.sum(hot))
        if anyhot is None:
            anyhot = hot
        else:
            anyhot = np.logical_or(anyhot, hot)
        if ps is not None:
            plt.hist(sedsn.ravel(), range=(-5, 15), bins=100,
                     histtype='step', label='SED '+sedname)
            allhots.append(('SED '+sedname, hot))
    nhot = np.sum(anyhot)
    print('N hot pixels (any SED):', nhot)
    nhots.append(nhot)
    
    return dict(rates=np.array(nhots) / float(H*W),
                allhots=allhots, detmaps=detmaps, detsigs=detsigs)

psf_sigmas = np.array([1.0, 2.0])
sig1s = np.array([1.0, 1.0])
thresh = 3.
seds = [('g', [0., 1.]),
        ('r', [1., 0.]),
        ('flat', [1., 1.]),
        ('red',  [1., 2.5]),
        ]

ps = PlotSequence('fp')

R = falsepos_rate(psf_sigmas, sig1s, thresh, seds, ps)
plt.legend(loc='upper right')
ps.savefig()

allhots = R['allhots']
detmaps = R['detmaps']
detsigs = R['detsigs']
sn_g = detmaps[0] / detsigs[0]
sn_r = detmaps[1] / detsigs[1]
for name,hot in allhots:
    plt.clf()
    plt.plot(sn_g[hot], sn_r[hot], 'k.')
    plt.xlabel('g band S/N')
    plt.ylabel('r band S/N')
    plt.title(name)
    plt.axis('scaled')
    plt.axis([-5,5,-5,5])
    ps.savefig()
