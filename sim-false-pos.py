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

def hot_pixels(img, thresh, peaks):
    from scipy.ndimage.measurements import label, find_objects
    if not peaks:
        return img > thresh

    # Return only the maximum pixel in each blob above threshold.
    peaks = np.zeros(img.shape, bool)
    labels,nlabels = label(img > thresh)
    # max label == n labels; 0 means not an object
    #print('Max label:', labels.max(), 'nlabels', nlabels)
    for i,slices in enumerate(find_objects(labels, max_label=nlabels)):
        if slices is None:
            continue
        slicey,slicex = slices
        subimg = img[slicey,slicex]
        imax = np.argmax(subimg * (labels[slicey,slicex] == (i+1)))
        iy,ix = np.unravel_index(imax, subimg.shape)
        x0,y0 = slicex.start, slicey.start
        peaks[y0+iy, x0+ix] = True
    return peaks

def falsepos_rate(psf_sigmas, sig1s, thresh, seds, ps,
                  fluxgrid=None, peaks=False):
    '''
    fluxgrid: [ (grid for band 1), (grid for band 2), ... ]
          grid for band 1 = NY x NX flux values
    '''
    H,W = 1000,1000
    psf_norms = 1. / (2. * np.sqrt(np.pi) * psf_sigmas)

    imgs = []
    for sig1 in zip(sig1s):
        img = np.random.normal(scale=sig1, size=(H,W)).astype(np.float32)
        imgs.append(img)

    if fluxgrid is not None:
        NY,NX = None,None
        xx,yy = None,None
        for iband,(img, fluxes, psf_sigma) in enumerate(
                zip(imgs, fluxgrid, psf_sigmas)):
            if fluxes is None:
                continue
            # grids must be the same shape in each band
            ny,nx = fluxes.shape
            if NY is None:
                NY = ny
                NX = nx
                spacex = int(W / float(NX))
                spacey = int(H / float(NY))
                xx,yy = np.meshgrid(np.arange(NX) * spacex,
                                    np.arange(NY) * spacey)
            else:
                assert(NY == ny)
                assert(NX == nx)

            # Create PSF postage stamp of fixed size...
            P = 12
            px,py = np.meshgrid(np.arange(-P,P+1), np.arange(-P,P+1))
            pp = np.exp(-0.5 * (px**2 + py**2) / psf_sigma**2)
            pp /= np.sum(pp)
            ph,pw = pp.shape
            assert(xx.shape == yy.shape)
            assert(xx.shape == fluxes.shape)
            for x,y,flux in zip(xx.ravel(), yy.ravel(), fluxes.ravel()):
                print('Adding flux', flux, 'at (%i, %i) in band' % (x,y), iband)
                img[y:y+ph, x:x+pw] += flux * pp

    detmaps = []
    for img,psf_sigma,psf_norm in zip(imgs, psf_sigmas, psf_norms):
        detim = gaussian_filter(img, psf_sigma) / psf_norm**2
        detmaps.append(detim)
    detsigs = sig1s / psf_norms

    if ps is not None:
        for detmap,detsig in zip(detmaps,detsigs):
            plt.clf()
            plt.imshow(detmap, interpolation='nearest', origin='lower',
                       vmin=-3*detsig, vmax=10*detsig)
            ps.savefig()
    
    print('1 / detsigs =', 1./detsigs)
    
    nhots = []
    allhots = []

    if ps is not None:
        plt.clf()
        
    chisq = 0.
    for detmap,detsig1 in zip(detmaps,detsigs):
        chisq = chisq + (detmap / detsig1)**2
    hot = hot_pixels(chisq, thresh**2, peaks)
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
        hot = hot_pixels(sedsn, thresh, peaks)
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
    if ps is not None:
        allhots.append(('Any SED', anyhot))

    return dict(rates=np.array(nhots) / float(H*W),
                allhots=allhots, detmaps=detmaps, detsigs=detsigs)

psf_sigmas = np.array([1.5, 1.5])
sig1s = np.array([1.0, 1.0])
thresh = 3.
seds = [('g', [1., 0.]),
        ('r', [0., 1.]),
        ('flat', [1., 1.]),
        ('red',  [1., 2.5]),
        ]

ps = PlotSequence('fp')

#R = falsepos_rate(psf_sigmas, sig1s, thresh, seds, ps,
#                  [])

flux_g,flux_r = np.meshgrid(np.arange(1, 21), np.arange(1,21))
flux_g *= 3.
flux_r *= 3.

R = falsepos_rate(psf_sigmas, sig1s, thresh, seds, ps,
                  fluxgrid=(flux_g, flux_r), peaks=True)

plt.legend(loc='upper right')

import scipy.stats
# xl,xh = plt.xlim()
# xx = np.linspace(xl, xh, 300)
df = len(psf_sigmas)
# yy = scipy.stats.chi2.pdf(xx, df)
# yl,yh = plt.ylim()
# plt.plot(xx, yh * yy / yy.max(), 'k--')
print('Expected number of chisq: %.1f' %
      ((1 - scipy.stats.chi2.cdf(thresh**2, df))*1000000))
print('Expected number of SED: %.1f' %
      ((1 - scipy.stats.norm.cdf(thresh))*1000000))

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
    #plt.axis([-5,5,-5,5])
    plt.axis([-5,20,-5,20])
    ps.savefig()
