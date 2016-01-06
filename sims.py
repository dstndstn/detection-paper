import matplotlib
matplotlib.use('Agg')
import pylab as plt

from astrometry.util.plotutils import *
from scipy.ndimage.filters import *

def sim_multi_epoch():
    ps = PlotSequence('sim1')

    W,H = 15,15
    sig1s = [1.]*4
    psfsigmas = [1.5, 2.0, 2.5, 3.0]

    flux = 10.
    cx,cy = W/2, H/2
    
    N = 100000

    fdet = []
    fco = []
    
    for i in range(N):
        if i % 1000 == 0:
            print '.', 
        imgs = []
        r2 = (((np.arange(W)-cx)**2)[np.newaxis,:] +
              ((np.arange(H)-cy)**2)[:,np.newaxis])
        psfimgs = []
        for sig1,psfsig in zip(sig1s, psfsigmas):
            img = np.random.normal(size=(H,W)) * sig1
            psfimg = 1./(2.*np.pi * psfsig**2) * np.exp(-0.5 * r2/psfsig**2)
            img += flux * psfimg
            psfimgs.append(psfimg)
            imgs.append(img)
            
            
        det = np.zeros_like(imgs[0])
        detiv = 0.
        for sig1,psfsig,img in zip(sig1s, psfsigmas, imgs):
            d = gaussian_filter(img, psfsig)
            psfnorm = 1./(2.*np.sqrt(np.pi)*psfsig)
            iv = psfnorm**2 / sig1**2
            det += d / psfnorm**2 * iv
            detiv += iv
            #print 'PSF norm for psfsigma', psfsig, '=', psfnorm
        det /= detiv

        fdet.append(det[cy,cx])

        # Coadd
        
        coadd = np.zeros_like(imgs[0])
        coiv = 0.
        coivs = []
        for sig1,psfsig,img in zip(sig1s, psfsigmas, imgs):
            iv = 1.
            coivs.append(iv)
            coadd += iv * img
            coiv += iv
        coadd /= coiv
        # Convolve by coadd's PSF
        conv = np.zeros_like(coadd)
        ivsum = 0
        for iv,psfsig in zip(coivs, psfsigmas):
            conv += iv * gaussian_filter(coadd, psfsig)
            ivsum += iv
        conv /= ivsum
        psfsum = np.zeros_like(coadd)
        for iv,psfimg in zip(coivs, psfimgs):
            psfsum += iv * psfimg
        psfsum /= ivsum
        psfnorm = np.sqrt(np.sum(psfsum ** 2))
        #print 'Coadd PSF norm', psfnorm
        conv /= psfnorm**2

        fco.append(conv[cy,cx])

    plt.clf()
    plt.hist(fdet, 50, range=(0,20), histtype='step', color='b')
    plt.hist(fco, 50, range=(0,20), histtype='step', color='r')
    ps.savefig()


if __name__ == '__main__':
    sim_multi_epoch()
        
        
