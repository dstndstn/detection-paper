from __future__ import print_function
#import matplotlib
#matplotlib.use('Agg')
import sys
import pylab as plt
import numpy as np
import scipy
import scipy.optimize

from astrometry.util.plotutils import *
from astrometry.util.fits import *
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
        
    return dict(rates=np.array(nhots) / float(H*W),
                allhots=allhots, detmaps=detmaps, detsigs=detsigs)


def data_likelihood(d, sig):
    # flux prior exp(-f alpha)
    alpha = 0.5

    w = np.array([
        0.00310626,  0.00122104,  0.00118075,  0.00121068,  0.00124752,
        0.00132523,  0.00146281,  0.00164702,  0.00198952,  0.00237692,
        0.00313561,  0.00414643,  0.00588025,  0.00869339,  0.01187148,
        0.01404854,  0.01641211,  0.02039321,  0.02837903,  0.04234629,
        0.05702619,  0.05451238,  0.04028896,  0.03298758,  0.03019344,
        0.02942554,  0.02917341,  0.02930638,  0.02934092,  0.02951591,
        0.02954584,  0.02948195,  0.02945547,  0.02962931,  0.02970875,
        0.02967133,  0.02946583,  0.02954815,  0.02982388,  0.02828405,
        0.02371752,  0.01993443,  0.01561946,  0.01050319,  0.00771653,
        0.0066706 ,  0.00580657,  0.00513826,  0.00453729,  0.00401634,
        0.00349424,  0.00312065,  0.00276778,  0.00249551,  0.00221805,
        0.00204363,  0.00187209,  0.00173797,  0.00156873,  0.00146684,
        0.00139546,  0.0013402 ,  0.00126767,  0.00122047,  0.00119917,
        0.00116521,  0.00114621,  0.00110419,  0.00110419,  0.00108692,
        0.00107311,  0.00106332,  0.00105756,  0.00105756,  0.00105123,
        0.00104835,  0.00104375,  0.0010426 ,  0.00104605,  0.00103799,
        0.00103051,  0.00103166,  0.00102648,  0.00102418,  0.00102763,
        0.00103108,  0.00102072,  0.00102821,  0.00101957,  0.00103166,
        0.0010259 ,  0.00102303,  0.00102936,  0.00103108,  0.00103224,
        0.00103742,  0.00103857,  0.0010426 ,  0.0010472 ,  0.00296465])    
    lin = np.linspace(0, 1, len(w))
    s = np.vstack((lin, 1-lin)).T
    ns,nb = s.shape

    # f = np.linspace(0, 20, 300)
    # pf = alpha * np.exp(-f * alpha)
    # ptot = 0.
    # for i in range(ns):
    #     pg = 1./(np.sqrt(2.*np.pi)*sig[0]) * np.exp(-0.5 * (d[0] - f*s[i,0])**2 / sig[0]**2)
    #     pr = 1./(np.sqrt(2.*np.pi)*sig[1]) * np.exp(-0.5 * (d[1] - f*s[i,1])**2 / sig[1]**2)
    #     df = f[1]-f[0]
    #     ptot = ptot + w[i] * pg*pr*pf
    # print('ptot: ', np.sum(ptot)*df)


    prefac = alpha * np.prod(1. / np.sqrt(2.*np.pi) * sig) * np.exp(-0.5 * np.sum(d**2 / sig**2))
    pdata = 0.
    for w_i, s_i in zip(w, s):
        a = alpha - np.sum(d * s_i / sig**2)
        b = 0.5 * np.sum(s_i**2 / sig**2)
        pds = np.sqrt(np.pi)/(2.*np.sqrt(b)) * np.exp(a**2 / (4.*b)) * (1. - scipy.special.erf(a / (2.*np.sqrt(b))))
        #print('pdata for this sed:', pds*prefac)
        pdata += w_i * pds
    pdata *= prefac
    #print('pdata:', pdata)

    return pdata

class find_thresh(object):
    def __init__(self, d, sig, logodds):
        self.d = d
        self.sig = sig
        self.logodds = logodds

    def __call__(self, f):
        pd = data_likelihood(self.d * f, self.sig)
        ph0 = 1./(2.*np.pi*np.prod(self.sig)) * np.exp(-0.5 * np.sum((self.d * f / self.sig)**2))
        return np.log10(pd) - (np.log10(ph0) + self.logodds)
        


def main():
    # Going Bayesian... did I do my math right?


    # For a range of "angles", find flux required to exceed threshold.
    angles = np.linspace(0, 90, 50)
    angles *= np.pi/180.
    g = np.cos(angles)
    r = np.sin(angles)
    sig = np.array([1.,1.])

    f = []
    for gi,ri in zip(g,r):
        print('g,r', gi,ri)
        func = find_thresh(np.array([gi,ri]), sig, 6.)
        f0 = scipy.optimize.newton(func, 5.)
        print('-> f0', f0)
        f.append(f0)
        
    f = np.array(f)

    plt.clf()
    plt.plot(g*f, r*f, 'b-')
    plt.plot(g*6., r*6., 'k--', alpha=0.5)
    plt.xlabel('g flux')
    plt.ylabel('r flux')
    plt.axis('scaled')
    plt.savefig('thresh.png')

    thresh_g = g*f
    thresh_r = r*f
    
    
    
    gmax, rmax = 10,10
    g,r = np.meshgrid(np.linspace(0, gmax, 21),
                      np.linspace(0, rmax, 21))
    sig = np.array([1.,1.])
    pd = np.array([data_likelihood(np.array([gi,ri]), sig)
                   for gi,ri in zip(g.ravel(), r.ravel())]).reshape(g.shape)
    plt.clf()
    ima = dict(interpolation='nearest', origin='lower', extent=(0,gmax,0,rmax))
    plt.imshow(pd, **ima)
    plt.colorbar()
    plt.xlabel('g flux')    
    plt.ylabel('r flux')    
    plt.title('p(data)')
    plt.savefig('pdata.png')

    pH0 = 1./(2.*np.pi*np.prod(sig)) * np.exp(-0.5 * (g**2/sig[0]**2 + r**2/sig[1]**2))
    
    plt.clf()
    plt.imshow(pH0, **ima)
    plt.colorbar()
    plt.xlabel('g flux')    
    plt.ylabel('r flux')    
    plt.title('p(H0)')
    plt.savefig('ph0.png')

    plt.clf()
    plt.imshow(pd / pH0, vmin=0, vmax=100., **ima)
    plt.colorbar()
    plt.xlabel('g flux')    
    plt.ylabel('r flux')    
    plt.title('p(data) / p(H0)')
    plt.savefig('pfact.png')

    plt.clf()
    plt.imshow(np.log10(pd / pH0), vmin=-6, vmax=12., **ima)
    plt.colorbar()
    plt.xlabel('g flux')    
    plt.ylabel('r flux')    
    plt.title('log(p(data) / p(H0))')

    for f in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]:
        plt.plot(f*thresh_g, f*thresh_r, 'k-', alpha=0.5)
    
    plt.savefig('pfact2.png')

    


    
    # g and r data
    d = np.array([2., 6.5,])
    sig = np.array([1., 1.])
    # # SED
    # s = np.array([[ 0.5, 0.5 ],])
    # # SED weights
    # w = np.array([1.])
    # SEDs
    s = np.array([[ 0.5, 0.5 ],
                  [ 0.2, 0.8 ],
                  [ 0.8, 0.2 ],
                  ])
    w = np.array([ 0.5, 0.25, 0.25 ])
    # flux prior exp(-f alpha)
    alpha = 0.5
    
    # p(data | source exists) = \sum w_i \int N(data | f * sed_i, sig**2) p(f) df

    prefac = alpha * np.prod(1. / np.sqrt(2.*np.pi) * sig) * np.exp(-0.5 * np.sum(d**2 / sig**2))
    
    pdata = 0.
    for w_i, s_i in zip(w, s):
        print('w_i', w_i, 's_i', s_i)
        a = alpha - np.sum(d * s_i / sig**2)
        b = 0.5 * np.sum(s_i**2 / sig**2)
        print('a', a, 'b', b)
        pds = np.sqrt(np.pi)/(2.*np.sqrt(b)) * np.exp(a**2 / (4.*b)) * (1. - scipy.special.erf(a / (2.*np.sqrt(b))))
        print('pdata for this sed:', pds*prefac)
        pdata += w_i * pds
        #print('pdata total', pdata)

    # prefactors
    pdata *= prefac

    print('pdata', pdata)

    f = np.linspace(0, 20, 300)
    
    plt.clf()

    # gflux = f * s[:,0]
    # rflux = f * s[:,1]
    # plt.subplot(2,1,1)
    # plt.plot(f, gflux, 'g-')
    # plt.axhline(d[0], color='g')
    # plt.plot(f, rflux, 'r-')
    # plt.axhline(d[1], color='r')
    # plt.ylabel('flux')

    #plt.subplot(2,1,2)

    ptot = 0.
    ns,nb = s.shape
    for i,sty in enumerate(['-', '--', ':']):
        pg = 1./(np.sqrt(2.*np.pi)*sig[0]) * np.exp(-0.5 * (d[0] - f*s[i,0])**2 / sig[0]**2)
        pr = 1./(np.sqrt(2.*np.pi)*sig[1]) * np.exp(-0.5 * (d[1] - f*s[i,1])**2 / sig[1]**2)
        pf = alpha * np.exp(-f * alpha)

        plt.plot(f, pg, 'g', linestyle=sty)
        plt.plot(f, pr, 'r', linestyle=sty)
        plt.plot(f, pf, 'k', linestyle=sty)
        plt.plot(f, pg*pr*pf, 'b', linestyle=sty)

        df = f[1]-f[0]
        p = np.sum(pg*pr*pf) * df
        print('    p:', p)
        ptot += w[i] * p

    print('ptot:', ptot)
        
    plt.yscale('log')
    plt.ylim(1e-10, 1.)
    
    pH0 = 1./(np.sqrt(2.*np.pi)*np.prod(sig)) * np.exp(-0.5 * np.sum(d**2 / sig**2))
    print('p(H0):', pH0)
    
    plt.ylabel('prob')
    plt.xlabel('total flux f')
    
    plt.savefig('prob.png')



def empirical_sed_priors():
    alpha = 0.5
    f = np.linspace(0, 20, 300)
    d = np.array([2., 6.5,])
    sig = np.array([1., 1.])

    T = fits_table('sweep-240p005-250p010-cut.fits')
    print(len(T), 'sources')
    T.cut((T.nobs_g > 0) * (T.nobs_r > 0))
    #    (T.decam_nobs[:,4] > 0))
    print(len(T), 'with all Nobs > 0')
    
    T.gflux = np.maximum(0, T.flux_g)
    T.rflux = np.maximum(0, T.flux_r)
    T.gsn = T.flux_g * np.sqrt(T.flux_ivar_g)
    T.rsn = T.flux_r * np.sqrt(T.flux_ivar_r)
    T.cut(np.hypot(T.gsn, T.rsn) >= 10.)
    print(len(T), 'SN>10')

    sed = np.clip(T.gflux / (T.gflux + T.rflux), 0., 1.)

    plt.clf()
    sed_n,b,p = plt.hist(sed, range=(0,1), bins=100)
    plt.savefig('sedprior.png')

    plt.clf()
    plt.hist(sed, range=(0,1), bins=100, log=True)
    plt.savefig('sedprior2.png')

    plt.clf()
    #plt.plot(T.gflux, T.rflux, 'k.')
    #plt.xscale('symlog')
    #plt.yscale('symlog')
    loghist(np.log10(T.gflux), np.log10(T.rflux), 200, range=((0,4),(0,4)))
    plt.xlabel('log g flux')
    plt.ylabel('log r flux')
    plt.savefig('fluxes.png')

    plt.clf()
    plt.plot(T.gflux, T.rflux, 'k.', alpha=0.01)
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.xlabel('g flux')
    plt.ylabel('r flux')
    plt.savefig('fluxes2.png')

    plt.clf()
    loghist(T.gflux, T.rflux, 200, range=((0,10),(0,10)))
    plt.xlabel('g flux')
    plt.ylabel('r flux')
    plt.savefig('fluxes3.png')


    #### SED prior computed from "sed_n" above -- histogram.
    w = 0.9 * sed_n / np.sum(sed_n) + 0.1 / len(sed_n)

    lin = np.linspace(0, 1, len(w))
    s = np.vstack((lin, 1-lin)).T

    print('s =', repr(s))
    print('w =', repr(w))
    
    plt.clf()
    pf = alpha * np.exp(-f * alpha)
    plt.plot(f, pf, 'k-')

    print('Weights range:', w.min(), w.max())

    ptot = 0.
    ns,nb = s.shape
    for i in range(ns):
        pg = 1./(np.sqrt(2.*np.pi)*sig[0]) * np.exp(-0.5 * (d[0] - f*s[i,0])**2 / sig[0]**2)
        pr = 1./(np.sqrt(2.*np.pi)*sig[1]) * np.exp(-0.5 * (d[1] - f*s[i,1])**2 / sig[1]**2)

        ww = w[i] * 5.
        
        plt.plot(f, pg, 'g-', alpha=ww)
        plt.plot(f, pr, 'r-', alpha=ww)
        plt.plot(f, pg*pr*pf, 'b-', alpha=ww)

        df = f[1]-f[0]
        #p = np.sum(pg*pr*pf) * df
        #print('    p:', p)
        ptot = ptot + w[i] * pg*pr*pf

        #print('w[i]', w[i], 'f', f.shape, 'pg', pg.shape, 'pr', pr.shape, 'pf', pf.shape)
        #print('gflux', (f*s[i,0]).shape)
        #print('rflux', (f*s[i,1]).shape)
        

    print('ptot:', np.sum(ptot)*df)

    plt.plot(f, ptot, 'm-')
    
    plt.yscale('log')
    plt.ylim(1e-10, 1.)
    
    plt.ylabel('prob')
    plt.xlabel('total flux f')
    
    plt.savefig('prob2.png')

    # probability in the g,r flux plane
    Ngr = 100
    pgr = 0
    #fmax = f.max()
    fmax = 10
    for i in range(ns):
        pg = 1./(np.sqrt(2.*np.pi)*sig[0]) * np.exp(-0.5 * (d[0] - f*s[i,0])**2 / sig[0]**2)
        pr = 1./(np.sqrt(2.*np.pi)*sig[1]) * np.exp(-0.5 * (d[1] - f*s[i,1])**2 / sig[1]**2)
        nh,xe,ye = np.histogram2d(f * s[i,0], f * s[i,1], bins=Ngr,
                                   range=((0,fmax),(0,fmax)),
                                   weights = w[i] * pg * pr * pf)
        pgr = pgr + nh.T

        plt.clf()
        plt.imshow(nh.T, extent=(0,fmax,0,fmax), interpolation='nearest',
                   origin='lower', cmap='hot', vmin=0, vmax=0.0001)
        plt.xlabel('g flux')
        plt.ylabel('r flux')
        plt.colorbar()
        plt.plot(d[0], d[1], 'go')
        plt.title('SED = [%.2f, %.2f], W=%.4f' % (s[i,0],s[i,1],w[i]))
        plt.savefig('pgr-%03i.png' % i)

    
    plt.clf()
    plt.imshow(pgr, extent=(0,fmax,0,fmax), interpolation='nearest', origin='lower',
               cmap='hot')
    plt.xlabel('g flux')
    plt.ylabel('r flux')
    plt.colorbar()
    plt.plot(d[0], d[1], 'go')
    plt.savefig('pgr.png')
    
    sys.exit(0)










    psf_sigmas = np.array([1.5, 1.5])
    sig1s = np.array([1.0, 1.0])
    thresh = 3.
    seds = [('g', [1., 0.]),
            ('r', [0., 1.]),
            ('flat', [1., 1.]),
            ('red',  [1., 2.5]),
            ]
    
    ps = PlotSequence('fp')
    
    T = fits_table('sweep-240p005-250p010-cut.fits')
    print(len(T), 'sources')
    T.cut((T.decam_nobs[:,1] > 0) * (T.decam_nobs[:,2] > 0) *
          (T.decam_nobs[:,4] > 0))
    print(len(T), 'with all Nobs > 0')
    
    T.gflux = np.maximum(0, T.decam_flux[:,1])
    T.rflux = np.maximum(0, T.decam_flux[:,2])
    
    T.gmag = -2.5 * (np.log10(T.gflux) - 9.)
    T.rmag = -2.5 * (np.log10(T.rflux) - 9.)
    
    plt.clf()
    plt.plot(T.gflux, T.rflux, 'k.')
    plt.xlabel('g flux')
    plt.ylabel('r flux')
    ps.savefig()
    
    T.gsn = T.decam_flux[:,1] * np.sqrt(T.decam_flux_ivar[:,1])
    T.rsn = T.decam_flux[:,2] * np.sqrt(T.decam_flux_ivar[:,2])
    T.zsn = T.decam_flux[:,4] * np.sqrt(T.decam_flux_ivar[:,4])
    
    # plt.clf()
    # plt.plot(T.gsn, T.rsn, 'k.')
    # plt.xlabel('g S/N')
    # plt.ylabel('r S/N')
    # ps.savefig()
    # 
    # plt.axis([-5,20,-5,20])
    # ps.savefig()
    
    I = np.flatnonzero((T.gsn < 20) * (T.rsn < 20))[:10000]
    
    plt.clf()
    plt.scatter(T.gsn[I], T.rsn[I], s=5, c=np.clip(T.zsn[I], 0, 20),
                edgecolors='none')
    plt.colorbar()
    plt.xlabel('g S/N')
    plt.ylabel('r S/N')
    plt.axis([-5,20,-5,20])
    ps.savefig()
    
    # plt.clf()
    # loghist(T.gsn, T.rsn, 200, range=((-5,20),(-5,20)))
    # plt.xlabel('g S/N')
    # plt.ylabel('r S/N')
    # ps.savefig()
    
    I = np.flatnonzero((np.hypot(T.gsn, T.rsn) > 6.))
    # plt.clf()
    # plt.hist(T.rsn[I] / T.gsn[I], bins=100, range=(-1, 5))
    # plt.xlabel('r S/N / g s/N')
    # ps.savefig()
    # 
    # plt.clf()
    # plt.hist(T.rflux[I] / T.gflux[I], bins=100, range=(-1, 5))
    # plt.xlabel('r flux / g flux')
    # ps.savefig()
    # 
    # plt.clf()
    # plt.hist(np.log10(T.rflux[I] / T.gflux[I]), bins=100, range=(-1, 1.5))
    # plt.xlim(-1,1.5)
    # plt.xlabel('log10(r flux / g flux)')
    # ps.savefig()
    
    plt.clf()
    plt.hist(T.gmag[I] - T.rmag[I], bins=100, range=(-1.5, 3))
    plt.xlim(-1.5, 3)
    plt.xlabel('g - r (mag)')
    ps.savefig()
    
    
    frac = T.gflux[I] / T.rflux[I]
    frac[T.rflux[I] == 0] = 1e3
    
    # number of "SEDs" (g/r fractions)
    NS = 4
    pcts = np.linspace(0, 100, NS*2+1)[1:-1:2]
    print('Percents', pcts)
    ptiles = np.percentile(frac, pcts)
    print('Percentiles', ptiles)
    seds = [('pct%i' % i, [p, 1.]) for i,p in enumerate(ptiles)]
    
    if True:
        R = falsepos_rate(psf_sigmas, sig1s, thresh, seds, ps, [])
    else:
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
    
    lines = []
    
    for (name,hot),sed in zip(allhots, seds+[None,None]):
        # HACK
        if 'pct' in name:
            # print('Not plotting name', name)
            # print('sed', sed)
            # print('detsigs', detsigs)
            sedname,sedvals = sed
            ff = 1./(detsigs * sedvals)
            ff /= np.sqrt(np.sum(ff**2))
            slope = -ff[0] / ff[1]
            b = thresh/ff[1]
            #lines.append(([0, thresh/ff[0]],[thresh/ff[1],0]))
            xx = np.array([-5,20])
            lines.append((xx, b+slope*xx))
            #continue
        plt.clf()
        plt.plot(sn_g[hot], sn_r[hot], 'k.')
        plt.xlabel('g band S/N')
        plt.ylabel('r band S/N')
        plt.title(name)
        plt.axis('scaled')
        #plt.axis([-5,5,-5,5])
        plt.axis([-5,20,-5,20])
        ps.savefig()


if __name__ == '__main__':
    # plt.clf()
    # f = np.logspace(0, 1, 50)
    # plt.semilogy(np.exp(f * -1), 'k-', label='exp(-f)')
    # plt.semilogy(np.exp(f * -2), 'b-', label='exp(-2f)')
    # plt.semilogy(f**-1, 'r-', label='1/f')
    # plt.semilogy(f**-2, 'g-', label='1/f^2')
    # plt.legend()
    # plt.savefig('pf.png')
    # sys.exit(0)

    empirical_sed_priors()

    #main()
