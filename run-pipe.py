
from __future__ import print_function
import os
import numpy as np

from legacypipe.survey import *
from legacypipe.runbrick import *
from astrometry.util.stages import CallGlobalTime, runstage

from legacypipe.detection import _detmap

def detection_maps(tims, targetwcs, bands, mp, minmax=True):
    '''
    minmax: remove the min and max images in the coadd?
    '''

    # Render the per-band detection maps
    H,W = targetwcs.shape
    ibands = dict([(b,i) for i,b in enumerate(bands)])

    detmaps = [np.zeros((H,W), np.float32) for b in bands]
    detivs  = [np.zeros((H,W), np.float32) for b in bands]
    satmaps = [np.zeros((H,W), bool)       for b in bands]

    if minmax:
        maxdetmaps = [np.empty((H,W), np.float32) for b in bands]
        maxdetivs  = [np.empty((H,W), np.float32) for b in bands]
        mindetmaps = [np.empty((H,W), np.float32) for b in bands]
        mindetivs  = [np.empty((H,W), np.float32) for b in bands]
        default_max = -1e12
        default_min = +1e12
        for d in maxdetmaps:
            d[:,:] = default_max
        for d in mindetmaps:
            d[:,:] = default_min

    apodize = 10
    for tim, (band, Yo, Xo, incmap, inciv, sat) in zip(
        tims, mp.map(_detmap, [(tim, targetwcs, apodize) for tim in tims])):
        if Yo is None:
            continue
        ib = ibands[tim.band]

        if minmax:
            # Keep track of the min & max values going into the coadd
            Kmax = np.flatnonzero(incmap > maxdetmaps[ib][Yo,Xo])
            if len(Kmax):
                maxdetmaps[ib][Yo[Kmax],Xo[Kmax]] = incmap[Kmax]
                maxdetivs [ib][Yo[Kmax],Xo[Kmax]] = inciv [Kmax]
            Kmin = np.flatnonzero(incmap < mindetmaps[ib][Yo,Xo])
            if len(Kmin):
                mindetmaps[ib][Yo[Kmin],Xo[Kmin]] = incmap[Kmin]
                mindetivs [ib][Yo[Kmin],Xo[Kmin]] = inciv [Kmin]

        detmaps[ib][Yo,Xo] += incmap * inciv
        detivs [ib][Yo,Xo] += inciv
        if sat is not None:
            satmaps[ib][Yo,Xo] |= sat

    if minmax:
        # Subtract off the min & max.
        for detmap,detiv,minmap,miniv,maxmap,maxiv in zip(detmaps, detivs, mindetmaps, mindetivs, maxdetmaps, maxdetivs):
            detmap -= (minmap * miniv + maxmap * maxiv)
            detiv  -= (miniv + maxiv)

    for detmap,detiv in zip(detmaps, detivs):
        detmap /= np.maximum(1e-16, detiv)
    return detmaps, detivs, satmaps

def stage_detect(targetrd=None, pixscale=None, targetwcs=None,
                 W=None,H=None,
                 bands=None, ps=None, tims=None,
                 plots=False, plots2=False,
                 brickname=None,
                 mp=None, nsigma=None,
                 survey=None, brick=None,
                 **kwargs):
    #from legacypipe.detection import (detection_maps, sed_matched_filters,
    #                    run_sed_matched_filters, segment_and_group_sources)
    from scipy.ndimage import binary_dilation
    from scipy.ndimage import label, find_objects, center_of_mass

    # Also compute a single exposure per band
    print('Computing single-exposure detection maps...')
    # One each in g,r,i with good seeing
    good_expnums = [563982, 569613, 567442]
    onetims = [tim for tim in tims if tim.imobj.expnum in good_expnums]

    # Single-image "coadd" image
    C = make_coadds(onetims, bands, targetwcs)
    for im,band in zip(C.coimgs, bands):
        fn = os.path.join(survey.output_dir, 'image-one-%s.fits' % band)
        fitsio.write(fn, im.astype(np.float32), clobber=True)
        print('Wrote', fn)
    #rgb = survey.get_rgb(C.coimgs, bands)
    #imsave_jpeg(out.fn, rgb, origin='lower', **kwa)

    detmaps, detivs, satmap = detection_maps(onetims, targetwcs, bands, mp,
                                             minmax=False)
    for i,band in enumerate(bands):
        fn = os.path.join(survey.output_dir, 'detmap-one-%s.fits' % band)
        fitsio.write(fn, detmaps[i].astype(np.float32), clobber=True)
        print('Wrote', fn)
        fn = os.path.join(survey.output_dir, 'detiv-one-%s.fits' % band)
        fitsio.write(fn, detivs[i].astype(np.float32), clobber=True)
        print('Wrote', fn)

    # Compute the coadded detection maps
    print('Computing detection maps...')
    detmaps, detivs, satmap = detection_maps(tims, targetwcs, bands, mp)
    for i,band in enumerate(bands):
        fn = os.path.join(survey.output_dir, 'detmap-%s.fits' % band)
        fitsio.write(fn, detmaps[i].astype(np.float32), clobber=True)
        print('Wrote', fn)
        fn = os.path.join(survey.output_dir, 'detiv-%s.fits' % band)
        fitsio.write(fn, detivs[i].astype(np.float32), clobber=True)
        print('Wrote', fn)

def stage_galdetect(targetrd=None, pixscale=None, targetwcs=None,
                    W=None,H=None,
                    bands=None, ps=None, tims=None,
                    plots=False, plots2=False,
                    brickname=None,
                    mp=None, nsigma=None,
                    survey=None, brick=None,
                    **kwargs):
    from legacypipe.detection import (detection_maps, sed_matched_filters,
                        run_sed_matched_filters, segment_and_group_sources)
    from scipy.ndimage.morphology import binary_dilation
    from scipy.ndimage.measurements import label, find_objects, center_of_mass

    print('Computing detection maps...')
    # 1", 2", 4" FWHM
    gal_fwhms = np.array([1.]) #, 2., 4.])
    gal_sigmas = gal_fwhms/2.35

    # 0.7" r_e EXP
    #gal_re = np.array([0.7])

    detmaps,detivs = galaxy_detection_maps(tims, gal_sigmas, True,
                                           targetwcs, bands, mp)

    i = 0
    for fwhm in gal_fwhms:
        #for re in gal_re:
        for ib,band in enumerate(bands):
            fn = os.path.join(survey.output_dir, 'galdetmap-%.1f-%s.fits' % (fwhm,band))
            #fn = os.path.join(survey.output_dir, 'galdetmap-re%.1f-%s.fits' % (re, band))
            fitsio.write(fn, detmaps[i].astype(np.float32), clobber=True)
            print('Wrote', fn)
            fn = os.path.join(survey.output_dir, 'galdetiv-%.1f-%s.fits' % (fwhm, band))
            #fn = os.path.join(survey.output_dir, 'galdetiv-re%.1f-%s.fits' % (re, band))
            fitsio.write(fn, detivs[i].astype(np.float32), clobber=True)
            print('Wrote', fn)
            i += 1

def galaxy_detection_maps(tims, galsigmas, gaussian, targetwcs, bands, mp):
    # Render the detection maps
    # Returns in order of galsigmas then bands
    # eg s1b1, s1b2, s1b3, s2b1, s2b2, ...
    H,W = targetwcs.shape
    ibands = dict([(b,i) for i,b in enumerate(bands)])

    N = len(bands) * len(galsigmas)
    detmaps = [np.zeros((H,W), np.float32) for i in range(N)]
    detivs  = [np.zeros((H,W), np.float32) for i in range(N)]

    args = []
    imaps = []
    for isig,s in enumerate(galsigmas):
        for tim in tims:
            args.append((tim, s, gaussian, targetwcs, H, W))
            imaps.append(isig * len(bands) + ibands[tim.band])
    R = mp.map(_galdetmap, args)

    for imap,res in zip(imaps, R):
        Yo,Xo,incmap,inciv = res
        if Yo is None:
            continue
        detmaps[imap][Yo,Xo] += incmap * inciv
        detivs [imap][Yo,Xo] += inciv
    for detmap,detiv in zip(detmaps, detivs):
        detmap /= np.maximum(1e-16, detiv)
    return detmaps, detivs

def _galdetmap(X):
    from scipy.ndimage.filters import gaussian_filter
    from legacypipe.survey import tim_get_resamp
    (tim, galsize, gaussian, targetwcs, H, W) = X
    R = tim_get_resamp(tim, targetwcs)
    if R is None:
        return None,None,None,None
    ie = tim.getInvvar()
    assert(tim.psf_sigma > 0)
    pixscale = tim.subwcs.pixel_scale()

    if gaussian:
        # convert galsize (in sigmas in arcsec) to pixels
        galsigma = galsize / pixscale
        sigma = np.hypot(tim.psf_sigma, galsigma)
        gnorm = 1./(2. * np.sqrt(np.pi) * sigma)
        detim = tim.getImage().copy()
        detim[ie == 0] = 0.
        detim = gaussian_filter(detim, sigma) / gnorm**2

    else:
        galsigs = np.sqrt(ExpGalaxy.profile.var[:,0,0]) * galsize / pixscale
        galamps = ExpGalaxy.profile.amp
        #print('Galaxy sigma: %.2f, PSF sigma: %.2f' % (galsigma, tim.psf_sigma))
        print('Galaxy amps', galamps, 'sigmas', galsigs)

        sz = 20
        xx,yy = np.meshgrid(np.arange(-sz, sz+1), np.arange(-sz, sz+1))
        rr = xx**2 + yy**2
        normim = 0
        detim = 0
        img = tim.getImage().copy()
        img[ie ==  0] = 0.
        for amp,sig in zip(galamps, galsigs):
            sig = np.hypot(tim.psf_sigma, sig)
            detim  += amp * gaussian_filter(img, sig)
            normim += amp * 1./(2.*np.pi*sig**2) * np.exp(-0.5 * rr / sig**2)
        #print('Normimg:', normim.sum())
        gnorm = np.sqrt(np.sum(normim**2))
        print('Galnorm', gnorm, 'vs psfnorm', 1./(2. * np.sqrt(np.pi) * tim.psf_sigma), 'seeing', tim.psf_fwhm/pixscale)
        detim /= gnorm**2

    detsig1 = tim.sig1 / gnorm
    subh,subw = tim.shape
    detiv = np.zeros((subh,subw), np.float32) + (1. / detsig1**2)
    detiv[ie == 0] = 0.
    (Yo,Xo,Yi,Xi) = R
    return Yo, Xo, detim[Yi,Xi], detiv[Yi,Xi]

def main():
    from legacypipe.runbrick import run_brick, get_parser, get_runbrick_kwargs
    parser = get_parser()
    #parser.add_argument('--subset', type=int, help='COSMOS subset number [0 to 4, 10 to 12]', default=0)
    opt = parser.parse_args()
    if opt.brick is None and opt.radec is None:
        parser.print_help()
        return -1

    optdict = vars(opt)
    verbose = optdict.pop('verbose')

    if verbose == 0:
        lvl = logging.INFO
    else:
        lvl = logging.DEBUG
    logging.basicConfig(level=lvl, format='%(message)s', stream=sys.stdout)
    # tractor logging is *soooo* chatty
    logging.getLogger('tractor.engine').setLevel(lvl + 10)
    # silence "findfont: score(<Font 'DejaVu Sans Mono' ...)" messages
    logging.getLogger('matplotlib.font_manager').disabled = True
    # route warnings through the logging system
    logging.captureWarnings(True)

    survey, kwargs = get_runbrick_kwargs(**optdict)
    if kwargs in [-1,0]:
        return kwargs

    #kwargs.update(prereqs_update={'detect': 'mask_junk',
    #                              'galdetect': 'mask_junk'})

    #kwargs.update(prereqs_update={'detect': 'tims'})
    kwargs.update(prereqs_update={'detect': 'outliers'})
    
    stagefunc = CallGlobalTime('stage_%s', globals())
    kwargs.update(stagefunc=stagefunc)

    #kwargs.update(stages=['image_coadds', 'detect'])
    #kwargs.update(stages=['galdetect'])
    #kwargs.update(stages=['detect',]) # 'srcs']) # with early_coadds, srcs:image_coadds
    #kwargs.update(stages=['srcs'])

    run_brick(opt.brick, survey, **kwargs)
    return 0
    
if __name__ == '__main__':
    import sys
    sys.exit(main())
