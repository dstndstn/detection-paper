from __future__ import print_function
import os
import numpy as np

from legacypipe.survey import *
from legacypipe.runbrick import *
from astrometry.util.stages import CallGlobalTime, runstage

def stage_detect(targetrd=None, pixscale=None, targetwcs=None,
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
    gal_fwhms = np.array([1., 2., 4.])
    gal_sigmas = gal_fwhms/2.35

    detmaps,detivs = galaxy_detection_maps(tims, gal_sigmas, targetwcs, bands, mp)

    i = 0
    for fwhm in gal_fwhms:
        for ib,band in enumerate(bands):
            fn = os.path.join(survey.output_dir, 'galdetmap-%.1f-%s.fits' % (fwhm,band))
            fitsio.write(fn, detmaps[i].astype(np.float32), clobber=True)
            print('Wrote', fn)
            fn = os.path.join(survey.output_dir, 'galdetiv-%.1f-%s.fits' % (fwhm, band))
            fitsio.write(fn, detivs[i].astype(np.float32), clobber=True)
            print('Wrote', fn)
            i += 1

def galaxy_detection_maps(tims, galsigmas, targetwcs, bands, mp):
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
            args.append((tim, s, targetwcs, H, W))
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
    (tim, galsigma, targetwcs, H, W) = X
    R = tim_get_resamp(tim, targetwcs)
    if R is None:
        return None,None,None,None
    ie = tim.getInvvar()
    assert(tim.psf_sigma > 0)
    pixscale = tim.subwcs.pixel_scale()
    # convert galsigma to pixels
    galsigma /= pixscale
    print('Galaxy sigma: %.2f, PSF sigma: %.2f' % (galsigma, tim.psf_sigma))
    sigma = np.hypot(tim.psf_sigma, galsigma)
    gnorm = 1./(2. * np.sqrt(np.pi) * sigma)
    detim = tim.getImage().copy()
    detim[ie == 0] = 0.
    detim = gaussian_filter(detim, sigma) / gnorm**2
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

    survey, kwargs = get_runbrick_kwargs(**optdict)
    if kwargs in [-1,0]:
        return kwargs

    kwargs.update(prereqs_update={'detect': 'mask_junk',
                                  'galdetect': 'mask_junk'})

    stagefunc = CallGlobalTime('stage_%s', globals())
    kwargs.update(stagefunc=stagefunc)

    #kwargs.update(stages=['image_coadds', 'detect'])
    kwargs.update(stages=['galdetect'])
    #kwargs.update(stages=['detect',]) # 'srcs']) # with early_coadds, srcs:image_coadds
    #kwargs.update(stages=['srcs'])

    run_brick(opt.brick, survey, **kwargs)
    return 0
    
if __name__ == '__main__':
    import sys
    sys.exit(main())
