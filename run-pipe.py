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

    # for N in [1,5,25]:
    #     print('Rendering detection maps with', N, 'images per band...')
    #     count = Counter()
    #     keeptims = []
    #     for i,tim in enumerate(tims):
    #         key = (tim.ccdname, tim.filter)
    #         print('Tim', tim)
    #         print('Checking', k)
    #         print('Found', count[k], 'existing')
    #         if count[k] >= N:
    #             continue
    #         print('Keeping', tim)
    #         keeptims.append(tim)

    detmaps, detivs, satmap = detection_maps(tims, targetwcs, bands, mp)

    for i,band in enumerate(bands):
        fn = os.path.join(survey.survey_dir, 'detmap-%s.fits' % band)
        fitsio.write(fn, detmaps[i].astype(np.float32), clobber=True)
        fn = os.path.join(survey.survey_dir, 'detiv-%s.fits' % band)
        fitsio.write(fn, detivs[i].astype(np.float32), clobber=True)


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

    kwargs.update(prereqs_update={'detect': 'mask_junk'})

    stagefunc = CallGlobalTime('stage_%s', globals())
    kwargs.update(stagefunc=stagefunc)

    kwargs.update(stages=['image_coadds', 'detect'])
    #kwargs.update(stages=['detect',]) # 'srcs']) # with early_coadds, srcs:image_coadds
    #kwargs.update(stages=['srcs'])

    run_brick(opt.brick, survey, **kwargs)
    return 0
    
if __name__ == '__main__':
    import sys
    sys.exit(main())
