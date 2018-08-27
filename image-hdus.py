from astrometry.util.fits import fits_table
import numpy as np
import os

T = fits_table('survey-ccds-snx3-25.fits.gz')
print(len(T), 'CCDs')

print('Filenames:', T.image_filename)

indir = '/global/cscratch1/sd/dstn/detection/' #decam/cp'
#outdir = '/global/cscratch1/sd/dstn/detection/subimages/' #decam/cp-sub'

T.image_filename = np.array([fn.strip() for fn in T.image_filename])
print(len(set(T.image_filename)), 'unique exposures')
for fn in set(T.image_filename):
    infn = indir + fn
    outfn = infn.replace('decam/cp/', 'decam/cp-sub/')
    assert(infn != outfn)

    J, = np.nonzero(T.image_filename == fn)
    print(len(J), 'HDUs')
    assert(len(J) == 2)

    for tag in ['_ooi_', '_ood_', '_oow_']:
        cmd = ('fitsgetext -i %s -o %s -e 0 -e %i -e %i' %
               (infn.replace('_ooi_', tag), outfn.replace('_ooi_', tag),
                T.image_hdu[J[0]], T.image_hdu[J[1]]))
        print(cmd)
        #os.system(cmd)

    T.image_hdu[J[0]] = 1
    T.image_hdu[J[1]] = 2

T.writeto('survey-ccds-snx3-25-subs.fits.gz')
