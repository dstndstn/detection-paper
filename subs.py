from astrometry.util.fits import *
from collections import Counter

T = fits_table('survey-ccds-snx3.fits.gz')
print(len(T))

T.cut(np.array([c.strip() in ['N4', 'S4'] for c in T.ccdname]))
print(len(T), 'N4/S4')

exposures = Counter(T.expnum)
print('Number of exposures:', exposures.most_common())
T.cut(np.array([exposures[e] == 2 for e in T.expnum]))
print(len(T), 'CCDs from exposures in both N4 & S4')

# Ig = T[T.filter == 'g']

for N in [1, 4, 16, 25]: #[1,5,25]:
    Ikeep = []
    for band in ['g','r','i']:
        for chip in ['N4', 'S4']:
            I, = np.nonzero([(c.strip() == chip) and (f.strip() == band)
                             for c,f in zip(T.ccdname, T.filter)])
            print(len(I), band, chip)
            Ikeep.append(I[:N])
        print('N', N, 'band', band, ': exposures', np.unique(T.expnum[np.array(Ikeep)]))
    Ikeep = np.hstack(Ikeep)
    T[Ikeep].writeto('survey-ccds-snx3-%i.fits.gz' % N)


N = 1
Ib = []
Ic = []
Id = []
for band in ['g','r','i']:
    for chip in ['N4', 'S4']:
        I, = np.nonzero([(c.strip() == chip) and (f.strip() == band)
                         for c,f in zip(T.ccdname, T.filter)])
        print(len(I), band, chip)
        Ib.append(I[-1])
        Ic.append(I[-2])
        Id.append(I[-3])
        print('N', N, 'band', band, ': exposures', np.unique(T.expnum[np.array(Ikeep)]))

T[Ib].writeto('survey-ccds-snx3-1b.fits.gz')
T[Ic].writeto('survey-ccds-snx3-1c.fits.gz')
T[Id].writeto('survey-ccds-snx3-1d.fits.gz')
