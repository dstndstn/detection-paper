import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
import pylab as plt
import numpy as np
#from scipy.ndimage.filters import gaussian_filter, correlate, correlate1d
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
import fitsio

g_det = fitsio.read('25/detmap-g.fits')
g_detiv = fitsio.read('25/detiv-g.fits')
r_det = fitsio.read('25/detmap-r.fits')
r_detiv = fitsio.read('25/detiv-r.fits')

g_det1 = fitsio.read('1d/detmap-g.fits')
g_detiv1 = fitsio.read('1d/detiv-g.fits')
r_det1 = fitsio.read('1d/detmap-r.fits')
r_detiv1 = fitsio.read('1d/detiv-r.fits')

g_sn1 = g_det1 * np.sqrt(g_detiv1)
r_sn1 = r_det1 * np.sqrt(r_detiv1)
goodpix1 = np.logical_and(g_detiv1 > 0.5 * np.median(g_detiv1),
                         r_detiv1 > 0.5 * np.median(r_detiv1))

# img = plt.imread('16/legacysurvey-custom-036450m04600-image.jpg')
# img = np.flipud(img)
# H,W,three = img.shape

# ra,dec = 36.45, -4.6
# pixscale = 0.262 / 3600.
# wcs = Tan(ra, dec, W/2.+0.5, H/2.+0.5,
#         -pixscale, 0., 0., pixscale,
#         float(W), float(H))

g_sn = g_det * np.sqrt(g_detiv)
r_sn = r_det * np.sqrt(r_detiv)
# 
# goodpix = np.logical_and(g_detiv > 0.5 * np.median(g_detiv),
#                          r_detiv > 0.5 * np.median(r_detiv))

def sedsn(detmaps, detivs, sed):
    H,W = detmaps[0].shape
    sedmap = np.zeros((H,W), np.float32)
    sediv  = np.zeros((H,W), np.float32)
    for iband in range(len(detmaps)):
        # We convert the detmap to canonical band via
        #   detmap * w
        # And the corresponding change to sig1 is
        #   sig1 * w
        # So the invvar-weighted sum is
        #    (detmap * w) / (sig1**2 * w**2)
        #  = detmap / (sig1**2 * w)
        sedmap += detmaps[iband] * detivs[iband] / sed[iband]
        sediv  += detivs [iband] / sed[iband]**2
    sedmap /= np.maximum(1e-16, sediv)
    sedsn   = sedmap * np.sqrt(sediv)
    return sedsn
# 
# detmaps = [g_det, r_det]
# detivs = [g_detiv, r_detiv]
# flat_sn = sedsn(detmaps, detivs, [1., 1.])
red_sed = [2.5, 1.]
# red_sn  = sedsn(detmaps, detivs, red_sed)
# blue_sn = sedsn(detmaps, detivs, [1., 2.5])

def detect_sources(snmap, threshold, good=None):
    hot = (snmap > threshold)
    hot = binary_dilation(hot, iterations=2)
    hot = binary_fill_holes(hot)
    blobs,nblobs = label(hot)
    print(nblobs, 'blobs')
    #print('blobs min', blobs.min(), 'max', blobs.max())
    slices = find_objects(blobs)
    px,py = [],[]
    for i,slc in enumerate(slices):
        blob_loc = blobs[slc]
        sn_loc = snmap[slc]
        imax = np.argmax((blob_loc == (i+1)) * sn_loc)
        y,x = np.unravel_index(imax, blob_loc.shape)
        y0,x0 = slc[0].start, slc[1].start
        px.append(x0+x)
        py.append(y0+y)
    px,py = np.array(px),np.array(py)
    if good is not None:
        I = np.flatnonzero(good[py,px])
        px = px[I]
        py = py[I]
    return px,py

# gx,gy = detect_sources(g_sn, 5., good=goodpix)
# rx,ry = detect_sources(r_sn, 5., good=goodpix)
# fx,fy = detect_sources(flat_sn, 5., good=goodpix)
# redx,redy = detect_sources(red_sn, 5., good=goodpix)
# 
# chix,chiy = detect_sources(np.hypot(g_sn, r_sn), 5., good=goodpix)

#c4x,c4y = detect_sources(np.hypot(g_sn, r_sn), 4., good=goodpix)

#c3x,c3y = detect_sources(np.hypot(g_sn, r_sn), 3., good=goodpix)

# Detect on the single image
c3x,c3y = detect_sources(np.hypot(g_sn1, r_sn1), 3., good=goodpix1)

# Compute the S/N required for g-only or r-only to trigger the "red" SED detector
dm=[np.array([[1,0]]), np.array([[0,1]])]
div=[np.ones(2), np.ones(2)]
sn = sedsn(dm, div, red_sed)
#print(sn)
#sn[0,0] / sn[0,1]
sng = sn[0,0]
snr = sn[0,1]

plt.figure(figsize=(5,4))
plt.subplots_adjust(right=0.95, top=0.98)

from matplotlib.patches import Circle
plt.clf()

# Annotate points as "true" or "false" based on deeper data.
real = (np.hypot(g_sn[c3y,c3x], r_sn[c3y,c3x]) >  10.)
#fake = np.flatnonzero(np.hypot(g_sn[c3y,c3x], r_sn[c3y,c3x]) <= 10.)
fake = np.logical_not(real)
plt.plot(g_sn1[c3y,c3x][real], r_sn1[c3y,c3x][real], '.', color='0.5', alpha=0.2, label='Real Peaks')
plt.plot(g_sn1[c3y,c3x][fake], r_sn1[c3y,c3x][fake], '.', color='k', alpha=0.5, label='False Peaks')

#plt.plot(g_sn[c3y,c3x], r_sn[c3y,c3x], '.', color='0.5', alpha=0.2, label='Chi-squared Peaks')
# cheat!
#plt.gca().add_artist(Circle((0, 0), 3, color='0.5'))
# chi2
a = np.linspace(0, 2.*np.pi, 100)
plt.plot(5.*np.sin(a), 5.*np.cos(a), 'b-', label='Chi-squared detection')
# r
plt.axhline(5., color='r', linestyle=':', label='r-band only detection')
# red
m=-sng/snr
b=5./snr
xx = np.array([-20,40])
plt.plot(xx, b+m*xx, 'm-', mew=2, linestyle='--', label="``Red'' SED-matched detection")
#plt.legend(loc='lower right')
plt.legend(loc='upper right')
plt.axis('square')
#plt.axis('equal')
#plt.axis([-10,20,-10,20])
plt.axis([-10,30,-10,20])
plt.xlabel('g band S/N')
plt.ylabel('r band S/N')
plt.axhline(0, color='k', alpha=0.25)
plt.axvline(0, color='k', alpha=0.25);
plt.savefig('sed-matched.pdf')
plt.savefig('sed-matched.png')

