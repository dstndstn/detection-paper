import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pylab as plt
import numpy as np
import fitsio
import sys
import time
from astrometry.util.fits import *
from astrometry.util.file import pickle_to_file, unpickle_from_file
from astrometry.util.util import Tan
from astrometry.util.plotutils import *
from astrometry.util.starutil import *
from astrometry.util.starutil_numpy import *
from astrometry.libkd.spherematch import match_xy, match_radec
from collections import Counter
from scipy.ndimage.filters import *
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
from scipy.special import erfc, logsumexp


def jpl_query(ra, dec, mjd):
    import requests
    # DECam
    latlongargs = dict(lon='70.81489', lon_u='W',
                       lat='30.16606', lat_u='S',
                       alt='2215.0', alt_u='m')
    hms = ra2hmsstring(ra, separator=':')
    dms = dec2dmsstring(dec)
    if dms.startswith('+'):
        dms = dms[1:]
    date = mjdtodate(mjd)
    print('date', date)
    date = '%i-%02i-%02i %02i:%02i:%02i' % (date.year, date.month, date.day,
                                            date.hour, date.minute, date.second)
    print('date string:', date)
    # '2016-03-01 00:42'
    s = requests.Session()
    r = s.get('https://ssd.jpl.nasa.gov/sbfind.cgi')
    #r2 = s.get('https://ssd.jpl.nasa.gov/sbfind.cgi?s_time=1')
    print('JPL lookup: setting date', date)
    r3 = s.post('https://ssd.jpl.nasa.gov/sbfind.cgi', data=dict(obs_time=date, time_zone='0', check_time='Use Specified Time'))
    print('Reply code:', r3.status_code)
    #r4 = s.get('https://ssd.jpl.nasa.gov/sbfind.cgi?s_loc=1')
    print('JPL lookup: setting location', latlongargs)
    latlongargs.update(s_pos="Use Specified Coordinates")
    r5 = s.post('https://ssd.jpl.nasa.gov/sbfind.cgi', data=latlongargs)
    print('Reply code:', r5.status_code)
    #r6 = s.get('https://ssd.jpl.nasa.gov/sbfind.cgi?s_region=1')
    print('JPL lookup: setting RA,Dec', (hms, dms))
    r7 = s.post('https://ssd.jpl.nasa.gov/sbfind.cgi', data=dict(ra_1=hms, dec_1=dms,
                                                                 ra_2='w0 0 45', dec_2='w0 0 45', sys='J2000', check_region_1="Use Specified R.A./Dec. Region"))
    print('Reply code:', r7.status_code)
    #r8 = s.get('https://ssd.jpl.nasa.gov/sbfind.cgi?s_constraint=1')
    print('JPL lookup: clearing mag limit')
    r9 = s.post('https://ssd.jpl.nasa.gov/sbfind.cgi', data=dict(group='all', limit='1000', mag_limit='', mag_required='yes', two_pass='yes', check_constraints="Use Specified Settings"))
    print('Reply code:', r9.status_code)
    print('JPL lookup: submitting search')
    r10 = s.post('https://ssd.jpl.nasa.gov/sbfind.cgi', data=dict(search="Find Objects"))
    txt = r10.text
    txt = txt.replace('<a href="sbdb.cgi', '<a href="https://ssd.jpl.nasa.gov/sbdb.cgi')
    if '<pre>' in txt:
        i0 = txt.index('<pre>')
        i1 = txt.index('</pre>', i0)
        print(txt[i0:i1])
    else:
        print(txt)

# 1
# jpl_query(36.56102704691466, -4.670403734599411, 57714.19666554)
# -> 1466 Mundleria

# 4
# jpl_query(36.43302303215179, -4.6610239398835915, 57653.18272374)
# -> 63059 2000 WA118

# 5
# jpl_query(36.453468408983994, -4.660223580431442, 57614.36879513)
# -> 43342 2000 RO67

# 6
# jpl_query(36.55642461654786, -4.655848930341187, 57636.23166751)
# -> 63059 2000 WA118

# 8
# jpl_query(36.405931058993566, -4.690934375617985, 57715.03741036)
# -> 1466 Mundleria

# 9
# jpl_query(36.557884974189555, -4.655775931800927, 57636.25212985)
# -> 63059 2000 WA118

# *** three in a row
# 10
# jpl_query(36.431927745821085, -4.660951135059059, 57653.19172575)
# -> 63059 2000 WA118

# 13 (three)
#jpl_query(36.452519159695925, -4.660078029052925, 57614.35276412)
# -> 43342 2000 RO67

# 14
# jpl_query(36.49361846690702, -4.489485785856188, 57701.0896792)
# -> NOT FOUND?

# 15
# jpl_query(36.479386652726994, -4.574272458183519, 57660.35787177)
# -> 159502 2000 WW31

# 22
# jpl_query(36.3880453003972, -4.637077573496249, 57752.08623574)
# -> 23275 2000 YP101

# 29
# jpl_query(36.58504995432683, -4.667124561799415, 57714.07199208)
# -> 1466 Mundleria

# 30
# jpl_query(36.590601384876706, -4.677021211507089, 57730.07658341)
# -> 21354 1997 FM

# 31
# jpl_query(36.3654691532076, -4.755047525778143, 57624.2981817)
# -> Just an extremely red source

# 32
# jpl_query(36.386878285125924, -4.622958594369826, 57730.07658341)
# -> Not found

# 34
# jpl_query(36.34424966987521, -4.540569264179978, 57714.07199208)
# -> Not found

# 36 jpl_query(36.329873436791395, -4.504614950467419, 57614.32912007)
# -> very red

# 39 jpl_query(36.40761412746831, -4.631184012969891, 57614.32912007)
# -> very red

# 40 jpl_query(36.58238715599062, -4.48372573941448, 57614.32912007)
# -> very red

# 41 jpl_query(36.355197857223395, -4.5616762877343335, 57624.2981817)
# -> very red

# 43 jpl_query(36.54854190675149, -4.679575558482741, 57624.2981817)
# -> very red


def sdss_rgb(imgs, bands, scales=None, m=0.03, Q=20):
    rgbscales=dict(g=(2, 6.0),
                   r=(1, 3.4),
                   i=(0, 3.0),
                   z=(0, 2.2))
    if scales is not None:
        rgbscales.update(scales)
    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[:,:,plane] = np.clip((img * scale + m) * fI / I, 0, 1)
    return rgb


def sedsn(detmaps, detivs, sed):
    H,W = detmaps[0].shape
    sedmap = np.zeros((H,W), np.float32)
    sediv  = np.zeros((H,W), np.float32)
    for detmap,detiv,s in zip(detmaps,detivs,sed):
        if s == 0:
            continue
        # We convert the detmap to canonical band via
        #   F_i = detmap_i / sed_i
        # And the corresponding change to sig1 is
        #   sig_f = sig1_i / sed_i
        # And invvar is
        #   invvar_f,i = sed_i^2 / sig1_i^2
        #   invvar_f,i = sed_i^2 * detiv_i
        # So the invvar-weighted accumulator is
        #   F_i * invvar_f,i = (detmap_i / sed_i) * (sed_i^2 * detiv_i)
        #                    = detmap_i * detiv_i * sed_i
        sedmap += detmap * detiv * s
        sediv  += detiv  * s**2
    sedmap /= np.maximum(1e-16, sediv)
    sedsn   = sedmap * np.sqrt(sediv)
    return sedsn

def colorsed(gr, ri):
    return np.array([10.**(-0.4*gr), 1., 10.**(-0.4*-ri)])

class SED(object):
    def __init__(self, name, plotcolor, plotsym, sed):
        self.name = name
        self.plotcolor = plotcolor
        self.plotsym = plotsym
        self.sed = sed
        self.tname = name.lower().replace('-','_') + '_sn'
    def __repr__(self):
        return self.name + ': ' + str(self.sed)

def detect_sources(snmap, threshold):
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
        #if i == 0:
        #    plt.subplot(2,2,1)
        #    plt.imshow(blob_loc, interpolation='nearest', origin='lower')
        #    plt.colorbar()
        #    plt.subplot(2,2,2)
        #    plt.imshow((blob_loc==(i+1))*sn_loc, interpolation='nearest', origin='lower')
        #    plt.subplot(2,2,3)
        #    plt.plot(x, y, 'ro')
    return np.array(px),np.array(py)



def sed_matched_figs(detect_sn, good, img, sedlist, DES, g_det, r_det, i_det,
                     wcs):
    x,y = detect_sources(detect_sn, 100.)
    sources = fits_table()
    sources.x = x
    sources.y = y
    sources.cut(good[sources.y, sources.x])
    print('Cut to', len(sources), 'good sources')
    sz = 20
    H,W = good.shape
    sources.cut((sources.x > sz) * (sources.y > sz) *
                (sources.x < (W-sz)) * (sources.y < (H-sz)))
    print(len(sources), 'not near edges')

    for s in sedlist:
        sources.set(s.tname, s.snmap[sources.y, sources.x])

    # sources.g_sn = (g_det[sources.y, sources.x] * np.sqrt(g_detiv[sources.y, sources.x]))
    # sources.r_sn = (r_det[sources.y, sources.x] * np.sqrt(r_detiv[sources.y, sources.x]))
    # sources.i_sn = (i_det[sources.y, sources.x] * np.sqrt(i_detiv[sources.y, sources.x]))
    sources.g_flux = g_det[sources.y, sources.x]
    sources.r_flux = r_det[sources.y, sources.x]
    sources.i_flux = i_det[sources.y, sources.x]
    sources.ra,sources.dec = wcs.pixelxy2radec(sources.x+1, sources.y+1)
    sources.g_mag = -2.5*(np.log10(sources.g_flux) - 9)
    sources.r_mag = -2.5*(np.log10(sources.r_flux) - 9)
    sources.i_mag = -2.5*(np.log10(sources.i_flux) - 9)
    sources.imax = np.argmax(np.vstack([sources.get(s.tname) for s in sedlist]), axis=0)

    plt.figure(figsize=(5,4))
    plt.subplots_adjust(left=0.15, right=0.97, bottom=0.12, top=0.98)
    
    plt.clf()
    for i,s in enumerate(sedlist):
        if not np.all(s.sed > 0):
            continue
        I = np.flatnonzero(sources.imax == i)
        plt.plot(sources.g_mag[I] - sources.r_mag[I],
                 sources.r_mag[I] - sources.i_mag[I],
                 s.plotsym, label=s.name, color=s.plotcolor, alpha=0.5,
                 mfc='none', ms=5)
        gr = -2.5 * np.log10(s.sed[0] / s.sed[1])
        ri = -2.5 * np.log10(s.sed[1] / s.sed[2])
        plt.plot(gr, ri, 'o', color='k', mfc='none', ms=8, mew=3)
    plt.axis([-0.5, 2.5, -0.5, 2])
    plt.xlabel('g - r (mag)')
    plt.ylabel('r - i (mag)')
    plt.legend(loc='upper left')
    plt.savefig('best-color.pdf')
    
    
    plt.figure(figsize=(4,4))
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    
    plt.clf()
    xlo,xhi = 500,1100
    ylo,yhi = 500,1100
    plt.imshow(img[ylo:yhi, xlo:xhi], origin='lower', interpolation='nearest',
               extent=[xlo,xhi,ylo,yhi])
    ax = plt.axis()
    
    for i,s in enumerate(sedlist):
        if not np.all(s.sed > 0):
            continue
        I = np.flatnonzero((sources.imax == i) * 
                           (sources.x >= xlo) * (sources.x <= xhi) *
                           (sources.y >= ylo) * (sources.y <= yhi))
        print(len(I), s.name)
        plt.plot(sources.x[I], sources.y[I],
                 s.plotsym, label=s.name, color=s.plotcolor, alpha=0.5,
                 mfc='none', ms=15)
    plt.axis(ax)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('image-sources.pdf')

    for i,s in enumerate(sedlist):
        I = np.flatnonzero(sources.imax == i)
        J = np.argsort(-sources.get(s.tname)[I])
        plt.clf()
        show_sources(sources[I[J]], img)
        plt.savefig('best-%s.pdf' % s.name.lower())

    #####  Run detection at different thresholds ####
    tsedlist = []
    for i,s in enumerate(sedlist):
        if not np.all(s.sed > 0):
            continue
        tsedlist.append(s)
    snmap = None
    for i,s in enumerate(tsedlist):
        if snmap is None:
            snmap = s.snmap
        else:
            snmap = np.maximum(snmap, s.snmap)
    for thresh in [30]: #10, 30, 100]:
        x,y = detect_sources(snmap, thresh)
        tsources = fits_table()
        tsources.x = x
        tsources.y = y
        #tsources.cut(good[tsources.y, tsources.x])
        print('Threshold', thresh)
        print('Cut to', len(tsources), 'good sources')
        sz = 20
        H,W = good.shape
        tsources.cut((tsources.x > sz) * (tsources.y > sz) *
                    (tsources.x < (W-sz)) * (tsources.y < (H-sz)))
        print(len(tsources), 'not near edges')

        for s in tsedlist:
            tsources.set(s.tname, s.snmap[tsources.y, tsources.x])
        tsources.g_flux = g_det[tsources.y, tsources.x]
        tsources.r_flux = r_det[tsources.y, tsources.x]
        tsources.i_flux = i_det[tsources.y, tsources.x]
        tsources.ra,tsources.dec = wcs.pixelxy2radec(tsources.x+1, tsources.y+1)
        tsources.g_mag = -2.5*(np.log10(tsources.g_flux) - 9)
        tsources.r_mag = -2.5*(np.log10(tsources.r_flux) - 9)
        tsources.i_mag = -2.5*(np.log10(tsources.i_flux) - 9)
        tsources.imax = np.argmax(np.vstack([tsources.get(s.tname)
                                             for s in tsedlist]), axis=0)
        plt.clf()
        xlo,xhi = 500,1100
        ylo,yhi = 500,1100
        plt.imshow(img[ylo:yhi, xlo:xhi], origin='lower', interpolation='nearest',
                   extent=[xlo,xhi,ylo,yhi])
        ax = plt.axis()
        for i,s in enumerate(tsedlist):
            I = np.flatnonzero((tsources.imax == i) * 
                               (tsources.x >= xlo) * (tsources.x <= xhi) *
                               (tsources.y >= ylo) * (tsources.y <= yhi))
            print(len(I), s.name)
            plt.plot(tsources.x[I], tsources.y[I],
                     s.plotsym, label=s.name, color=s.plotcolor, alpha=0.5,
                     mfc='none', ms=15, mew=2)
        plt.axis(ax)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('image-sources-%i.pdf' % thresh)

        # boundary = (snmap > thresh)
        # boundary = np.logical_xor(boundary, binary_dilation(boundary,
        #                                                     structure=np.ones((3,3))))
        # rgb = img[ylo:yhi, xlo:xhi].copy()
        # rgb[:,:,1][boundary[ylo:yhi, xlo:xhi]] = 255
        # plt.clf()
        # plt.imshow(rgb, origin='lower', interpolation='nearest',
        #            extent=[xlo,xhi,ylo,yhi])
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('image-blobs-%i.pdf' % thresh)
        
        plt.figure(figsize=(5,4))
        plt.subplots_adjust(left=0.15, right=0.97, bottom=0.12, top=0.98)
        plt.clf()
        lp,lt = [],[]
        for i,s in enumerate(tsedlist):
            I = np.flatnonzero(tsources.imax == i)
            plt.plot(tsources.g_mag[I] - tsources.r_mag[I],
                     tsources.r_mag[I] - tsources.i_mag[I],
                     s.plotsym, color=s.plotcolor, alpha=0.2,
                     mfc='none', ms=5)
            # For the legend
            p = plt.plot(-1, -1, s.plotsym, color=s.plotcolor, mfc='none', ms=5)
            lp.append(p[0])
            lt.append(s.name)
            
            gr = -2.5 * np.log10(s.sed[0] / s.sed[1])
            ri = -2.5 * np.log10(s.sed[1] / s.sed[2])
            plt.plot(gr, ri, 'o', color='k', mfc='none', ms=8, mew=3)
        plt.axis([-0.5, 2.5, -0.5, 2])
        plt.xlabel('g - r (mag)')
        plt.ylabel('r - i (mag)')
        plt.legend(lp, lt, loc='upper left')
        plt.savefig('best-color-%i.pdf' % thresh)

    ##############################
    
    
    
    # Artifacts from single-band detections
    I = np.hstack((np.flatnonzero(sources.imax == 3)[:6],
                   np.flatnonzero(sources.imax == 4)[:18],
                   np.flatnonzero(sources.imax == 5)[:12]))
    
    plt.figure(figsize=(4,4))
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    plt.clf()
    show_sources(sources[I], img, R=6, C=6, sz=30, divider=1)
    plt.savefig('singleband.pdf')

    MI,MJ,d = match_radec(sources.ra, sources.dec, DES.ra, DES.dec, 1./3600, nearest=True)
    print(len(MI), 'matches')
    MDES = DES[MJ]
    Msources = sources[MI]
    
    ## FIXME -- select only isolated stars?
    colorbins = np.linspace(-0.5, 4.0, 10)
    II = []
    K = []
    DES.gi = DES.mag_auto_g - DES.mag_auto_i
    for clo,chi in zip(colorbins, colorbins[1:]):
        C = np.flatnonzero((DES.gi >= clo) * (DES.gi < chi))
        minmag = np.vstack((DES.mag_auto_g, DES.mag_auto_r, DES.mag_auto_i)).max(axis=0)[C]
        C = C[np.argsort(np.abs(minmag - 17.9))]
        C = C[DES.spread_model_r[C] < 0.01]
        II.extend(C[:10])
        K.append(C[0])
    
    fw,fh = 6,4
    sl,sr,sb,st = 0.15, 0.98, 0.12, 0.98
    plt.figure(figsize=(fw,fh))
    plt.subplots_adjust(left=sl, right=sr, bottom=sb, top=st)
    plt.clf()
    plt.axhline(1., color='orange', lw=5)
    plt.axhline(1., color='k', alpha=0.5)

    plt.axvline(0., color='b', lw=2, alpha=0.2)
    plt.axvline(1.3, color='orange', lw=2, alpha=0.2)
    plt.axvline(2.5, color='r', lw=2, alpha=0.2)

    plt.plot(MDES.mag_auto_g - MDES.mag_auto_i, Msources.blue_sn / Msources.yellow_sn, 'bD', alpha=0.3,
            label='Blue SED-matched filter', ms=3)
    plt.plot(MDES.mag_auto_g - MDES.mag_auto_i, Msources.red_sn  / Msources.yellow_sn, 'rs', alpha=0.5,
            label='Red SED-matched filter', ms=3)
    plt.xlabel('DES g - i color (mag)')
    plt.ylabel('Relative strength of SED filter vs Yellow')
    plt.legend(loc='upper left')

    # Position the postage stamp images just right...
    # axes width,height
    w = fw * (sr-sl)
    h = fh * (st-sb)
    n = len(colorbins)-1
    # image size
    s = w/n
    # fraction of vertical axis devoted to image
    fim = s/h
    # fraction devoted to plot
    fplot = 1.-fim
    # scale
    ys = (1.3 - 0.7) / fplot
    # lower limit
    ymin = 1.3 - ys
    # image top
    ymax = ymin + ys * fim
    
    ax = [-0.5, 4.0, ymin, 1.3]
    plt.axis(ax)
    aspect = plt.gca().get_aspect()
    
    for clo,chi,k in zip(colorbins, colorbins[1:], K):
        x,y = DES.x[k], DES.y[k]
        plt.imshow(img[y-sz:y+sz+1, x-sz:x+sz+1], interpolation='nearest', origin='lower',
                  extent=[clo,chi,ymin,ymax], zorder=20)
    plt.yticks([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    plt.axis(ax)
    plt.gca().set_aspect(aspect)
    plt.savefig('strength.pdf')
    
def show_sources(T, img, R=10, C=10, sz=10, divider=0):
    imgrows = []
    k = 0
    for i in range(R):
        imgrow = []
        for j in range(C):
            if k >= len(T):
                sub = np.zeros((sz*2+1,sz*2+1,3), np.uint8)
            else:
                f = T[k]
                sub = img[f.y-sz : f.y+sz+1, f.x-sz : f.x+sz+1, :]
            imgrow.append(sub)
            if divider and j < C-1:
                imgrow.append(np.zeros((sz*2+1, divider, 3), np.uint8) + 255)
            k += 1
        imgrow = np.hstack(imgrow)
        #print('imgrow', imgrow.shape)
        imgrows.append(imgrow)
        if divider and i < R-1:
            rh,rw,three = imgrow.shape
            imgrows.append(np.zeros((divider, rw, 3), np.uint8) + 255)
        
    imgrows = np.vstack(reversed(imgrows))
    plt.imshow(imgrows, interpolation='nearest', origin='lower')
    plt.xticks([]); plt.yticks([])


#### Bayesian SED-matched detection

def log_pratio_bayes(seds, weights, D, Div, alpha):
    '''
    N: # SEDs
    J: # bands
    H,W: # pixels

    seds: N x J
    weights: N
    D: J x H x W
    Div: J x H x W

    D is detection map
    Div is detection map inverse-variance
    '''
    J,H,W = D.shape
    N = len(weights)
    assert(seds.shape == (N,J))
    assert(weights.shape == (N,))
    assert(Div.shape == (J,H,W))

    terms = np.empty((N,H,W), np.float32)
    terms[:,:,:] = -1e6
    for i in range(N):
        # sum over bands j
        a_i = alpha - np.sum(D * seds[i,:,np.newaxis,np.newaxis] * Div, axis=0)
        assert(a_i.shape == (H,W))
        # sum over bands j
        b_i = 0.5 * np.sum(seds[i,:,np.newaxis,np.newaxis]**2 * Div, axis=0)
        assert(b_i.shape == (H,W))
        beta_i = 2 * np.sqrt(b_i)
        ok = np.nonzero(b_i)
        c_i = a_i[ok] / beta_i[ok]
        terms[i,:,:][ok] = np.log(weights[i] / beta_i[ok]) + np.log(erfc(c_i)) + c_i**2
        print('.', end='')
    print()
    lse = logsumexp(terms, axis=0)
    return lse + np.log(alpha * np.sqrt(np.pi))

def bayes_figs(DES, detmaps, detivs, good, wcs, img):

    # First, build empirical SED prior "library" from DES sources
    DES.flux_g = np.maximum(0, DES.flux_auto_g)
    DES.flux_r = np.maximum(0, DES.flux_auto_r)
    DES.flux_i = np.maximum(0, DES.flux_auto_i)
    flux = DES.flux_g + DES.flux_r + DES.flux_i
    K = np.flatnonzero(flux > 0)
    DES.cut(K)
    flux = flux[K]
    DES.f_g = DES.flux_g / flux
    DES.f_r = DES.flux_r / flux
    DES.f_i = DES.flux_i / flux
    print('Kept', len(DES), 'with positive flux')

    nbins=21
    edge = 1. / (nbins-1) / 2.
    #N,xe,ye = loghist(DES.f_g, DES.f_r, range=((0-edge,1+edge),(0-edge,1+edge)), nbins=nbins);
    N,xe,ye = np.histogram2d(DES.f_g, DES.f_r,
                             range=((0-edge,1+edge),(0-edge,1+edge)),
                             bins=nbins)
    N = N.T

    print(np.sum(N > 0), np.sum(N > N.sum()*0.001))
    NN = N.copy()
    NN[N < N.sum()*0.001] = np.nan

    plt.figure(figsize=(3,3))
    plt.subplots_adjust(left=0.2, right=0.98, bottom=0.15, top=0.98)

    plt.clf()
    x0,x1 = -edge, 1+edge
    y0,y1 = x0,x1
    plt.imshow(NN, interpolation='nearest', origin='lower', extent=(x0,x1,y0,y1),
               cmap=antigray, zorder=20)
    plt.plot([x0, x0, x1, x0], [y0,y1,y0,y0], 'k-', zorder=30)
    plt.gca().set_frame_on(False)
    p = Polygon(np.array([[x0, x0, x1, x0], [y0, y1, y0, y0]]).T, color=(0.9,0.9,1),
                zorder=15)
    plt.gca().add_artist(p)
    plt.xlabel('flux fraction g')
    plt.ylabel('flux fraction r')
    plt.savefig('bayes-prior-sed.pdf')

    #iy,ix = np.nonzero(N)
    iy,ix = np.nonzero(N > N.sum()*0.001)
    print(len(iy), 'significant bins')
    # Find f_{g,r} histogram midpoints
    mx = (xe[:-1] + xe[1:]) / 2.
    my = (ye[:-1] + ye[1:]) / 2.
    fg = mx[ix]
    fr = my[iy]
    fi = 1. - (fg + fr)
    fn = N[iy,ix]
    seds = np.clip(np.vstack((fg,fr,fi)).T, 0., 1.)
    weights = fn / np.sum(fn)
    ok = np.flatnonzero(np.sum(seds, axis=1) == 1)
    seds = seds[ok,:]
    weights = weights[ok]
    print(len(weights), 'color-color bins are populated; max weight', weights.max())

    plt.clf()
    plothist(DES.mag_auto_g - DES.mag_auto_r, DES.mag_auto_r - DES.mag_auto_i,
             range=((-0.5, 3),(-0.5, 2)), nbins=20,
             imshowargs=dict(cmap=antigray), dohot=False, docolorbar=False)
    plt.xlabel('g - r (mag)')
    plt.ylabel('r - i (mag)')
    plt.savefig('bayes-data-cc.pdf')

    plt.clf()
    fg_grid,fr_grid = np.meshgrid(xe, ye)
    fi_grid = 1. - (fg_grid + fr_grid)
    gr_grid = -2.5 * np.log10(fg_grid / fr_grid)
    ri_grid = -2.5 * np.log10(fr_grid / fi_grid)
    good_grid = (fi_grid >= 0) * np.isfinite(gr_grid) * np.isfinite(ri_grid)
    h,w = good_grid.shape
    cm = antigray
    ng = 0
    for j0 in range(h-1):
        for i0 in range(w-1):
            j1 = j0+1
            i1 = i0+1
            if not(good_grid[j0,i0] and good_grid[j0,i1] and good_grid[j1,i0] and good_grid[j1,i1]):
                continue
            if N[j0,i0] == 0:
                continue
            ng += N[j0,i0]
            xx = [gr_grid[j0,i0], gr_grid[j0,i1], gr_grid[j1,i1], gr_grid[j1,i0], gr_grid[j0,i0]]
            yy = [ri_grid[j0,i0], ri_grid[j0,i1], ri_grid[j1,i1], ri_grid[j1,i0], ri_grid[j0,i0]]
            xy = np.vstack((xx, yy)).T
            poly = Polygon(xy, color=cm(N[j0,i0]/N.max()))
            plt.gca().add_artist(poly)
    #plt.gca().set_fc('0.5')
    plt.axis([-0.5, 3, -0.5,2])
    plt.xlabel('g - r (mag)')
    plt.ylabel('r - i (mag)')
    print(ng, 'of', N.sum(), 'plotted')
    plt.savefig('bayes-prior-cc.pdf')

    H,W = detmaps[0].shape
    bayesfn = 'lprb.pickle'
    if os.path.exists(bayesfn):
        print('Reading cached', bayesfn)
        lprb = unpickle_from_file(bayesfn)
    else:
        # Number of bands
        J = 3
        # Build detection-map & iv arrays
        D = np.zeros((J,H,W))
        Div = np.zeros((J,H,W))
        for i,(d,div) in enumerate(zip(detmaps,detivs)):
            D[i,:,:] = d
            Div[i,:,:] = div
        alpha = 1.
        t0 = time.process_time()
        lprb = log_pratio_bayes(seds, weights, D, Div, alpha)
        t1 = time.process_time()
        print('Bayes took', t1-t0, 'CPU-seconds')
        pickle_to_file(lprb, bayesfn)
        print('Writing cache', bayesfn)
        # Bayes took 276.118402 CPU-seconds

    sz = 20
    bx,by = detect_sources(lprb, 2000)
    bsources = fits_table()
    bsources.x = bx
    bsources.y = by
    bsources.lprb = lprb[bsources.y, bsources.x]
    bsources.cut((bsources.x > sz) * (bsources.x < (W-sz)) * (bsources.y > sz) * (bsources.y < (H-sz)))
    bsources.cut(good[bsources.y, bsources.x])
    print('Kept', len(bsources))

    g_det, r_det, i_det = detmaps
    
    bsources.g_flux = g_det[bsources.y, bsources.x]
    bsources.r_flux = r_det[bsources.y, bsources.x]
    bsources.i_flux = i_det[bsources.y, bsources.x]
    bsources.ra,bsources.dec = wcs.pixelxy2radec(bsources.x+1, bsources.y+1)
    bsources.g_mag = -2.5*(np.log10(bsources.g_flux) - 9)
    bsources.r_mag = -2.5*(np.log10(bsources.r_flux) - 9)
    bsources.i_mag = -2.5*(np.log10(bsources.i_flux) - 9)
    bsources.gr = bsources.g_mag - bsources.r_mag
    bsources.ri = bsources.r_mag - bsources.i_mag
    I = np.argsort(-bsources.lprb)
    bsources.cut(I)

    # save for later...
    bsources_orig = bsources.copy()
    
    # g + r + i detections
    xg,yg = detect_sources(detmaps[0] * np.sqrt(detivs[0]), 50.)
    xr,yr = detect_sources(detmaps[1] * np.sqrt(detivs[1]), 50.)
    xi,yi = detect_sources(detmaps[2] * np.sqrt(detivs[2]), 50.)
    print('Detected', len(xg),len(xr),len(xi), 'gri')
    xm,ym = xg.copy(),yg.copy()
    for xx,yy in [(xr,yr),(xi,yi)]:
        I,J,d = match_xy(xm,ym, xx,yy, 5.)
        print('Matched:', len(I))
        U = np.ones(len(xx), bool)
        U[J] = False
        print('Unmatched:', np.sum(U))
        xm = np.hstack((xm, xx[U]))
        ym = np.hstack((ym, yy[U]))
    print('Total of', len(xm), 'g+r+i')

    sources = fits_table()
    sources.x = xm
    sources.y = ym
    iy,ix = np.round(sources.y).astype(int), np.round(sources.x).astype(int)
    sources.sn_g = (detmaps[0] * np.sqrt(detivs[0]))[iy,ix]
    sources.sn_r = (detmaps[1] * np.sqrt(detivs[1]))[iy,ix]
    sources.sn_i = (detmaps[2] * np.sqrt(detivs[2]))[iy,ix]
    sources.g_flux = g_det[sources.y, sources.x]
    sources.r_flux = r_det[sources.y, sources.x]
    sources.i_flux = i_det[sources.y, sources.x]
    sources.ra,sources.dec = wcs.pixelxy2radec(sources.x+1, sources.y+1)
    sources.g_mag = -2.5*(np.log10(sources.g_flux) - 9)
    sources.r_mag = -2.5*(np.log10(sources.r_flux) - 9)
    sources.i_mag = -2.5*(np.log10(sources.i_flux) - 9)
    sources.gr = sources.g_mag - sources.r_mag
    sources.ri = sources.r_mag - sources.i_mag
    sources.sn_max = np.maximum(sources.sn_g, np.maximum(sources.sn_r, sources.sn_i))
    sources.cut((sources.x > sz) * (sources.x < (W-sz)) * (sources.y > sz) * (sources.y < (H-sz)))
    sources.cut(good[sources.y, sources.x])
    print('Kept', len(sources))
    I = np.argsort(-sources.sn_max)
    sources.cut(I)

    sources_orig = sources.copy()
    
    N = min(len(sources), len(bsources))
    sources  =  sources[:N]
    bsources = bsources[:N]
    print('Cut both to', N)
    
    # Define unmatched sources as ones that are not in the other's hot region
    hot = binary_fill_holes(np.logical_or(detmaps[0] * np.sqrt(detivs[0]) > 50.,
                            np.logical_or(detmaps[1] * np.sqrt(detivs[1]) > 50.,
                                          detmaps[2] * np.sqrt(detivs[2]) > 50.)))
    Bhot = binary_fill_holes(lprb > 2000.)
    UB = np.flatnonzero((hot[bsources.y, bsources.x] == False))
    US = np.flatnonzero((Bhot[sources.y, sources.x] == False))
    print(len(UB), 'unmatched Bayesian', len(US), 'g+r+i')
    
    plt.figure(figsize=(6,4))
    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.12, top=0.98)

    plt.clf()
    plt.plot(sources.gr[US],  sources.ri[US], 'o', mec='r', mfc='none',
             label='g+r+i only')
    plt.plot(bsources.gr[UB], bsources.ri[UB], 'kx',
             label='Bayesian only')
    plt.xlabel('g - r (mag)')
    plt.ylabel('r - i (mag)')
    plt.legend()
    plt.axis([-5, 5, -3, 3])
    plt.savefig('bayes-vs-gri.pdf')

    plt.figure(figsize=(4,4))
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    plt.clf()
    show_sources(bsources[UB], img, R=10, C=10, divider=1)
    #plt.suptitle('Bayesian only');
    plt.savefig('bayes-only.pdf')

    plt.clf()
    show_sources(sources[US], img, R=10, C=10, divider=1)
    #plt.suptitle('Bayesian only');
    plt.savefig('gri-only.pdf')


    #####
    sources = sources_orig.copy()
    
    # What if we demand 2-band detection?
    #bsources = bsources_orig
    # Detect at a higher threshold to get ~2400 sources
    Bhot = binary_fill_holes(lprb > 4000.)
    bx,by = detect_sources(lprb, 4000.)
    bsources = fits_table()
    bsources.x = bx
    bsources.y = by
    bsources.lprb = lprb[bsources.y, bsources.x]
    bsources.cut((bsources.x > sz) * (bsources.x < (W-sz)) * (bsources.y > sz) * (bsources.y < (H-sz)))
    bsources.cut(good[bsources.y, bsources.x])
    print('Kept', len(bsources))
    bsources.g_flux = g_det[bsources.y, bsources.x]
    bsources.r_flux = r_det[bsources.y, bsources.x]
    bsources.i_flux = i_det[bsources.y, bsources.x]
    bsources.ra,bsources.dec = wcs.pixelxy2radec(bsources.x+1, bsources.y+1)
    bsources.g_mag = -2.5*(np.log10(bsources.g_flux) - 9)
    bsources.r_mag = -2.5*(np.log10(bsources.r_flux) - 9)
    bsources.i_mag = -2.5*(np.log10(bsources.i_flux) - 9)
    bsources.gr = bsources.g_mag - bsources.r_mag
    bsources.ri = bsources.r_mag - bsources.i_mag
    I = np.argsort(-bsources.lprb)
    bsources.cut(I)

    sthresh = 50.
    hotg = binary_fill_holes(detmaps[0] * np.sqrt(detivs[0]) > sthresh)
    hotr = binary_fill_holes(detmaps[1] * np.sqrt(detivs[1]) > sthresh)
    hoti = binary_fill_holes(detmaps[2] * np.sqrt(detivs[2]) > sthresh)
    hotsum = hotg*1 + hotr*1 + hoti*1
    hot = (hotsum >= 2)
    K = np.flatnonzero(hot[sources.y,sources.x])
    # 2386 detected in 2 or more bands
    print(len(K), 'detected in 2 or more bands')
    sources.cut(K)

    N = min(len(sources), len(bsources))
    sources  =  sources[:N]
    bsources = bsources[:N]
    print('Cut both to', N)

    UB = np.flatnonzero((hot[bsources.y, bsources.x] == False))
    US = np.flatnonzero((Bhot[sources.y, sources.x] == False))
    print(len(UB), 'unmatched Bayesian', len(US), '2 of (g+r+i)')

    plt.figure(figsize=(6,4))
    plt.subplots_adjust(left=0.1, right=0.98, bottom=0.12, top=0.98)

    plt.clf()
    plt.plot(sources.gr[US],  sources.ri[US], 'o', mec='r', mfc='none',
             label='g+r+i only')
    plt.plot(bsources.gr[UB], bsources.ri[UB], 'kx',
             label='Bayesian only')
    plt.xlabel('g - r (mag)')
    plt.ylabel('r - i (mag)')
    plt.legend()
    plt.axis([-5, 5, -3, 3])
    plt.savefig('bayes-vs-2gri.pdf')

    plt.figure(figsize=(4,4))
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    plt.clf()
    show_sources(bsources[UB], img, R=10, C=10, divider=1)
    plt.savefig('bayes-2only.pdf')

    plt.clf()
    show_sources(sources[US], img, R=10, C=10, divider=1)
    plt.savefig('2gri-only.pdf')
    

    
#### Galaxy detection
def galaxy_figs(sedlist, good, wcs, img):
    s = '1.0'
    #s = 're0.7'
    g_galdet = fitsio.read('25/galdetmap-'+s+'-g.fits')
    g_galdetiv = fitsio.read('25/galdetiv-'+s+'-g.fits')
    r_galdet = fitsio.read('25/galdetmap-'+s+'-r.fits')
    r_galdetiv = fitsio.read('25/galdetiv-'+s+'-r.fits')
    i_galdet = fitsio.read('25/galdetmap-'+s+'-i.fits')
    i_galdetiv = fitsio.read('25/galdetiv-'+s+'-i.fits')
    gdetmaps = [g_galdet, r_galdet, i_galdet]
    gdetivs = [g_galdetiv, r_galdetiv, i_galdetiv]
    for s in sedlist:
        s.galsnmap = sedsn(gdetmaps, gdetivs, s.sed)
    yellow_gal = sedlist[1].galsnmap
    x,y = detect_sources(yellow_gal, 100.)
    
    gals = fits_table()
    gals.x = x
    gals.y = y
    gals.cut(good[gals.y, gals.x])
    print('Cut to', len(gals), 'good gals')
    sz = 20
    H,W = good.shape
    gals.cut((gals.x > sz) * (gals.y > sz) * (gals.x < (W-sz)) * (gals.y < (H-sz)))
    print(len(gals), 'not near edges')
    #gals.cut((g_galdetiv[gals.y, gals.x] > 0) * (r_galdetiv[gals.y, gals.x] > 0) * (i_galdetiv[gals.y, gals.x] > 0))
    #print(len(gals), 'with gri obs')
    sns = []
    for s in sedlist:
        gals.set(s.tname, s.galsnmap[gals.y, gals.x])
        gals.set(s.tname+'_psf', s.snmap[gals.y, gals.x])
        sns.append(s.galsnmap[gals.y, gals.x])
    gals.sn_max = np.max(np.vstack(sns), axis=0)
    # These are PSF fluxes/mags
    # gals.g_sn = (g_det[gals.y, gals.x] * np.sqrt(g_detiv[gals.y, gals.x]))
    # gals.r_sn = (r_det[gals.y, gals.x] * np.sqrt(r_detiv[gals.y, gals.x]))
    # gals.i_sn = (i_det[gals.y, gals.x] * np.sqrt(i_detiv[gals.y, gals.x]))
    # gals.g_flux = g_det[gals.y, gals.x]
    # gals.r_flux = r_det[gals.y, gals.x]
    # gals.i_flux = i_det[gals.y, gals.x]
    gals.ra,gals.dec = wcs.pixelxy2radec(gals.x+1, gals.y+1)
    # gals.g_mag = -2.5*(np.log10(gals.g_flux) - 9)
    # gals.r_mag = -2.5*(np.log10(gals.r_flux) - 9)
    # gals.i_mag = -2.5*(np.log10(gals.i_flux) - 9)
    # gals.g_galflux = g_galdet[gals.y, gals.x]
    # gals.r_galflux = r_galdet[gals.y, gals.x]
    # gals.i_galflux = i_galdet[gals.y, gals.x]
    # gals.g_galmag = -2.5*(np.log10(gals.g_galflux) - 9)
    # gals.r_galmag = -2.5*(np.log10(gals.r_galflux) - 9)
    # gals.i_galmag = -2.5*(np.log10(gals.i_galflux) - 9)
    #I = np.argsort(-gals.yellow_sn)
    #gals.cut(I)

    plt.figure(figsize=(4,4))
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    plt.clf()
    I = np.argsort(-(gals.yellow_sn - gals.yellow_sn_psf))
    show_sources(gals[I], img, sz=20)
    plt.savefig('galaxies.pdf')
    
    plt.figure(figsize=(5,4))
    plt.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.98)
    plt.clf()
    plt.semilogx(gals.yellow_sn, gals.yellow_sn / gals.yellow_sn_psf, 'k.', alpha=0.25)
    plt.ylim(0.9, 1.2)
    plt.xlabel('Galaxy detection S/N')
    plt.ylabel('Galaxy / PSF detection ratio')
    plt.savefig('galaxies-relsn.pdf')


def main():
    g_det = fitsio.read('25/detmap-g.fits')
    g_detiv = fitsio.read('25/detiv-g.fits')
    r_det = fitsio.read('25/detmap-r.fits')
    r_detiv = fitsio.read('25/detiv-r.fits')
    i_det = fitsio.read('25/detmap-i.fits')
    i_detiv = fitsio.read('25/detiv-i.fits')

    detmaps = [g_det, r_det, i_det]
    detivs = [g_detiv, r_detiv, i_detiv]

    Ng = fitsio.read('25/legacysurvey-custom-036450m04600-nexp-g.fits.fz')
    Nr = fitsio.read('25/legacysurvey-custom-036450m04600-nexp-r.fits.fz')
    Ni = fitsio.read('25/legacysurvey-custom-036450m04600-nexp-i.fits.fz')
    good = ((Ng >= 12) * (Nr >= 12) * (Ni >= 12))

    gco = fitsio.read('25/legacysurvey-custom-036450m04600-image-g.fits.fz')
    rco = fitsio.read('25/legacysurvey-custom-036450m04600-image-r.fits.fz')
    ico = fitsio.read('25/legacysurvey-custom-036450m04600-image-i.fits.fz')
    s = 4
    scale = dict(g=(2, 6.0*s), r=(1, 3.4*s), i=(0, 3.0*s))
    img = sdss_rgb([gco,rco,ico], 'gri', scales=scale)
    img = (np.clip(img, 0, 1) * 255.).astype(np.uint8)
    H,W,three = img.shape

    ra,dec = 36.45, -4.6
    pixscale = 0.262 / 3600.
    wcs = Tan(ra, dec, W/2.+0.5, H/2.+0.5, -pixscale, 0., 0., pixscale,
              float(W), float(H))

    # plt.figure(figsize=(4,4))
    # plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    # plt.clf()
    # plt.imshow(img[500:1100, 500:1100], origin='lower', interpolation='nearest')
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig('image.pdf')

    sedlist = [
        SED('Blue',   'c',      'D', colorsed(0., 0.)),
        SED('Yellow', 'orange', 'o', colorsed(1., 0.3)),
        SED('Red',    'r',      's', colorsed(1.5, 1.)),
        SED('g-only', 'g',      '^', np.array([1., 0., 0.])),
        SED('r-only', 'pink',   'v', np.array([0., 1., 0.])),
        SED('i-only', 'm',      '*', np.array([0., 0., 1.])),
    ]
    for s in sedlist:
        print('%8s' % s.name, '   '.join(['%6.3f' % x for x in s.sed]))
    for s in sedlist:
        s.snmap = sedsn(detmaps, detivs, s.sed)

    # Use Yellow to do the actual detection
    detect_sn = sedlist[1].snmap

    # Read DES catalog in region
    '''
    https://des.ncsa.illinois.edu/easyweb/db-access

    SELECT RA, DEC,
    MAG_AUTO_G, MAG_AUTO_R, MAG_AUTO_I,
    FLUX_AUTO_G, FLUX_AUTO_R, FLUX_AUTO_I,
    FLAGS_G, FLAGS_R, FLAGS_I,
    SPREAD_MODEL_R
    from DR1_MAIN
    where RA between 36.3 and 36.6
    and DEC between -4.76 and -4.44

    -> des-db-4.fits
    '''
    DES = fits_table('des-db-4.fits')
    print(len(DES), 'DES')
    DES.cut((DES.flags_g < 4) * (DES.flags_r < 4) * (DES.flags_i < 4))
    print(len(DES), 'un-flagged')
    ok,x,y = wcs.radec2pixelxy(DES.ra, DES.dec)
    DES.x = (x-1).astype(np.int)
    DES.y = (y-1).astype(np.int)

    # Cut of DES catalog for the SED-matched filter figure
    sz = 20
    Ides = np.flatnonzero((DES.x > sz) * (DES.y > sz) *
                          (DES.x < (W-sz)) * (DES.y < (H-sz)) *
                          good[np.clip(DES.y, 0, H-1), np.clip(DES.x, 0, W-1)])

    sed_matched_figs(detect_sn, good, img, sedlist, DES[Ides],
                     g_det, r_det, i_det, wcs)

    #galaxy_figs(sedlist, good, wcs, img)

    bayes_figs(DES, detmaps, detivs, good, wcs, img)
    
if __name__ == '__main__':
    main()
