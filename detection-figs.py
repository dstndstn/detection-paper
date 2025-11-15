import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import os
import sys
import time
import pylab as plt
import numpy as np
import scipy
from functools import reduce
import fitsio
from astrometry.util.fits import fits_table
from astrometry.util.file import pickle_to_file, unpickle_from_file
from astrometry.util.util import Tan
from astrometry.util.plotutils import PlotSequence
#from astrometry.util.starutil import *
#from astrometry.util.starutil_numpy import *
from astrometry.libkd.spherematch import match_xy, match_radec
from collections import Counter
#from scipy.ndimage.filters import *
from scipy.ndimage import label, find_objects
from scipy.ndimage import binary_dilation, binary_fill_holes
from scipy.special import erfc, logsumexp
from scipy.interpolate import CubicSpline
import bayes_figure

antigray = matplotlib.colormaps['gray_r']

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
    return blobs, np.array(px),np.array(py)



def sed_matched_figs(detect_sn, good, img, sedlist, DES, g_det, r_det, i_det,
                     wcs):
    _,x,y = detect_sources(detect_sn, 100.)
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
        show_sources(sources[I[J]].x, sources[I[J]].y, img)
        plt.savefig('best-%s.pdf' % s.name.lower())

    #####  Run detection at different thresholds ####
    plt.figure(figsize=(3.5,3.5))

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
        _,x,y = detect_sources(snmap, thresh)
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
        #plt.figure(figsize=(3.5,3.5))
        #bayes_figure.subplots()
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.12, top=0.98)
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
                     mfc='none', ms=10, mew=2)
            #mfc='none', ms=15, mew=2)
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
        
        #plt.figure(figsize=(5,4))
        #plt.figure(figsize=(3.5,3.5))
        #bayes_figure.subplots()
        #plt.subplots_adjust(left=0.15, right=0.97, bottom=0.12, top=0.98)
        plt.clf()
        plt.subplots_adjust(left=0.15, right=0.98, bottom=0.12, top=0.98)
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
            plt.plot(gr, ri, s.plotsym, color='k', mfc='none', ms=8, mew=3)
        plt.axis([-0.5, 2.5, -0.5, 2])
        plt.xlabel('g - r (mag)')
        plt.ylabel('r - i (mag)')
        plt.legend(lp, lt, loc='upper left')
        plt.xticks([0,1,2])
        plt.ylim(-0.4, 1.9)
        plt.savefig('best-color-%i.pdf' % thresh)

    ##############################
    
    
    
    # Artifacts from single-band detections
    I = np.hstack((np.flatnonzero(sources.imax == 3)[:6],
                   np.flatnonzero(sources.imax == 4)[:18],
                   np.flatnonzero(sources.imax == 5)[:12]))
    
    plt.figure(figsize=(4,4))
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    plt.clf()
    show_sources(sources[I].x, sources[I].y, img, R=6, C=6, sz=30, divider=1)
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
    
def show_sources(x, y, img, R=10, C=10, sz=10, divider=0,
                 row_dividers=None, row_divider_size=1):
    imgrows = []
    k = 0
    for i in range(R):
        imgrow = []
        for j in range(C):
            if k >= len(x):
                sub = np.zeros((sz*2+1,sz*2+1,3), np.uint8)
            else:
                #f = T[k]
                #sub = img[f.y-sz : f.y+sz+1, f.x-sz : f.x+sz+1, :]
                sub = img[y[k]-sz : y[k]+sz+1, x[k]-sz : x[k]+sz+1, :]
            #print('subimage', sub.shape)
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
        if row_dividers is not None and i+1 in row_dividers:
            rh,rw,three = imgrow.shape
            imgrows.append(np.zeros((row_divider_size, rw, 3), np.uint8) + 255)
            
    imgrows = np.vstack(list(reversed(imgrows)))
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
    from astrometry.util.plotutils import plothist

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
    #_,bx,by = detect_sources(lprb, 2000)
    for thresh in [#2000, 1500, 1000, 500]:
                   #500, 400, 300,
                   200]:

        bayes_thresh = thresh
        _,bx,by = detect_sources(lprb, thresh)
        print('Bayes detection at', thresh, ':', len(bx), 'sources')

    bsources = fits_table()
    bsources.x = bx
    bsources.y = by
    bsources.lprb = lprb[bsources.y, bsources.x]
    bsources.cut((bsources.x > sz) * (bsources.x < (W-sz)) * (bsources.y > sz) * (bsources.y < (H-sz)))
    bsources.cut(good[bsources.y, bsources.x])
    print('Kept', len(bsources), 'Bayesian sources')

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

    for thresh in [#50, 40, 30, 25, 20
                   #20, 18, 16,
                   15,]:

        gri_thresh = thresh
        _,xg,yg = detect_sources(detmaps[0] * np.sqrt(detivs[0]), thresh)
        _,xr,yr = detect_sources(detmaps[1] * np.sqrt(detivs[1]), thresh)
        _,xi,yi = detect_sources(detmaps[2] * np.sqrt(detivs[2]), thresh)
        print('gri at thresh', thresh, 'detected', len(xg),len(xr),len(xi), 'gri')

    # _,xg,yg = detect_sources(detmaps[0] * np.sqrt(detivs[0]), 50.)
    # _,xr,yr = detect_sources(detmaps[1] * np.sqrt(detivs[1]), 50.)
    # _,xi,yi = detect_sources(detmaps[2] * np.sqrt(detivs[2]), 50.)
    # print('Detected', len(xg),len(xr),len(xi), 'gri')
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
    hot = binary_fill_holes(np.logical_or(detmaps[0] * np.sqrt(detivs[0]) > gri_thresh,
                            np.logical_or(detmaps[1] * np.sqrt(detivs[1]) > gri_thresh,
                                          detmaps[2] * np.sqrt(detivs[2]) > gri_thresh)))
    Bhot = binary_fill_holes(lprb > bayes_thresh)
    UB = np.flatnonzero((hot[bsources.y, bsources.x] == False))
    US = np.flatnonzero((Bhot[sources.y, sources.x] == False))
    print('Unmatched:', len(UB), 'Bayesian and', len(US), 'g+r+i')

    MS = np.flatnonzero(Bhot[sources.y, sources.x])

    plt.figure(figsize=(3.5,3.5))
    plt.subplots_adjust(left=0.15, right=0.98, bottom=0.12, top=0.98)

    plt.clf()
    # for the legend only
    p1 = plt.plot(-10, -10, 'k.')
    plt.plot(sources.gr[MS], sources.ri[MS], 'k.', alpha=0.1,
             label='Both')
    p2 = plt.plot(sources.gr[US],  sources.ri[US], 'o', mec='r', mfc='none',
             label='g+r+i only')
    p3 = plt.plot(bsources.gr[UB], bsources.ri[UB], 'bx',
             label='Bayesian only')
    plt.xlabel('g - r (mag)')
    plt.ylabel('r - i (mag)')
    plt.legend((p1[0],p2[0],p3[0]),
               ('Detected by both',
                'Only detected by g+r+i', 'Only detected by Bayesian'),
               loc='upper left', framealpha=1.0)
    plt.axis([-0.5, 3.5, -0.4, 2.35])
    plt.savefig('bayes-vs-gri.pdf')

    
    R,C = 3,8
    plt.figure(figsize=(C*1, R*1))
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    plt.clf()
    show_sources(bsources[UB].x, bsources[UB].y, img, R=R, C=C, divider=1)
    plt.savefig('bayes-only.pdf')

    plt.clf()
    show_sources(sources[US].x, sources[US].y, img, R=R, C=C, divider=1)
    plt.savefig('gri-only.pdf')

    # #####
    # sources = sources_orig.copy()
    # 
    # # What if we demand 2-band detection?
    # #bsources = bsources_orig
    # # Detect at a higher threshold to get ~2400 sources
    # Bhot = binary_fill_holes(lprb > 4000.)
    # _,bx,by = detect_sources(lprb, 4000.)
    # bsources = fits_table()
    # bsources.x = bx
    # bsources.y = by
    # bsources.lprb = lprb[bsources.y, bsources.x]
    # bsources.cut((bsources.x > sz) * (bsources.x < (W-sz)) * (bsources.y > sz) * (bsources.y < (H-sz)))
    # bsources.cut(good[bsources.y, bsources.x])
    # print('Kept', len(bsources))
    # bsources.g_flux = g_det[bsources.y, bsources.x]
    # bsources.r_flux = r_det[bsources.y, bsources.x]
    # bsources.i_flux = i_det[bsources.y, bsources.x]
    # bsources.ra,bsources.dec = wcs.pixelxy2radec(bsources.x+1, bsources.y+1)
    # bsources.g_mag = -2.5*(np.log10(bsources.g_flux) - 9)
    # bsources.r_mag = -2.5*(np.log10(bsources.r_flux) - 9)
    # bsources.i_mag = -2.5*(np.log10(bsources.i_flux) - 9)
    # bsources.gr = bsources.g_mag - bsources.r_mag
    # bsources.ri = bsources.r_mag - bsources.i_mag
    # I = np.argsort(-bsources.lprb)
    # bsources.cut(I)
    # 
    # sthresh = 50.
    # hotg = binary_fill_holes(detmaps[0] * np.sqrt(detivs[0]) > sthresh)
    # hotr = binary_fill_holes(detmaps[1] * np.sqrt(detivs[1]) > sthresh)
    # hoti = binary_fill_holes(detmaps[2] * np.sqrt(detivs[2]) > sthresh)
    # hotsum = hotg*1 + hotr*1 + hoti*1
    # hot = (hotsum >= 2)
    # K = np.flatnonzero(hot[sources.y,sources.x])
    # # 2386 detected in 2 or more bands
    # print(len(K), 'detected in 2 or more bands')
    # sources.cut(K)
    # 
    # N = min(len(sources), len(bsources))
    # sources  =  sources[:N]
    # bsources = bsources[:N]
    # print('Cut both to', N)
    # 
    # UB = np.flatnonzero((hot[bsources.y, bsources.x] == False))
    # US = np.flatnonzero((Bhot[sources.y, sources.x] == False))
    # print(len(UB), 'unmatched Bayesian', len(US), '2 of (g+r+i)')
    # 
    # plt.figure(figsize=(6,4))
    # plt.subplots_adjust(left=0.1, right=0.98, bottom=0.12, top=0.98)
    # 
    # plt.clf()
    # plt.plot(sources.gr[US],  sources.ri[US], 'o', mec='r', mfc='none',
    #          label='g+r+i only')
    # plt.plot(bsources.gr[UB], bsources.ri[UB], 'kx',
    #          label='Bayesian only')
    # plt.xlabel('g - r (mag)')
    # plt.ylabel('r - i (mag)')
    # plt.legend()
    # plt.axis([-5, 5, -3, 3])
    # plt.savefig('bayes-vs-2gri.pdf')
    # 
    # plt.figure(figsize=(4,4))
    # plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    # plt.clf()
    # show_sources(bsources[UB].x, bsources[UB].y, img, R=10, C=10, divider=1)
    # plt.savefig('bayes-2only.pdf')
    # 
    # plt.clf()
    # show_sources(sources[US].x, sources[US].y, img, R=10, C=10, divider=1)
    # plt.savefig('2gri-only.pdf')
    

    
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
    _,x,y = detect_sources(yellow_gal, 100.)
    
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
    show_sources(gals[I].x, gals[I].y, img, sz=20)
    plt.savefig('galaxies.pdf')
    
    plt.figure(figsize=(5,4))
    plt.subplots_adjust(left=0.12, right=0.98, bottom=0.12, top=0.98)
    plt.clf()
    plt.semilogx(gals.yellow_sn, gals.yellow_sn / gals.yellow_sn_psf, 'k.', alpha=0.25)
    plt.ylim(0.9, 1.2)
    plt.xlabel('Galaxy detection S/N')
    plt.ylabel('Galaxy / PSF detection ratio')
    plt.savefig('galaxies-relsn.pdf')

def chisq_fig(good, img, g_det, g_detiv, r_det, r_detiv, wcs):
    g_det1 = fitsio.read('1d/detmap-g.fits')
    g_detiv1 = fitsio.read('1d/detiv-g.fits')
    r_det1 = fitsio.read('1d/detmap-r.fits')
    r_detiv1 = fitsio.read('1d/detiv-r.fits')

    g_sn1 = g_det1 * np.sqrt(g_detiv1)
    r_sn1 = r_det1 * np.sqrt(r_detiv1)
    goodpix1 = np.logical_and(g_detiv1 > 0.5 * np.median(g_detiv1),
                              r_detiv1 > 0.5 * np.median(r_detiv1))

    g_sn = g_det * np.sqrt(g_detiv)
    r_sn = r_det * np.sqrt(r_detiv)

    red_sed = [1., 2.5]

    # Detect on the single image
    #_,c3x,c3y = detect_sources(np.hypot(g_sn1, r_sn1), 3.)
    _,c3x,c3y = detect_sources(np.hypot(g_sn1, r_sn1), 4.5)
    keep = goodpix1[c3y, c3x]
    c3x = c3x[keep]
    c3y = c3y[keep]

    # Compute the S/N required for g-only or r-only to trigger the
    # "red" SED detector
    dm=[np.array([[1,0]]), np.array([[0,1]])]
    div=[np.ones(2), np.ones(2)]
    sn = sedsn(dm, div, red_sed)
    sng = sn[0,0]
    snr = sn[0,1]

    plt.figure(figsize=(6,4))
    plt.subplots_adjust(right=0.95, top=0.98)

    from matplotlib.patches import Circle
    plt.clf()
    # Annotate points as "true" or "false" based on deeper data.
    real = (np.hypot(g_sn[c3y,c3x], r_sn[c3y,c3x]) >  10.)
    fake = np.logical_not(real)
    #plt.plot(g_sn1[c3y,c3x][fake], r_sn1[c3y,c3x][fake], '.', color='0.5', alpha=0.2, label='False Peaks')
    plt.plot(g_sn1[c3y,c3x][fake], r_sn1[c3y,c3x][fake], 'x', color='0.5', alpha=0.5, label='False Detections')
    plt.plot(g_sn1[c3y,c3x][real], r_sn1[c3y,c3x][real], '.', color='k', alpha=0.5, label='Real Detections')
    a = np.linspace(0, 2.*np.pi, 100)
    plt.plot(5.*np.sin(a), 5.*np.cos(a), 'b-', label='Chi-squared detection')
    # r
    plt.axhline(5., color='r', linestyle=':', label='r-band only detection')
    # red
    m=-sng/snr
    b=5./snr
    xx = np.array([-20,40])
    plt.plot(xx, b+m*xx, 'm-', mew=2, linestyle='--', label="``Red'' SED-matched detection")
    plt.legend(loc='upper right', framealpha=1.)
    plt.axis('square')
    plt.axis([-10,40,-10,20])
    plt.xlabel('g band S/N')
    plt.ylabel('r band S/N')
    plt.axhline(0, color='k', alpha=0.25)
    plt.axvline(0, color='k', alpha=0.25)
    plt.savefig('sed-matched.pdf')
    
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
    DES.x = (x-1).astype(np.int32)
    DES.y = (y-1).astype(np.int32)

    # Cut of DES catalog for the SED-matched filter figure
    sz = 20
    Ides = np.flatnonzero((DES.x > sz) * (DES.y > sz) *
                          (DES.x < (W-sz)) * (DES.y < (H-sz)) *
                          good[np.clip(DES.y, 0, H-1), np.clip(DES.x, 0, W-1)])

    #chisq_fig(good, img, g_det, g_detiv, r_det, r_detiv, wcs)
    
    sed_matched_figs(detect_sn, good, img, sedlist, DES[Ides],
                     g_det, r_det, i_det, wcs)

    # called from new_main instead!
    #bayes_figs(DES, detmaps, detivs, good, wcs, img)

    #galaxy_figs(sedlist, good, wcs, img)

def find_peaks(img, thresh, goodmap=None):
    # peaks
    # (we leave a margin of 1 pixel around the edges)

    # iy,ix = np.nonzero((img[1:-1, 1:-1] >= thresh) *
    #                    (img[1:-1, 1:-1] > img[ :-2,  :-2]) *
    #                    (img[1:-1, 1:-1] > img[ :-2, 1:-1]) *
    #                    (img[1:-1, 1:-1] > img[ :-2, 2:  ]) *
    #                    (img[1:-1, 1:-1] > img[1:-1,  :-2]) *
    #                    (img[1:-1, 1:-1] > img[1:-1, 2:  ]) *
    #                    (img[1:-1, 1:-1] > img[2:  ,  :-2]) *
    #                    (img[1:-1, 1:-1] > img[2:  , 1:-1]) *
    #                    (img[1:-1, 1:-1] > img[2:  , 2:  ]))
    #ix += 1
    #iy += 1

    # Demand a pixel above threshold and the largest in a 5x5 block
    iy,ix = np.nonzero((img >= thresh) *
                       (img >= scipy.ndimage.maximum_filter(img, 5)))
    #H,W = img.shape
    #K = np.flatnonzero((ix >= 2) * (iy >= 2) * (ix < W-2) * (iy < H-2))
    #iy = iy[K]
    #ix = ix[K]

    if goodmap is not None:
        Igood = np.flatnonzero(goodmap[iy,ix])
        ix,iy = ix[Igood], iy[Igood]
    return ix, iy

def chisq_detection_raw(detmaps, detivs, goodmap, thresh=25.):#, peaks=False):
    chisq = 0.
    for detmap, detiv in zip(detmaps, detivs):
        sn = detmap * np.sqrt(detiv)
        chisq = chisq + sn**2
    #if not peaks:
    #    blobs,x,y = detect_sources(chisq, thresh)
    #else:
    x,y = find_peaks(chisq, thresh, goodmap)
    # return in sorted order (brightest first)
    sval = chisq[y,x]
    I = np.argsort(-sval)
    x,y = x[I],y[I]
    return x,y, chisq

def chisq_detection_pos(detmaps, detivs, goodmap, thresh=25.):
    chisq = 0.
    for detmap, detiv in zip(detmaps, detivs):
        sn = detmap * np.sqrt(detiv)
        chisq = chisq + np.maximum(sn, 0)**2
    x,y = find_peaks(chisq, thresh, goodmap)
    # return in sorted order (brightest first)
    sval = chisq[y,x]
    I = np.argsort(-sval)
    x,y = x[I],y[I]
    return x,y, chisq

def sed_union_detection(seds, goodmap, thresh=5.):
    maxsn = 0.
    for sed in seds:
        sn = sed.snmap
        maxsn = np.maximum(maxsn, sn)
    x,y = find_peaks(maxsn, thresh, goodmap)
    # return in sorted order (brightest first)
    sval = maxsn[y,x]
    I = np.argsort(-sval)
    x,y = x[I],y[I]
    return x,y, maxsn

def sed_mixture_detection(seds, weights, detmaps, detivs,
                          goodmap, thresh, alpha=1.):
    nseds = len(seds)
    nbands = len(detmaps)
    H,W = detmaps[0].shape
    
    D   = np.empty((nbands, H, W), np.float32)
    Div = np.empty((nbands, H, W), np.float32)
    for i in range(nbands):
        D[i,:,:] = detmaps[i]
        Div[i,:,:] = detivs[i]
    SED = np.zeros((nseds, nbands), np.float32)
    for j in range(nseds):
        SED[j,:] = seds[j].sed

    log_pratio = log_pratio_bayes(SED, weights, D, Div, alpha)
    x,y = find_peaks(log_pratio, thresh, goodmap)
    # return in sorted order (highest-logprob first)
    sval = log_pratio[y,x]
    I = np.argsort(-sval)
    x,y = x[I],y[I]
    return x,y, log_pratio

# from bayes_figure import get_pratio
# def sed_mixture_detection(fluxes, sig_fluxes, seds, alpha=1.):
#     # fluxes: npix x nbands array
#     # sig_fluxes: nbands
#     # seds: [(name, scalar sed, weight -- ignored)]
#     x = 0.
#     for sed in seds:
#         name,sed_vec,wt = sed
#         pr = get_pratio(fluxes.T[:, :, np.newaxis], sig_fluxes, sed_vec,
#                         alpha=alpha)
#         pr = pr[:,0]
#         x = x + wt * pr
#     return x





# def sed_union_threshold(seds, falsepos_rate):
#     x = x.reshape(pbg.shape)
#     # Find threshold
#     X = scipy.optimize.root_scalar(lambda th: np.sum(pbg * (x > th)) - falsepos_rate,
#                                    method='bisect', bracket=(0, 1e10))
#     print('SED(union) Thresh:', X)
#     assert(X.converged)
#     return X.root



# def image_grid(x, y, rgb, s=8, Nmax=100, first_and_last=True):
# 
#     H,W,three = rgb.shape
#     I = np.flatnonzero((x >= s) * (y >= s) * (x < W-s) * (y < H-s))
#     x,y = x[I],y[I]
# 
#     n = min(Nmax, len(x))
#     C = int(np.ceil(1.0 * np.sqrt(n)))
#     R = int(np.ceil(n / C))
#     if R*C < len(x):
#         print('Only showing', R*C, 'of', len(x), 'image cutouts')
# 
#         if first_and_last:
#             R2 = R - R//2
#             I = np.hstack((np.arange(C * R//2),
#                            np.arange(len(x) - C*R2, len(x))))
#             x = x[I]
#             y = y[I]
# 
#     plt.clf()
#     plt.subplots_adjust(hspace=0, wspace=0)
#     for i in range(len(x)):
#         if i >= R*C:
#             break
#         plt.subplot(R,C,1+i)
#         plt.imshow(rgb[y[i]-s:y[i]+s+1, x[i]-s:x[i]+s+1,:],
#                    interpolation='nearest', origin='lower')
#         plt.xticks([]); plt.yticks([])

def image_grid(x, y, rgb):  #, s=8, Nmax=100, first_and_last=True):
    #R = 8
    #C = 8
    R = 6
    #C = 7
    C = 6
    sz = 8

    plt.figure(figsize=(3,3))
    #plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.9)
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88)

    H,W,three = rgb.shape
    I = np.flatnonzero((x >= sz) * (y >= sz) * (x < W-sz) * (y < H-sz))
    x,y = x[I],y[I]

    kwa = {}
    # First and last:
    if len(x) > (R*C):
        R2 = R - R//2
        I = np.hstack((np.arange(C * R//2),
                       np.arange(len(x) - C*R2, len(x))))
        x = x[I]
        y = y[I]
        kwa.update(row_dividers=[R//2])
    plt.clf()
    show_sources(x, y, rgb, sz=sz, R=R, C=C, **kwa)

def new_main():

    np.random.seed(123456789)
    
    basedir = 'detection-paper-data/'
    brickname = 'custom-036450m04600'
    brick = brickname
    bri = brickname[:3]

    bands = ['g', 'r', 'i']

    # Slice the brick images
    y0 = 200
    slc = slice(y0,None), slice(None)

    wcs = Tan(os.path.join(basedir, 'coadd', bri, brick,
                           'legacysurvey-%s-image-%s.fits.fz' % (brick, 'r')), 1)
    h,w = wcs.shape
    wcs = wcs.get_subimage(0, y0, w, h-y0)
    print('sub-WCS:', wcs)

    detmaps = [fitsio.read(os.path.join(basedir, 'detmap-one-%s.fits' % band))[slc]
               for band in bands]
    detivs  = [fitsio.read(os.path.join(basedir, 'detiv-one-%s.fits' % band))[slc]
               for band in bands]

    nco = [fitsio.read(os.path.join(basedir, 'coadd', bri, brick,
                                    'legacysurvey-%s-nexp-%s.fits.fz' % (brick, band)))[slc]
           for band in bands]

    deep_detmaps = [fitsio.read(os.path.join(basedir, 'detmap-%s.fits' % band))[slc]
               for band in bands]

    deep_co = [fitsio.read(os.path.join(basedir, 'coadd', bri, brick,
                        'legacysurvey-%s-image-%s.fits.fz' % (brick, band)))[slc]
               for band in bands]
    # stretch
    s = 16
    scale = dict(g=(2, 6.0*s), r=(1, 3.4*s), i=(0, 3.0*s))
    rgb = sdss_rgb(deep_co, bands, scales=scale)

    # single-image "coadd"
    one_co = [fitsio.read(os.path.join(basedir, 'image-one-%s.fits' % band))[slc]
              for band in bands]
    # stretch
    s = 1
    scale = dict(g=(2, 6.0*s), r=(1, 3.4*s), i=(0, 3.0*s))
    rgb_one = sdss_rgb(one_co, bands, scales=scale)

    # plt.clf()
    # plt.imshow(rgb, interpolation='nearest', origin='lower')
    # plt.savefig('rgb-co.png')
    # 
    # plt.clf()
    # plt.imshow(rgb_one, interpolation='nearest', origin='lower')
    # plt.savefig('rgb-one.png')

    print('90 percentile number of exposures:', [np.percentile(n, 90) for n in nco])

    goodmap = reduce(np.logical_and, [n>=10 for n in nco])
    print('Number of good pixels with >= 10 exposures:', np.sum(goodmap))
    goodmap = reduce(np.logical_and, [n>=90 for n in nco])
    print('Number of good pixels with >= 90 exposures:', np.sum(goodmap))

    for iv in detivs:
        goodmap *= (iv > 0.9 * iv.max())
    print('Number of good pixels with good detiv:', np.sum(goodmap))

    fake_detmaps = []
    for iv in detivs:
        with np.errstate(divide='ignore'):
            r = np.random.normal(size=iv.shape) / np.sqrt(iv)
        r[iv <= 0] = 0.
        fake_detmaps.append(r)

    # Confirm that they're ~Gaussian distributed
    # plt.clf()
    # for detmap,detiv,band in zip(detmaps, detivs, bands):
    #     plt.hist((detmap * np.sqrt(detiv))[detiv > 0], range=(-5, +5), bins=40, histtype='step', label=band)
    # for detmap,detiv,band in zip(fake_detmaps, detivs, bands):
    #     plt.hist((detmap * np.sqrt(detiv))[detiv > 0], range=(-5, +5), bins=40, histtype='step', label='Fake ' + band)
    # plt.xlabel('Detection map S/N')
    # plt.ylabel('Number of pixels')
    # plt.legend()
    # plt.savefig('detsn.png')

    # Peak-based detection gives better looking results
    # blobs, x, y, chisq = chisq_detection_raw(detmaps, detivs)
    # Igood = np.flatnonzero(goodmap[y,x])
    # print('Chi-squared method: found', len(x), 'sources in', len(np.unique(blobs))-1, 'blobs,', len(Igood), 'with good N exposures')
    # x,y = x[Igood], y[Igood]
    # sval = chisq[y,x]
    # I = np.argsort(-sval)
    # x,y = x[I],y[I]
    # plt.clf()
    # plt.imshow(chisq, interpolation='nearest', origin='lower', cmap='gray',
    #            vmin=0, vmax=100)
    # plt.colorbar()
    # plt.plot(x, y, 'o', mec='r', mfc='none', ms=10)
    # plt.savefig('chisq-det.png')

    from chi_squared_experiment import chisq_pos_isf
    import scipy

    g = scipy.stats.norm()
    g_sigma = 5.
    falsepos_rate = g.sf(g_sigma)
    print('Gaussian %.3f sigma survival function:' % g_sigma, falsepos_rate)
    n_bands = len(bands)
    ch = scipy.stats.chi2(n_bands)
    chi2_thresh = ch.isf(falsepos_rate)
    print('Chi2 thresh:', chi2_thresh)
    ch1 = scipy.stats.chi2(1)
    chi2_pos_thresh = chisq_pos_isf(n_bands, falsepos_rate)
    print('Chi2-pos thresh:', chi2_pos_thresh)

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
    #for s in sedlist:
    #    s.snmap = sedsn(detmaps, detivs, s.sed)


    if True:
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
        DES.x = (x-1).astype(int)
        DES.y = (y-1).astype(int)
        H,W = wcs.shape
        # Cut of DES catalog for the SED-matched filter figure
        sz = 20
        Ides = np.flatnonzero((DES.x > sz) * (DES.y > sz) *
                              (DES.x < (W-sz)) * (DES.y < (H-sz)) *
                              goodmap[np.clip(DES.y, 0, H-1), np.clip(DES.x, 0, W-1)])
        bayes_figs(DES, detmaps, detivs, goodmap, wcs, rgb)

    # Blue, Yellow, Red, i-only
    union_seds = sedlist[:3] + [sedlist[-1]]
    union_thresh = 5.296 #5.22

    # Bayesian Mixture SED method
    mix_seds = union_seds
    # mixture weights: blue, yellow, red, i-only
    mix_weights = np.array([0.33, 0.33, 0.33, 0.01])
    mix_thresh = 9.465 #9.30


    if True:
        # If the SEDs were independent, the false-pos rate would be len(union_seds)
        # times larger, so
        target_falsepos_rate = falsepos_rate / len(union_seds)
        th = g.isf(target_falsepos_rate)
        print('SED union threshold (safe):', th)

    else:
        n_fakes = 10
        big_fake_detmaps = []
        for i in range(n_fakes):
            fakes = []
            for iv in detivs:
                with np.errstate(divide='ignore'):
                    r = np.random.normal(size=iv.shape) / np.sqrt(iv)
                r[iv <= 0] = 0.
                fakes.append(r)
            big_fake_detmaps.append(fakes)
        del fakes
        del r
    
        # def union_false_pos_rate(thresh, big_fake_detmaps, detivs, goodmap, union_seds):
        #     n_det = 0
        #     n_pix = len(big_fake_detmaps) * np.sum(goodmap)
        #     for fdm in big_fake_detmaps:
        #         for s in union_seds:
        #             s.snmap = sedsn(fdm, detivs, s.sed)
        #         x,y,maxsn = sed_union_detection(union_seds, goodmap, thresh)
        #         n_det += len(x)
        #         del maxsn
        #     print('threshold', thresh, 'false pos:', n_det, '-> rate', n_det/n_pix)
        #     return n_det / n_pix
        # # Find threshold
        # X = scipy.optimize.root_scalar(
        #     lambda th: union_false_pos_rate(th, big_fake_detmaps, detivs, goodmap,
        #                                     union_seds) - falsepos_rate,
        #     method='bisect', bracket=(g_sigma, g_sigma*2.))
        # print('SED(union) Thresh:', X)
        # assert(X.converged)
        # union_thresh = X.root
    
        target_det = falsepos_rate * n_fakes * np.sum(goodmap)
        #target_det = int(np.floor(target_det))
        print('Aiming for %.1f' % target_det, 'false positive detections -> rate', target_det / (n_fakes * np.sum(goodmap)))

        thresh = g_sigma
        all_snvals = []
        for fdm in big_fake_detmaps:
            for s in union_seds:
                s.snmap = sedsn(fdm, detivs, s.sed)
            x,y,maxsn = sed_union_detection(union_seds, goodmap, thresh)
            snvals = maxsn[y, x]
            all_snvals.append(snvals)
        all_snvals = np.hstack(all_snvals)
        all_snvals.sort()
        # Reverse to put largest values first
        all_snvals = all_snvals[::-1]
        # interpolate...
        spl = CubicSpline(1 + np.arange(len(all_snvals)), all_snvals)
        union_thresh = spl(target_det)
        print('Union threshold:', union_thresh)

        idet = int(target_det)
        print('Values near threshold:', all_snvals[idet-2:idet+3])
        
        # print('Union thresholds:', all_snvals[-(target_det+2):-(target_det-1)])
        # # Could interpolate...
        # union_thresh = all_snvals[-target_det]
        # print('Union Threshold:', union_thresh)

        # Bayesian mixture
        all_snvals = []
        for fdm in big_fake_detmaps:
            x, y, bp = sed_mixture_detection(mix_seds, mix_weights, fdm, detivs,
                                             goodmap, mix_thresh)
            snvals = bp[y, x]
            all_snvals.append(snvals)
        all_snvals = np.hstack(all_snvals)
        all_snvals.sort()

        # Reverse to put largest values first
        all_snvals = all_snvals[::-1]
        # interpolate...
        spl = CubicSpline(1 + np.arange(len(all_snvals)), all_snvals)
        mix_thresh = spl(target_det)
        # print('Mixture thresholds:', all_snvals[-(target_det+2):-(target_det-1)])
        # # Could interpolate...
        # mix_thresh = all_snvals[-target_det]
        print('Mixture threshold:', mix_thresh)
        print('Values near threshold:', all_snvals[idet-2:idet+3])

        ndet_union = []
        ndet_mix = []
        ndet_chisq = []
        ndet_chipos = []
        
        for fdm in big_fake_detmaps:
            for s in union_seds:
                s.snmap = sedsn(fdm, detivs, s.sed)
            x, y, usn = sed_union_detection(union_seds, goodmap, union_thresh)
            ndet_union.append(len(x))
            
            x, y, bp = sed_mixture_detection(mix_seds, mix_weights, fdm, detivs,
                                        goodmap, mix_thresh)
            ndet_mix.append(len(x))

            x, y, chisq = chisq_detection_raw(fdm, detivs, goodmap,
                                      thresh=chi2_thresh)
            ndet_chisq.append(len(x))

            x, y, chisq = chisq_detection_pos(fdm, detivs, goodmap,
                                        thresh=chi2_pos_thresh)
            ndet_chipos.append(len(x))

        print('SED union: total of', np.sum(ndet_union), 'false detections')
        print(ndet_union)
        print('SED mixture: total of', np.sum(ndet_mix), 'false detections')
        print(ndet_mix)
        print('Chi-sq: total of', np.sum(ndet_chisq), 'false detections')
        print(ndet_chisq)
        print('Chi-pos: total of', np.sum(ndet_chipos), 'false detections')
        print(ndet_chipos)

        del big_fake_detmaps

    if False:
        # Compute SED S/Ns for fake data!
        for s in union_seds:
            s.snmap = sedsn(fake_detmaps, detivs, s.sed)
        x, y, usn = sed_union_detection(union_seds, goodmap, union_thresh)
        print('Union SED method: found', len(x), 'sources in fake data')
    
        # Bayesian SED method
        x, y, bp = sed_mixture_detection(mix_seds, mix_weights, fake_detmaps, detivs,
                                         goodmap, mix_thresh)
        print('Mixture SED method: found', len(x), 'sources in fake data')
    
        x, y, chisq = chisq_detection_raw(fake_detmaps, detivs, goodmap,
                                          thresh=chi2_thresh)
        print('Chi-squared method: found', len(x), 'sources in fake data')
    
        x, y, chisq = chisq_detection_pos(fake_detmaps, detivs, goodmap,
                                          thresh=chi2_pos_thresh)
        print('Chi-squared-pos method: found', len(x), 'sources in fake data')

    # Compute SED S/Ns for real data!
    for s in union_seds:
        s.snmap = sedsn(detmaps, detivs, s.sed)

    def plot_cc():
        #plt.figure(figsize=(3.5,3.5))
        plt.figure(figsize=(3,3))
        plt.subplots_adjust(left=0.2, right=0.99, bottom=0.2, top=0.87)

    def plot_oneimage():
        plt.figure(figsize=(2.5,2.5))
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.85)

    H,W = detmaps[0].shape
    # Cut to sources in 500x500 subimage
    subH,subW = 500,500
    subslice = slice(0, subH), slice(0, subW)
    s = 15
    ax = [s, subW-s, s, subH-s]

    for band,iv in zip(bands, detivs):
        plt.clf()
        #plt.imshow(iv[subslice], interpolation='nearest', origin='lower')
        plt.hist(iv[subslice].ravel())
        plt.title('detection invvar: %s' % band)
        plt.savefig('detiv-%s.png' % band)
    
    x, y, usn = sed_union_detection(union_seds, goodmap, union_thresh)
    print('Union SED method: found', len(x), 'sources')
    union_x,union_y = x.copy(),y.copy()
    union_map = usn.copy()

    I = np.flatnonzero((x >= s) * (y >= s) * (x < subW-s) * (y < subH-s))
    x,y = x[I],y[I]
    print('Showing', len(x), 'in subimage')
    # plt.clf()
    # plt.imshow(union_map[subslice] * goodmap[subslice],
    #            interpolation='nearest', origin='lower', cmap='gray',
    #            vmin=0, vmax=union_thresh*2)
    # plt.colorbar()
    # plt.plot(x, y, 'o', mec='r', mfc='none', ms=10)
    # plt.axis(ax)
    # plt.title('SED Union peaks')
    # plt.savefig('sed-union-peaks.png')

    image_grid(x, y, rgb)
    plt.title('SED (union) detections')
    plt.savefig('sed-union-images.png')
    plt.savefig('sed-union-images.pdf')

    image_grid(x, y, rgb_one)
    plt.title('SED (union) detections (one image)')
    plt.savefig('sed-union-images-one.png')

    x, y, bp = sed_mixture_detection(mix_seds, mix_weights, detmaps, detivs,
                                     goodmap, mix_thresh)
    print('Mixture SED method: found', len(x), 'sources')
    mix_x,mix_y = x.copy(),y.copy()
    mix_map = bp.copy()

    I = np.flatnonzero((x >= s) * (y >= s) * (x < subW-s) * (y < subH-s))
    x,y = x[I],y[I]
    print('Showing', len(x), 'in subimage')
    plt.clf()
    plt.imshow(mix_map[subslice] * goodmap[subslice],
               interpolation='nearest', origin='lower', cmap='gray',
               vmin=0, vmax=mix_thresh*2)
    plt.colorbar()
    plt.plot(x, y, 'o', mec='r', mfc='none', ms=10)
    plt.axis(ax)
    plt.title('SED Mixture peaks')
    plt.savefig('sed-mix-peaks.png')

    image_grid(x, y, rgb)
    plt.title('SED (Bayes) detections')
    plt.savefig('sed-mix-images.png')
    plt.savefig('sed-mix-images.pdf')

    image_grid(x, y, rgb_one)
    plt.suptitle('SED (Bayesd) detections (one image)')
    plt.savefig('sed-mix-images-one.png')

    x, y, chisq = chisq_detection_raw(detmaps, detivs, goodmap,
                                      thresh=chi2_thresh)
    print('Chi-squared method: found', len(x), 'sources')
    chi_x,chi_y = x.copy(),y.copy()
    chi_map = chisq.copy()

    I = np.flatnonzero((x >= s) * (y >= s) * (x < subW-s) * (y < subH-s))
    x,y = x[I],y[I]
    print('Showing', len(x), 'in subimage')
    plt.clf()
    plt.imshow(chisq[subslice] * goodmap[subslice],
               interpolation='nearest', origin='lower', cmap='gray',
               vmin=0, vmax=chi2_thresh*2)
    plt.colorbar()
    plt.plot(x, y, 'o', mec='r', mfc='none', ms=10)
    plt.axis(ax)
    plt.title('Chisq peaks')
    plt.savefig('chisq-peaks.png')

    image_grid(x, y, rgb)
    plt.title(r'$\chi^2$ detections')
    plt.savefig('chisq-images.png')
    plt.savefig('chisq-images.pdf')

    image_grid(x, y, rgb_one)
    plt.suptitle('Chi-squared detections (the image)')
    plt.savefig('chisq-images-one.png')

    x, y, chisq = chisq_detection_pos(detmaps, detivs, goodmap,
                                      thresh=chi2_pos_thresh)
    print('Chi-squared-pos method: found', len(x), 'sources')
    chi_pos_x,chi_pos_y = x.copy(),y.copy()
    chi_pos_map = chisq.copy()

    I = np.flatnonzero((x >= s) * (y >= s) * (x < subW-s) * (y < subH-s))
    x,y = x[I],y[I]
    print('Showing', len(x), 'in subimage')

    plt.clf()
    plt.imshow(chisq[subslice] * goodmap[subslice],
               interpolation='nearest', origin='lower', cmap='gray',
               vmin=0, vmax=chi2_pos_thresh*2)
    plt.colorbar()
    plt.plot(x, y, 'o', mec='r', mfc='none', ms=10)
    plt.axis(ax)
    plt.title('Chisq-pos peaks')
    plt.savefig('chisq-pos-peaks.png')

    image_grid(x, y, rgb)
    plt.title(r'$\chi_+^2$ detections')
    plt.savefig('chisq-pos-images.png')
    plt.savefig('chisq-pos-images.pdf')

    image_grid(x, y, rgb_one)
    plt.suptitle('Chi-squared-pos detections (the image)')
    plt.savefig('chisq-pos-images-one.png')

    from astrometry.libkd.spherematch import match_xy

    I,J,d = match_xy(chi_x, chi_y, chi_pos_x, chi_pos_y, 10)

    plt.clf()
    plt.hist(d)
    plt.xlabel('Match distance (pixels)')
    plt.ylabel('Number of matches')
    plt.savefig('chi-match.png')

    chi_meth = (r'$\chi^2$', 'chisq', chi_x, chi_y, chi_map, chi2_thresh)
    chipos_meth = (r'$\chi_+^2$', 'chipos', chi_pos_x, chi_pos_y, chi_pos_map, chi2_pos_thresh)
    union_meth = ('SED (union)', 'sed-union', union_x, union_y, union_map, union_thresh)
    mix_meth = ('SED (Bayes)', 'sed-mix', mix_x, mix_y, mix_map, mix_thresh)

    colorcolor_plot = set()
    medimg_plot = set()

    for ((a_name,a_tag,a_x,a_y,a_map,a_thresh),
         (b_name,b_tag,b_x,b_y,b_map,b_thresh)) in [
        (chi_meth, chipos_meth),
        (chipos_meth, union_meth),
        (chipos_meth, mix_meth),
        (union_meth, mix_meth),
        # all combos...
        (chi_meth, union_meth),
        (chi_meth, mix_meth),
        ]:

        I,J,d = match_xy(a_x, a_y, b_x, b_y, 2)

        print(len(I), a_name, 'and', len(J), b_name, 'sources matched within 2 pix')
        print(len(np.unique(I)), a_name, 'unique and',
              len(np.unique(J)), b_name, 'unique')

        U = np.ones(len(a_x), bool)
        U[I] = False
        K = np.flatnonzero(U)
        print(len(K), 'unmatched', a_name, 'detections')

        V = np.ones(len(b_x), bool)
        V[J] = False
        L = np.flatnonzero(V)
        print(len(L), 'unmatched', b_name, 'detections')

        for (one_U, one_x, one_y, one_name, one_tag, one_map, two_name, two_tag, two_map, two_thresh) in [
            (K, a_x, a_y, a_name, a_tag, a_map, b_name, b_tag, b_map, b_thresh),
            (L, b_x, b_y, b_name, b_tag, b_map, a_name, a_tag, a_map, a_thresh),
            ]:

            Nmax = 200
            x,y = one_x[one_U], one_y[one_U]
            I = np.argsort(-one_map[y, x])
            x,y = x[I],y[I]
            # image_grid(x, y, rgb)
            # plt.suptitle('%s detections not in %s' % (one_name, two_name))
            # plt.savefig('unmatched-%s-%s.png' % (one_name, two_name))

            # Also check that A's unmatched x,y are below-threshold in B
            I = np.flatnonzero(two_map[y, x] < two_thresh)
            print(len(I), 'unmatched', one_name, 'detections are below-threshold in', two_name)
            x,y = x[I],y[I]
            image_grid(x, y, rgb)
            #plt.title('Detected in %s, not in %s' % (one_name, two_name))
            plt.title('Detected in %s,\nnot in %s' % (one_name, two_name))
            plt.savefig('unmatched-%s-%s-2.png' % (one_tag, two_tag))
            plt.savefig('unmatched-%s-%s-2.pdf' % (one_tag, two_tag))

            # plots we're not using in the paper
            if False:
                s = 10
                I = np.flatnonzero((y >= s) * (y < H-s))
                plt.clf()
                # Randomly select at most 20?
                nmax = 20
                if len(I) > nmax:
                    I = I[np.random.permutation(len(I))[:nmax]]
                for xi,yi in zip(x[I], y[I]):
                    #for band,co in zip(bands, deep_co):
                    for band,det in zip(bands, detmaps):
                    #for band,det,detiv in zip(bands, detmaps, detivs):
                        #plt.plot(co[xi, yi-s:yi+s+1], '-', color={'i':'m'}.get(band,band))
                        plt.plot(det[yi-s:yi+s+1, xi], '-', color={'i':'m'}.get(band,band), alpha=0.25)
                        #print('detiv', band, ':', detiv[yi, xi])
                plt.ylabel('Detection map value (flux)')
                plt.axhline(0, color='k', alpha=0.3)
                plt.ylim(-0.2, +0.4)
                plt.suptitle('Detected in %s, not in %s' % (one_name, two_name))
                plt.savefig('unmatched-%s-%s-3.png' % (one_tag, two_tag))
    
                s = 10
                I = np.flatnonzero((y >= s) * (y < H-s))
                plt.clf()
                # Randomly select at most 20?
                nmax = 20
                if len(I) > nmax:
                    I = I[np.random.permutation(len(I))[:nmax]]
                for xi,yi in zip(x[I], y[I]):
                    for band,det,detiv in zip(bands, detmaps, detivs):
                        plt.plot(det[yi-s:yi+s+1, xi] * np.sqrt(detiv[yi-s:yi+s+1, xi]),
                                 '-', color={'i':'m'}.get(band,band), alpha=0.25)
                plt.ylim(-5, +8)
                plt.axhline(0, color='k', alpha=0.3)
                plt.ylabel('Detection map S/N')
                plt.suptitle('Detected in %s, not in %s' % (one_name, two_name))
                plt.savefig('unmatched-%s-%s-4.png' % (one_tag, two_tag))
    
                plt.clf()
                for band,det in zip(bands, detmaps):
                    vals = []
                    for xi,yi in zip(x[I], y[I]):
                        vals.append(det[yi-s:yi+s+1, xi])
                    vals = np.stack(vals, axis=-1)
                    med = np.median(vals, axis=-1)
                    plt.plot(med, '-', color={'i':'m'}.get(band,band))
                plt.ylim(-0.075, +0.15)
                plt.axhline(0, color='k', alpha=0.3)
                plt.ylabel('Detection map value')
                plt.suptitle('Detected in %s, not in %s: median' % (one_name, two_name))
                plt.savefig('unmatched-%s-%s-5.png' % (one_tag, two_tag))
    
                plt.clf()
                for band,det,detiv in zip(bands, detmaps, detivs):
                    vals = []
                    for xi,yi in zip(x[I], y[I]):
                        vals.append(det[yi-s:yi+s+1, xi] * np.sqrt(detiv[yi-s:yi+s+1, xi]))
                    vals = np.stack(vals, axis=-1)
                    med = np.median(vals, axis=-1)
                    plt.plot(med, '-', color={'i':'m'}.get(band,band))
                plt.ylim(-5, +6)
                plt.axhline(0, color='k', alpha=0.3)
                plt.ylabel('Detection map S/N')
                plt.suptitle('Detected in %s, not in %s: median' % (one_name, two_name))
                plt.savefig('unmatched-%s-%s-6.png' % (one_tag, two_tag))

            plot_cc()
            plt.clf()
            detfluxes  = [detmap[y, x] for detmap in detmaps]
            deepfluxes = [detmap[y, x] for detmap in deep_detmaps]
            #for i,fluxes in enumerate([detfluxes, deepfluxes]):
            #    plt.subplot(1,2,i+1)

            xmin,xmax = -1, 3
            ymin,ymax = -2, 3
            
            for i,fluxes in enumerate([detfluxes, deepfluxes]):
                mags = [-2.5 * (np.log10(f) - 9) for f in fluxes]
                g,r,i = mags
                #plt.plot(np.clip(g - r, xmin, xmax), np.clip(r - i, ymin, ymax),
                #         '.', alpha=0.25)
                px = g-r
                py = r-i
                blu = '#1f77b4'
                kwa = dict(color=blu, alpha=0.25, ms=4)
                # good
                I = np.flatnonzero((px > xmin) * (px < xmax) * (py > ymin) * (py < ymax))
                plt.plot(px[I], py[I], 'o', **kwa)
                # off the left edge
                I = np.flatnonzero(px <= xmin)
                plt.plot([xmin]*len(I), np.clip(py[I], ymin, ymax), '<', **kwa)
                # off the right edge
                I = np.flatnonzero(px >= xmax)
                plt.plot([xmax]*len(I), np.clip(py[I], ymin, ymax), '>', **kwa)
                # off the top
                I = np.flatnonzero((px > xmin) * (px < xmax) * (py >= ymax))
                plt.plot(px[I], [ymax]*len(I), '^', **kwa)
                # off the bottom
                I = np.flatnonzero((px > xmin) * (px < xmax) * (py <= ymin))
                plt.plot(px[I], [ymin]*len(I), 'v', **kwa)

                plt.xlabel('g - r (mag)')
                plt.ylabel('r - i (mag)')
                break
            #ax = [-2, +6, -5, +3]
            #plt.subplot(1,2,1)
            #plt.title('One exposure')
            #plt.axis(ax)
            plt.axhline(0, color='k', alpha=0.2)
            plt.axvline(0, color='k', alpha=0.2)
            #plt.subplot(1,2,2)
            #plt.title('Deep coadd')
            plt.axis([xmin-0.1, xmax+0.1, ymin-0.1, ymax+0.1])
            plt.title('Detected in %s,\nnot in %s' % (one_name, two_name))
            plt.savefig('unmatched-%s-%s-7.png' % (one_tag, two_tag))
            plt.savefig('unmatched-%s-%s-7.pdf' % (one_tag, two_tag))

            if not one_name in colorcolor_plot:
                colorcolor_plot.add(one_name)
                plot_cc()
                # Color-color plot for just method A
                plt.clf()
                detfluxes  = [detmap[one_y, one_x] for detmap in detmaps]
                mags = [-2.5 * (np.log10(f) - 9) for f in detfluxes]
                g,r,i = mags
                xmin,xmax = -1, 3
                ymin,ymax = -2, 3
                blu = '#1f77b4'
                plt.plot(np.clip(g - r, xmin, xmax), np.clip(r - i, ymin, ymax),
                         '.', alpha=0.05, color=blu)
                plt.xlabel('g - r (mag)')
                plt.ylabel('r - i (mag)')
                #ax = [-2, +6, -5, +3]
                #plt.axis(ax)
                plt.axis([xmin-0.1, xmax+0.1, ymin-0.1, ymax+0.1])
                plt.axhline(0, color='k', alpha=0.2)
                plt.axvline(0, color='k', alpha=0.2)
                plt.title('Detected in %s' % (one_name))
                plt.savefig('colorcolor-%s.png' % (one_tag))
                plt.savefig('colorcolor-%s.pdf' % (one_tag))

            plt.clf()
            deepfluxes = [detmap[y, x] for detmap in deep_detmaps]
            for flux,band in zip(deepfluxes, bands):
                plt.hist(flux, range=(-0.05, +0.25), bins=40, histtype='step', color={'i':'m'}.get(band,band))
            plt.xlabel('flux (nanomaggies)')
            plt.suptitle('Detected in %s, not in %s: deep fluxes' % (one_name, two_name))
            plt.savefig('unmatched-%s-%s-8.png' % (one_tag, two_tag))

            # Compute the median image around unmatched sources!
            H,W,three = rgb.shape
            s = 8
            I = np.flatnonzero((x >= s) * (y >= s) * (x < W-s) * (y < H-s))
            imstack = []
            for xx,yy in zip(x[I], y[I]):
                imstack.append(rgb[yy-s:yy+s+1, xx-s:xx+s+1, :])
            imstack = np.stack(imstack, axis=-1)
            #print('image stack:', imstack.shape)
            medimg = np.median(imstack, axis=-1)
            plot_oneimage()
            plt.clf()
            plt.imshow(medimg, interpolation='nearest', origin='lower')
            plt.xticks([]); plt.yticks([])
            plt.title('Detected in %s,\nnot in %s' % (one_name, two_name))
            plt.savefig('unmatched-%s-%s-9.png' % (one_tag, two_tag))
            plt.savefig('unmatched-%s-%s-9.pdf' % (one_tag, two_tag))

            if not one_name in medimg_plot:
                medimg_plot.add(one_name)
                # Median source for method A
                s = 8
                I = np.flatnonzero((one_x >= s) * (one_y >= s) *
                                   (one_x < W-s) * (one_y < H-s))
                imstack = []
                for xx,yy in zip(one_x[I], one_y[I]):
                    imstack.append(rgb[yy-s:yy+s+1, xx-s:xx+s+1, :])
                imstack = np.stack(imstack, axis=-1)
                #print('image stack:', imstack.shape)
                medimg = np.median(imstack, axis=-1)
                plot_oneimage()
                plt.clf()
                plt.imshow(medimg, interpolation='nearest', origin='lower')
                plt.xticks([]); plt.yticks([])
                plt.title('Detected in %s' % (one_name))
                plt.savefig('median-%s.png' % (one_tag))
                plt.savefig('median-%s.pdf' % (one_tag))

    I,J,d = match_xy(chi_x, chi_y, chi_pos_x, chi_pos_y, 2)
    U = np.ones(len(chi_x), bool)
    U[I] = False
    K = np.flatnonzero(U)
    V = np.ones(len(chi_pos_x), bool)
    V[J] = False
    L = np.flatnonzero(V)
    plt.clf()
    ha = dict(histtype='step', range=(0, chi2_pos_thresh*2), bins=40, log=True)
    plt.hist(chi_pos_map[chi_pos_y, chi_pos_x], label='Chi+ detections', **ha)
    plt.hist(chi_pos_map[chi_y[I], chi_x[I]], label='Chi detections', **ha)
    plt.hist(chi_pos_map[chi_y[U], chi_x[U]], label='Chi-only detections', **ha)
    plt.hist(chi_pos_map[chi_pos_y[V], chi_pos_x[V]], label='Chi+-only detections',
             **ha)
    plt.xlabel('Value in chi+ map')
    plt.ylabel('Number of detections')
    plt.legend(loc='upper left')
    plt.savefig('chipos-vals.png')

    plt.clf()
    ha = dict(histtype='step', range=(0, chi2_thresh*2), bins=40, log=True)
    plt.hist(chi_map[chi_pos_y, chi_pos_x], label='Chi+ detections', **ha)
    plt.hist(chi_map[chi_y[I], chi_x[I]], label='Chi detections', **ha)
    plt.hist(chi_map[chi_y[U], chi_x[U]], label='Chi-only detections', **ha)
    plt.hist(chi_map[chi_pos_y[V], chi_pos_x[V]], label='Chi+-only detections', **ha)
    plt.xlabel('Value in chisq map')
    plt.ylabel('Number of detections')
    plt.legend(loc='upper left')
    plt.savefig('chisq-vals.png')







if __name__ == '__main__':
    #main()
    new_main()
