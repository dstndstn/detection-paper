'''
Make a set of diverse sources -- not our model -- 

make sources with a range of colours
in noisy data
close to threshold

choose SEDs that basically contain the range of colours
(have our SEDs not-quite contain all the sources)
-stick with our 3 SED models

- look at 5 discrete sources
   (red, red-yellow, yellow, yellow-blue, and blue)

- explore the plane of g S/N, r S/N, measure detection fraction
at a controlled false positive rate

chi-sq methods hacks for negative fluxes
- only include in the chi-sq sum positive fluxes
- add the logs of erfs?
- demand > -1sigma of flux
- demand all positive

measure survey speed

'''
import numpy as np
import scipy
import pylab as plt
import math
import sys
import matplotlib.lines as mlines

from bayes_figure import get_pratio
        
def sed_vec(th):
    r = np.deg2rad(th)
    #if th == 90.:
    #    r = np.pi/2.
    s = np.array([np.cos(r), np.sin(r)])
    s[np.abs(s) < 1e-16] = 0.
    return s

def chisq_detection_raw(sn):
    chisq = np.sum(sn**2, axis=1)
    return chisq

def chisq_detection_pos(sn):
    chisq = np.sum(np.maximum(sn, 0)**2, axis=1)
    return chisq

def chisq_pos_density(d, x):
    p = 0.
    for k in range(d+1):
        if k == 0:
            # delta function at 0
            pchi = 1. * (x == 0)
        else:
            pchi = scipy.stats.chi2(k).pdf(x)
            # scipy.stats.chi2 fails at x==0
            pchi[x == 0.] = 0.
        print('d=%i, k=%i.  d-choose-k %i, sum of pchi2(%i): %f' %
              (d, k, math.comb(d, k), k, np.sum(pchi)))
        p = p + math.comb(d, k) * pchi
    p *= 1./(2**d)
    return p

def chisq_pos_sf(d, x):
    p = 0.
    for k in range(d+1):
        if k == 0:
            # delta function at 0
            pchi = 1. * (x == 0)
        else:
            pchi = scipy.stats.chi2(k).sf(x)
            # scipy.stats.chi2 fails at x==0
            pchi[x == 0.] = 1.
        p = p + math.comb(d, k) * pchi
    p *= 1./(2**d)
    return p

def chisq_pos_isf(d, pval):
    lo = 0
    hi = 100
    # find hi in case it's really huge
    while chisq_pos_sf(d, np.array([hi]))[0] > pval:
        print('search for hi:', hi, chisq_pos_sf(d, np.array([hi]))[0], pval)
        lo = hi
        hi *= 2
    # now binary search
    while True:
        if lo == hi:
            return lo
        mid = (lo + hi) / 2.
        if mid == lo or mid == hi:
            return mid
        #print('search for isf: lo', lo, 'hi', hi, 'mid', mid, 'isf',
        #      chisq_pos_sf(d, np.array([mid])), pval)
        sf = chisq_pos_sf(d, np.array([mid]))[0]
        if sf > pval:
            lo = mid
        else:
            hi = mid
        #if np.abs(sf - pval) < pval*1e-6:
        #    # close enough!
        #    return mid

def sed_union_detection(fluxes, sig_fluxes, seds):
    # fluxes: npix x nbands array
    # sig_fluxes: nbands
    # seds: [(name, scalar sed, weight -- ignored)]
    npix,nb = fluxes.shape
    x = np.zeros(npix)
    for sed in seds:
        name,sed_vec,_ = sed
        pr = get_pratio(fluxes.T[:, :, np.newaxis], sig_fluxes, sed_vec)
        #print('p-ratio for SED', name, sed_vec, ':', pr.shape)
        pr = pr[:,0]
        x = np.maximum(x, pr)
    return x

def sed_mixture_detection(fluxes, sig_fluxes, seds):
    # fluxes: npix x nbands array
    # sig_fluxes: nbands
    # seds: [(name, scalar sed, weight -- ignored)]
    x = 0.
    for sed in seds:
        name,sed_vec,wt = sed
        pr = get_pratio(fluxes.T[:, :, np.newaxis], sig_fluxes, sed_vec)
        pr = pr[:,0]
        x = x + wt * pr
    return x

def main():

    real_sn = 6.

    # g,r
    n_bands = 2

    # sensitivities (noise levels) in the g,r bands.
    noise_levels = np.array([1.0, 1.0])
    #noise_levels = np.array([1.0, 0.5])
    
    real_seds = [('red', sed_vec(90.)),
                 ('orange', sed_vec(67.5)),
                 ('yellow', sed_vec(45.)),
                 ('green', sed_vec(22.5)),
                 ('blue', sed_vec(0.))]
    #print('SEDs:', real_seds)

    th_red = np.rad2deg(np.arctan2(2.5, 1.0))
    model_seds = [('r-only', sed_vec(90.), 0.02),
                  ('red', sed_vec(th_red), 0.49),
                  ('blue', sed_vec(45.), 0.49),]

    
    # Draw background samples to set/confirm thresholds?
    g = scipy.stats.norm()

    g_sigma = 4.
    falsepos_rate = g.sf(g_sigma)

    #print('Gaussian 4 sigma survival function:', g.sf(4.))
    #print('Gaussian 5 sigma survival function:', g.sf(5.))
    print('Gaussian %.3f sigma survival function:' % g_sigma, falsepos_rate)
    
    ch = scipy.stats.chi2(n_bands)
    ch1 = scipy.stats.chi2(1)

    chi2_thresh = ch.isf(falsepos_rate)
    print('Chi2 thresh:', chi2_thresh)

    if False:
        plt.clf()
        xx = np.linspace(0, 5, 500)
        yy1 = chisq_pos_density(1, xx)
        yy2 = chisq_pos_density(2, xx)
        yy3 = chisq_pos_density(3, xx)
        yych = ch.pdf(xx)
        print('Sums:', np.sum(yych), np.sum(yy1), np.sum(yy2), np.sum(yy3))
        plt.plot(xx, yy1, '-', label='chisq_pos(1 dof)')
        plt.plot(xx, yy2, '-', label='chisq_pos(2 dof)')
        plt.plot(xx, yy3, '-', label='chisq_pos(3 dof)')
        plt.plot(xx, yych, '-', label='chisq(2 dof)')
        plt.legend()
        plt.savefig('1.png')
        plt.yscale('log')
        plt.savefig('2.png')
        
        plt.clf()
        xx = np.linspace(0, 30, 500)
        yy1 = chisq_pos_sf(1, xx)
        yy2 = chisq_pos_sf(2, xx)
        yy3 = chisq_pos_sf(3, xx)
        yych = ch.sf(xx)
        yych1 = ch1.sf(xx)
        plt.plot(xx, yy1, '-', label='chisq_pos(1 dof)')
        plt.plot(xx, yy2, '-', label='chisq_pos(2 dof)')
        plt.plot(xx, yy3, '-', label='chisq_pos(3 dof)')
        plt.plot(xx, yych, '-', label='chisq(2 dof)')
        plt.plot(xx, yych1, '-', label='chisq(1 dof)')
        plt.legend()
        plt.savefig('3.png')
        plt.yscale('log')
        plt.savefig('4.png')

    chi2_pos_thresh = chisq_pos_isf(n_bands, falsepos_rate)
    print('thresh:', chi2_pos_thresh)


    # False pos rates for SED-matched detectors...
    sn_g = np.linspace(-10, +10, 201)
    sn_r = sn_g
    dg = sn_g[1] - sn_g[0]
    dr = sn_r[1] - sn_r[0]
    sn_shape = (len(sn_r), len(sn_g))
    snmesh = np.meshgrid(sn_g, sn_r)
    sn = np.vstack([x.ravel() for x in snmesh]).T
    sn_g,sn_r = snmesh
    pbg = dg * dr * 1./(2.*np.pi) * np.exp(-0.5 * (sn_g**2 + sn_r**2))
    print('sum pbg:', np.sum(pbg))
    # Union of SED-matched detections
    flux = sn * noise_levels[np.newaxis, :]
    x = sed_union_detection(flux, noise_levels, model_seds)
    x = x.reshape(sn_shape)
    # Find threshold
    X = scipy.optimize.root_scalar(lambda th: np.sum(pbg * (x > th)) - falsepos_rate,
                                   method='bisect', bracket=(0, 1e10))
    print('Thresh:', X)
    assert(X.converged)
    sed_union_th = X.root

    # Mixture of SED-matched detections
    x = sed_mixture_detection(flux, noise_levels, model_seds)
    x = x.reshape(sn_shape)
    # Find threshold
    X = scipy.optimize.root_scalar(lambda th: np.sum(pbg * (x > th)) - falsepos_rate,
                                   method='bisect', bracket=(0, 1e10))
    print('Thresh:', X)
    assert(X.converged)
    sed_mixture_th = X.root

    
    # plt.clf()
    # sn_extent = (sn_g.min(), sn_g.max(), sn_r.min(), sn_r.max())
    # plt.imshow(x > sed_mixture_th, extent=sn_extent, origin='lower', interpolation='nearest')
    # plt.savefig('1.png')

    sn_g = np.linspace(-5, +8, 600)
    sn_r = np.linspace(-5, +8, 600)
    sn_shape = (len(sn_r), len(sn_g))
    sn = np.meshgrid(sn_g, sn_r)
    sn = np.vstack([x.ravel() for x in sn]).T

    # Union of SED-matched detections
    flux = sn * noise_levels[np.newaxis, :]
    x = sed_union_detection(flux, noise_levels, model_seds)
    x = x.reshape(sn_shape)
    sed_union_det = (x > sed_union_th)

    # Mixture of SED-matched detections
    x = sed_mixture_detection(flux, noise_levels, model_seds)
    x = x.reshape(sn_shape)
    sed_mixture_det = (x > sed_mixture_th)

    x = chisq_detection_raw(sn)
    x = x.reshape(sn_shape)
    chisq_raw_det = (x > chi2_thresh)

    x = chisq_detection_pos(sn)
    x = x.reshape(sn_shape)
    chisq_pos_det = (x > chi2_pos_thresh)
    
    sn_extent = (sn_g.min(), sn_g.max(), sn_r.min(), sn_r.max())

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    plt.clf()
    p1 = plt.contour(chisq_raw_det,   extent=sn_extent, levels=[0.5], colors=[colors[0]])
    p2 = plt.contour(chisq_pos_det,   extent=sn_extent, levels=[0.5], colors=[colors[1]])
    p3 = plt.contour(sed_union_det,   extent=sn_extent, levels=[0.5], colors=[colors[2]])
    p4 = plt.contour(sed_mixture_det, extent=sn_extent, levels=[0.5], colors=[colors[3]])

    lines = [mlines.Line2D([], [], color=colors[i], label=lab)
             for i,lab in enumerate(['Chi2 (raw)', 'Chi2 (pos)', 'SED union', 'SED mixture'])]
    plt.legend(handles=lines)
    plt.axhline(0.)
    plt.axvline(0.)
    plt.title('Decision boundaries for chi-squared versus SED-match detectors')
    plt.xlabel('g-band S/N')
    plt.ylabel('r-band S/N')
    plt.savefig('1.png')


    #n_star = 1_000_000
    n_star = 1_000
    starnoise = np.random.normal(size=(n_star, n_bands))

    k = 3
    
    for sed_name, sed in real_seds:

        sns = g_sigma * sed
        print('SED', sed_name, 'S/N values:', sns)

        noisy = sns[np.newaxis,:] + starnoise

        det1 = (chisq_detection_raw(noisy) > chi2_thresh)
        det2 = (chisq_detection_pos(noisy) > chi2_pos_thresh)
        flux = noisy * noise_levels[np.newaxis, :]
        det3 = (sed_union_detection(flux, noise_levels, model_seds) > sed_union_th)
        det4 = (sed_mixture_detection(flux, noise_levels, model_seds) > sed_mixture_th)

        print('Detection rates:')
        print('  chi2(raw)    %6.2f %%' % (100. * np.sum(det1) / n_star))
        print('  chi2(pos)    %6.2f %%' % (100. * np.sum(det2) / n_star))
        print('  sed(union)   %6.2f %%' % (100. * np.sum(det3) / n_star))
        print('  sed(mixture) %6.2f %%' % (100. * np.sum(det4) / n_star))
        
        plt.clf()
        p1 = plt.contour(chisq_raw_det,   extent=sn_extent, levels=[0.5], colors=[colors[0]])
        p2 = plt.contour(chisq_pos_det,   extent=sn_extent, levels=[0.5], colors=[colors[1]])
        p3 = plt.contour(sed_union_det,   extent=sn_extent, levels=[0.5], colors=[colors[2]])
        p4 = plt.contour(sed_mixture_det, extent=sn_extent, levels=[0.5], colors=[colors[3]])

        lines = [mlines.Line2D([], [], color=colors[i], label=lab)
                 for i,lab in enumerate(['Chi2 (raw)', 'Chi2 (pos)', 'SED union', 'SED mixture'])]
        plt.legend(handles=lines)
        plt.axhline(0.)
        plt.axvline(0.)
        plt.title('Decision boundaries for chi-squared versus SED-match detectors')
        plt.xlabel('g-band S/N')
        plt.ylabel('r-band S/N')
        plt.plot(noisy[:,0], noisy[:,1], 'k.', alpha=0.1)
        plt.savefig('%i.png' % k)
        k += 1


    print('Detection-rate experiment...')
    n_star = 100_000
    starnoise = np.random.normal(size=(n_star, n_bands))
    #angles = np.linspace(90., 0., 19)
    angles = np.linspace(105., -15., 50)
    rates = []
    for angle in angles:
        sed = sed_vec(angle)
        sns = g_sigma * sed
        noisy = sns[np.newaxis,:] + starnoise
        det1 = (chisq_detection_raw(noisy) > chi2_thresh)
        det2 = (chisq_detection_pos(noisy) > chi2_pos_thresh)
        flux = noisy * noise_levels[np.newaxis, :]
        det3 = (sed_union_detection(flux, noise_levels, model_seds) > sed_union_th)
        det4 = (sed_mixture_detection(flux, noise_levels, model_seds) > sed_mixture_th)
        rates.append([100. * np.sum(d) / n_star for d in [det1,det2,det3,det4]])
    rates = np.array(rates)

    plt.clf()
    plt.plot(angles, rates, '-')
    plt.xlabel('SED angle (deg)')
    plt.ylabel('Detection rate (%)')
    plt.legend(['Chi2 (raw)', 'Chi2 (pos)', 'SED union', 'SED mixture'])
    plt.savefig('8.png')


    print('Detection-rate experiment 2...')
    #n_star = 1_000_000
    n_star = 100_000
    starnoise = np.random.normal(size=(n_star, n_bands))
    colors = np.linspace(-2, +4, 50)
    rates = []
    for color in colors:
        # g mag - r mag = color
        rmag = 0
        gmag = color
        rflux = 10.**(rmag / -2.5)
        gflux = 10.**(gmag / -2.5)
        sed = np.array([gflux, rflux]) / np.hypot(gflux, rflux)

        sns = g_sigma * sed
        noisy = sns[np.newaxis,:] + starnoise
        det1 = (chisq_detection_raw(noisy) > chi2_thresh)
        det2 = (chisq_detection_pos(noisy) > chi2_pos_thresh)
        flux = noisy * noise_levels[np.newaxis, :]
        det3 = (sed_union_detection(flux, noise_levels, model_seds) > sed_union_th)
        det4 = (sed_mixture_detection(flux, noise_levels, model_seds) > sed_mixture_th)
        rates.append([100. * np.sum(d) / n_star for d in [det1,det2,det3,det4]])
    rates = np.array(rates)

    plt.clf()
    plt.plot(colors, rates, '-')
    plt.xlabel('Star g-r color (mag)')
    plt.ylabel('Detection rate (\\%)')
    plt.legend(['Chi2 (raw)', 'Chi2 (pos)', 'SED union', 'SED mixture'])
    plt.title('Detection rate for stars of different colors')
    plt.axvline(0., linestyle='--', alpha=0.2, color='k')
    plt.axvline(1., linestyle=':', alpha=0.2, color='k')
    plt.savefig('9.png')




    
    
    #sys.exit(0)
    
    # plt.clf()
    # plt.imshow(np.log10(x),
    #            extent=sn_extent,
    #            origin='lower', interpolation='nearest')#, cmap='gray')
    # plt.colorbar()
    # plt.contour(np.log10(x), extent=sn_extent, levels=[5],
    #             colors='r')
    # plt.axhline(0.)
    # plt.axvline(0.)
    # plt.savefig('1.png')


    

    plt.clf()
    plt.imshow(np.log10(x),
               extent=sn_extent,
               origin='lower', interpolation='nearest')#, cmap='gray')
    plt.contour(np.log10(x), extent=sn_extent, levels=[5],
                colors='r')
    plt.axhline(0.)
    plt.axvline(0.)
    plt.savefig('2.png')




    
    n_bg = 10_000_000
    bg = np.random.normal(size=(n_bg, n_bands))
    x = chisq_detection_raw(bg)
    det = (x > chi2_thresh)
    print('False detections: %i (rate %.5g, irate %.5g)' % (np.sum(det), np.sum(det) / n_bg, n_bg/np.sum(det)))

    print('Only count positive fluxes:')
    x = chisq_detection_pos(bg)
    det = (x > chi2_pos_thresh)
    print('False detections: %i (rate %.5g, irate %.5g)' % (np.sum(det), np.sum(det) / n_bg, n_bg/np.sum(det)))

    if False:
        plt.clf()
        n,b,p = plt.hist(x, bins=30, range=(0,30))
        xl,xh = plt.xlim()
        xx = np.linspace(xl, xh, 100)
        yy = ch.pdf(xx)
        plt.plot(xx, yy/np.max(yy) * max(n), '-')
        plt.yscale('log')
        plt.savefig('1.png')

    
        
    for sed_name, sed in real_seds:

        pass
    
    
if __name__ == '__main__':
    main()
