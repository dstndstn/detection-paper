import matplotlib
matplotlib.rcParams['figure.figsize'] = (3.5,3.5)
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
matplotlib.rc('axes', titlesize='medium')
import numpy as np
import scipy
import pylab as plt
import math
import sys
import matplotlib.lines as mlines

import bayes_figure
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

def sed_mixture_detection(fluxes, sig_fluxes, seds, alpha=None):
    '''
    Computes the foreground-to-background likelihood ratio for the
    Bayesian SED mixture model.
    
    fluxes: npix x nbands array
    -- of detection-map values, D_j

    sig_fluxes: nbands
    -- uncertainties on the detection maps - sigma_(D_j)

    seds: [(name, sed, weight)]
    -- where each "sed" is an "nbands" element vector
    '''
    if alpha is None:
        alpha = 1.
    x = 0.
    for sed in seds:
        name,sed_vec,wt = sed
        pr = get_pratio(fluxes.T[:, :, np.newaxis], sig_fluxes, sed_vec,
                        alpha=alpha)
        pr = pr[:,0]
        x = x + wt * pr
    return x

def sed_union_threshold(flux_th, noise_level, model_seds, pbg, falsepos_rate):
    x = sed_union_detection(flux_th, noise_level, model_seds)
    x = x.reshape(pbg.shape)
    # Find threshold
    X = scipy.optimize.root_scalar(lambda th: np.sum(pbg * (x > th)) - falsepos_rate,
                                   method='bisect', bracket=(0, 1e10))
    print('SED(union) Thresh:', X)
    assert(X.converged)
    return X.root

def sed_mixture_threshold(flux_th, noise_level, model_seds, pbg, falsepos_rate, alpha=None):
    # "x" here is the p(flux|source) / p(flux|noise) likelihood ratio.
    x = sed_mixture_detection(flux_th, noise_level, model_seds, alpha=alpha)
    x = x.reshape(pbg.shape)
    # ie, the sum of pfg is 1.
    # pfg = x * pbg
    # and the sum of pbg is 1.
    # (the "flux_th" here is assumed to be a uniform grid in the g and r fluxes)

    # We want to set our threshold "th" on the likelihood ratio so that the
    # false positive rate is equal to the target "falsepos_rate".

    # Find threshold
    def fprate(th):
        return np.sum(pbg * (x > th))

    X = scipy.optimize.root_scalar(lambda th: fprate(th) - falsepos_rate,
                                   method='bisect', bracket=(0, 1e20), maxiter=200)
    print('SED(mixture) Thresh:', X)
    assert(X.converged)
    return X.root



def main(plots=None):

    def do_plot(name):
        return plots is None or name in plots

    real_sn = 6.

    # g,r
    n_bands = 2

    # sensitivities (noise levels) in the g,r bands.
    noise_levels = [np.array([1.0, 1.0]),
                    np.array([1.0, 0.5])]
    
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

    sn_g_th = np.linspace(-10, +10, 201)
    sn_r_th = sn_g_th
    dg = sn_g_th[1] - sn_g_th[0]
    dr = sn_r_th[1] - sn_r_th[0]
    sn_th_shape = (len(sn_r_th), len(sn_g_th))
    snmesh = np.meshgrid(sn_g_th, sn_r_th)
    sn_th = np.vstack([x.ravel() for x in snmesh]).T
    sn_g_th,sn_r_th = snmesh
    pbg = dg * dr * 1./(2.*np.pi) * np.exp(-0.5 * (sn_g_th**2 + sn_r_th**2))
    print('sum pbg:', np.sum(pbg))

    det_names = [r'$\chi^2$', r'$\chi_+^2$', 'SED (union)', 'SED (Bayes)']
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    mpl_colors = colors

    linestyles = ['solid', 'dashed', 'solid', 'dashed'] #, 'dashdot', 'dotted']
    linewidths = [1, 3, 1, 3]
    alphas = [1, 0.5, 1, 0.5]
    xcolors = [colors[0], colors[0], colors[1], colors[1]]
    #labels = ['$\chi^2$ (raw)', '$\chi^2$ (pos)', 'SED (union)', 'SED (mixture)']
    labels = [r'$\chi^2$', r'$\chi_+^2$', 'SED (union)', 'SED (Bayes)']

    if do_plot('alpha'):
        noise_level = noise_levels[0]

        #sn_g_thaa = np.linspace(-10, +10, 501)
        sn_g_thaa = np.linspace(-15, +15, 501)
        sn_r_thaa = sn_g_thaa
        dg_aa = sn_g_thaa[1] - sn_g_thaa[0]
        dr_aa = sn_r_thaa[1] - sn_r_thaa[0]
        sn_thaa_shape = (len(sn_r_thaa), len(sn_g_thaa))
        snmesh_aa = np.meshgrid(sn_g_thaa, sn_r_thaa)
        sn_g_thaa,sn_r_thaa = snmesh_aa
        sn_thaa = np.vstack([x.ravel() for x in snmesh_aa]).T
        pbg_aa = dg_aa * dr_aa * 1./(2.*np.pi) * np.exp(-0.5 * (sn_g_thaa**2 + sn_r_thaa**2))
        print('sum pbg_aa:', np.sum(pbg_aa))
        # recompute SED thresholds
        flux_thaa = sn_thaa * noise_level[np.newaxis, :]

        #sn_g = np.linspace(-5, +10, 600)
        #sn_r = np.linspace(-5, +10, 600)
        sn_g = np.linspace(-5, +9, 600)
        sn_r = np.linspace(-5, +9, 600)
        #sn_g = np.linspace(-5, +11, 600)
        #sn_r = np.linspace(-5, +11, 600)
        sn_shape = (len(sn_r), len(sn_g))
        sn = np.meshgrid(sn_g, sn_r)
        sn = np.vstack([x.ravel() for x in sn]).T
        # Union of SED-matched detections
        flux = sn * noise_level[np.newaxis, :]
        sn_extent = (sn_g.min(), sn_g.max(), sn_r.min(), sn_r.max())

        #acolors = [xcolors[0]]*3
        #alinestyles=['-','--',':']
        #alinewidths=[1, 2, 3]
        #aalphas=[1, 0.5, 0.3]

        # acolors = ['b']*3
        # alinestyles=['-','--','-']
        # alinewidths=[1, 2, 3]
        # aalphas=[1, 0.5, 0.3]

        acolors = ['b','r']
        alinestyles=['-','--']
        alinewidths=[1.5, 1.5]
        aalphas=[1, 1]
        
        plt.clf()
        bayes_figure.subplots()
        #plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.99)
        lines = []
        labels = []

        #alpha_vals = [0.01, 1., 5.] #3., 5., 8., 10.]
        #alpha_labels = ['Bright prior', 'Faint prior', 'Fainter prior']
        #alpha_vals = [5., 1., 0.01] #3., 5., 8., 10.]
        #alpha_vals = [5., 0.1, 0.01] #3., 5., 8., 10.]
        # alpha_vals = [5., 2., 0.01] #3., 5., 8., 10.]
        # alpha_labels = ['Faint prior', 'Medium prior', 'Bright prior']
        alpha_vals = [4., 1]
        alpha_labels = ['Faint prior', 'Bright prior']
        print('Alpha decision boundaries')
        #for j,fp_rate in enumerate([1e-1, 1e-2, 1e-3, 1e-4]):
        for i,(a, alpha_label) in enumerate(zip(alpha_vals, alpha_labels)):
            print('Alpha', a)

            #plt.clf()

            for j,fp_rate in enumerate([#1e-1, 1e-3, 1e-5, 1e-7, 1e-9,
                                        1e-1, 1e-4, 1e-7,
                                        #1e-13, 1e-19]):
                                        ]):
                if a < 5 and fp_rate < 1e-9:
                    continue
                print('FP rate', fp_rate)
                if j == 0:
                    labels.append(alpha_label)
                thresh = sed_mixture_threshold(flux_thaa, noise_level, model_seds, pbg_aa,
                                               fp_rate, alpha=a)
                print('Threshold:', thresh)

                x = sed_mixture_detection(flux, noise_level, model_seds, alpha=a)
                x = x.reshape(sn_shape)
                sed_det = (x > thresh)

                # plt.clf()
                # xsum = 0.
                # for k in range(3):
                #     plt.subplot(2,2,k+1)
                #     x1 = sed_mixture_detection(flux, noise_level, [model_seds[k]], alpha=a)
                #     x1 = x1.reshape(sn_shape)
                #     plt.imshow(x1/thresh, interpolation='nearest', origin='lower',
                #                vmin=0, vmax=2)
                #     plt.title(model_seds[k][0])
                #     xsum = xsum + x1
                # plt.subplot(2,2,4)
                # plt.imshow(xsum/thresh, interpolation='nearest', origin='lower',
                #            vmin=0, vmax=2)
                # plt.savefig('det-alpha%g-fp%i.png' % (a, int(np.round(np.log10(fp_rate)))))

                # plt.clf()
                # plt.imshow(x/thresh, interpolation='nearest', origin='lower',
                #            vmin=0, vmax=2)
                # plt.colorbar()
                # plt.savefig('det-alpha%g-fp%g.png' % (a, fp_rate))
                ai = i % len(acolors)
                plt.contour(sed_det, extent=sn_extent, levels=[0.5],
                            colors=[acolors[ai]], linestyles=[alinestyles[ai]],
                            linewidths=[alinewidths[ai]], alpha=aalphas[ai])
                ll = mlines.Line2D(
                    [], [], label=labels[i],
                    color=acolors[ai], linestyle=alinestyles[ai],
                    linewidth=alinewidths[ai], alpha=aalphas[ai])
                if j == 0:
                    lines.append(ll)
        plt.legend(handles=lines, framealpha=1.0)
        plt.axhline(0., color='k', alpha=0.25)
        plt.axvline(0., color='k', alpha=0.25)
        #plt.title('Decision boundaries for chi-squared versus SED-match detectors')
        plt.xlabel('g-band S/N')
        plt.ylabel('r-band S/N')
        plt.xticks(np.arange(-4, 10, 2))
        #plt.savefig('alpha-det-%g.png' % a)
        plt.title('Decision boundaries with different flux priors')
        plt.savefig('alpha-det.png')
        plt.savefig('alpha-det.pdf')
            
    # Colors of false-positive detections for the different detectors.
    if do_plot('chisq-det-color'):
        print('Detection-rate experiment 3...')
        noise_level = noise_levels[0]

        # recompute SED thresholds
        flux_th = sn_th * noise_level[np.newaxis, :]
        sed_union_th = sed_union_threshold(flux_th, noise_level, model_seds, pbg, falsepos_rate)
        sed_mixture_th = sed_mixture_threshold(flux_th, noise_level, model_seds, pbg, falsepos_rate)

        n_star = 100_000_000
        noisy = np.random.normal(size=(n_star, n_bands))
        # immediately cut stars within 3-sigma of zero
        noisy = noisy[np.sum(noisy**2, axis=1)>9., :]
        print('Cut to', len(noisy), '>3sigma')
        
        det1 = (chisq_detection_raw(noisy) > chi2_thresh)
        det2 = (chisq_detection_pos(noisy) > chi2_pos_thresh)
        flux = noisy * noise_level[np.newaxis, :]
        det3 = (sed_union_detection(flux, noise_level, model_seds) > sed_union_th)
        det4 = (sed_mixture_detection(flux, noise_level, model_seds) > sed_mixture_th)

        plt.figure(figsize=(4.5, 3.5))
        bayes_figure.subplots()
        plt.clf()
        plt.subplots_adjust(hspace=0.05)#, top=0.95)
        BOTH_NEG = -3
        R_NEG = -2
        G_NEG = 4
    
        histrange = BOTH_NEG, G_NEG
    
        ymax = 0
        for i,det in enumerate([det1,det2,det3,det4]):
            print('Number of false dets for', det_names[i], ':', np.sum(det))
            g = noisy[det, 0]
            r = noisy[det, 1]
            ok = np.flatnonzero((g > 0) * (r > 0))
            # good colors
            c = -2.5 * (np.log10(g[ok]) - np.log10(r[ok]))
            c = np.clip(c, -1, 2.9999)

            # fluxes in bad quadrants
            for condition,code,codetxt in [((g <= 0) * (r <= 0), BOTH_NEG, 'both neg'),
                                           ((g <= 0) * (r >  0), G_NEG,    'g neg'),
                                           ((g >  0) * (r <= 0), R_NEG,    'r neg')]:
                nbad = np.sum(condition)
                print('  %i with %s' % (nbad, codetxt))
                if nbad > 0:
                    c = np.append(c, [code] * nbad)

            plt.subplot(4,1,i+1)
            plt.hist(c, range=histrange, bins=28, color=mpl_colors[i],
                     label=det_names[i])
            if i == 1:
                plt.ylabel('Number of false detections', ha='center', y=0)
            plt.axvline(0., linestyle='--', alpha=0.2, color='k')
            plt.axvline(1., linestyle=':', alpha=0.2, color='k')
            plt.axvline(-1, linestyle='-', alpha=0.1, color='k')
            plt.axvline(+3, linestyle='-', alpha=0.1, color='k')
            plt.legend(loc='upper right', bbox_to_anchor=(0.9, 1))
            if True:#i > 0:
                yl,yh = plt.ylim()
                ymax = max(ymax, yh)
            if i < 3:
                plt.xticks([])
        for i in range(4):
            plt.subplot(4,1,i+1)
            plt.ylim(0, ymax + 3)
        ticks = list(range(-1, 3+1))
        plt.xticks([BOTH_NEG, R_NEG] + ticks + [G_NEG],
                   [r'$g,r<0$', r'$r<0$'] + ['%i'%i for i in ticks] + [r'$g<0$'])
        plt.xlabel(r"``Star'' $g-r$ color (mag)")
        plt.suptitle('Colors of false positive detections')
        plt.savefig('chisq-det-colors.png')
        plt.savefig('chisq-det-colors.pdf')

    if do_plot('decision-boundaries'):

        bayes_figure.subplots()

        for jnoise, noise_level in enumerate(noise_levels):
            print('Noise in g,r:', noise_level)
    
            # Union of SED-matched detections
            flux_th = sn_th * noise_level[np.newaxis, :]
            print('flux_th', flux_th.shape, 'sn_th_shape', sn_th_shape, 'pbg', pbg.shape)
            sed_union_th = sed_union_threshold(flux_th, noise_level, model_seds, pbg, falsepos_rate)
    
            # Mixture of SED-matched detections
            sed_mixture_th = sed_mixture_threshold(flux_th, noise_level, model_seds, pbg, falsepos_rate)
    
            # plt.clf()
            # sn_extent = (sn_g.min(), sn_g.max(), sn_r.min(), sn_r.max())
            # plt.imshow(x > sed_mixture_th, extent=sn_extent, origin='lower', interpolation='nearest')
            # plt.savefig('1.png')
    
    
            # Plot decision boundaries (numerically)
    
            #sn_g = np.linspace(-5, +7, 600)
            #sn_r = np.linspace(-5, +7, 600)
            sn_g = np.linspace(-6, +8, 600)
            sn_r = np.linspace(-6, +8, 600)
            sn_shape = (len(sn_r), len(sn_g))
            sn = np.meshgrid(sn_g, sn_r)
            sn = np.vstack([x.ravel() for x in sn]).T
    
            # Union of SED-matched detections
            flux = sn * noise_level[np.newaxis, :]
            x = sed_union_detection(flux, noise_level, model_seds)
            x = x.reshape(sn_shape)
            sed_union_det = (x > sed_union_th)
    
            # Mixture of SED-matched detections
            x = sed_mixture_detection(flux, noise_level, model_seds)
            x = x.reshape(sn_shape)
            sed_mixture_det = (x > sed_mixture_th)
    
            # # Mixture of SED-matched detections (alpha=alpha_sed)
            # x = sed_mixture_detection(flux, noise_level, model_seds, alpha=alpha_sed)
            # x = x.reshape(sn_shape)
            # sed_mixture_det_a2 = (x > sed_mixture_th_a2)
    
            x = chisq_detection_raw(sn)
            x = x.reshape(sn_shape)
            chisq_raw_det = (x > chi2_thresh)
    
            x = chisq_detection_pos(sn)
            x = x.reshape(sn_shape)
            chisq_pos_det = (x > chi2_pos_thresh)
    
            sn_extent = (sn_g.min(), sn_g.max(), sn_r.min(), sn_r.max())
    
    
            plt.clf()
            #plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.99)
            #plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.95)
            lines = []
            for i,vals in enumerate([chisq_raw_det, chisq_pos_det, sed_union_det, sed_mixture_det]):
                plt.contour(vals, extent=sn_extent, levels=[0.5],
                            colors=[xcolors[i]], linestyles=[linestyles[i]],
                            linewidths=[linewidths[i]], alpha=alphas[i])
                lines.append(mlines.Line2D(
                    [], [], label=labels[i],
                    color=xcolors[i], linestyle=linestyles[i],
                    linewidth=linewidths[i], alpha=alphas[i]))
            #plt.legend(handles=lines)
            plt.legend(handles=lines, loc='lower left')
            plt.axhline(0., color='k', alpha=0.25)
            plt.axvline(0., color='k', alpha=0.25)
            #plt.title('Decision boundaries for chi-squared versus SED-match detectors')
            plt.xlabel('$g$-band S/N')
            plt.ylabel('$r$-band S/N')
            #plt.title('$r$-band sensitivity = $g$-band sensitivity' +
            #          (' $\\times 2$' if (jnoise == 1) else ''))

            kwa = dict(xy=(-5,6),
                       bbox=dict(edgecolor='none', facecolor='white', alpha=1.))
            if jnoise == 0:
                plt.annotate('$r$-band sensitivity = $g$-band sensitivity', **kwa)
            else:
                plt.annotate('$r$-band sensitivity = $2 \\times$ $g$-band sensitivity',
                             **kwa)

            plt.xticks(np.arange(-4, 10, 2))

            name = ['chisq-decision-boundary',
                    'chisq-decision-boundary-sens'][jnoise]

            plt.title('Decision boundaries')
            
            plt.savefig('%s.png' % (name))
            plt.savefig('%s.pdf' % (name))

        # An earlier edition of the "faint/bright prior alpha plot"
        # if jnoise == 0:
        #     plt.clf()
        #     plt.subplots_adjust(left=0.1, bottom=0.1, right=0.99, top=0.99)
        #     lines = []
        #     labels = []
        # 
        #     for i,alpha_sed in enumerate([8., 4., 1., 0.01]): #, 0.25, 1., 4.]):
        #         labels.append('SED (Bayes, alpha=%g)' % alpha_sed)
        # 
        #         # Mixture of SED-matched detections (alpha=alpha_sed)
        #         x = sed_mixture_detection(flux_th, noise_level, model_seds, alpha=alpha_sed)
        #         x = x.reshape(sn_th_shape)
        #         # Find threshold
        #         X = scipy.optimize.root_scalar(lambda th: np.sum(pbg * (x > th)) - falsepos_rate,
        #                                        method='bisect', bracket=(0, 1e10))
        #         print('SED(mixture, alpha=%g) Thresh:' % alpha_sed, X)
        #         assert(X.converged)
        #         thresh = X.root
        #         #threshs.append(thresh)
        #         #sed_mixture_th_a2 = X.root
        # 
        #         x = sed_mixture_detection(flux, noise_level, model_seds, alpha=alpha_sed)
        #         x = x.reshape(sn_shape)
        #         sed_det = (x > thresh)
        #         
        #         plt.contour(sed_det, extent=sn_extent, levels=[0.5],
        #                     colors=[xcolors[i]], linestyles=[linestyles[i]],
        #                     linewidths=[linewidths[i]], alpha=alphas[i])
        #         lines.append(mlines.Line2D(
        #             [], [], label=labels[i],
        #             color=xcolors[i], linestyle=linestyles[i],
        #             linewidth=linewidths[i], alpha=alphas[i]))
        #     plt.legend(handles=lines)
        #     plt.axhline(0., color='k', alpha=0.25)
        #     plt.axvline(0., color='k', alpha=0.25)
        #     #plt.title('Decision boundaries for chi-squared versus SED-match detectors')
        #     plt.xlabel('g-band S/N')
        #     plt.ylabel('r-band S/N')
        #     plt.savefig('alpha%i.png' % (1+jnoise))
        #     plt.savefig('alpha%i.pdf' % (1+jnoise))

    if False:
        noise_level = noise_levels[0]

        #n_star = 1_000_000
        n_star = 1_000
        starnoise = np.random.normal(size=(n_star, n_bands))
    
        k = 2
        
        for sed_name, sed in real_seds:
    
            sns = g_sigma * sed
            print('SED', sed_name, 'S/N values:', sns)
    
            noisy = sns[np.newaxis,:] + starnoise
    
            det1 = (chisq_detection_raw(noisy) > chi2_thresh)
            det2 = (chisq_detection_pos(noisy) > chi2_pos_thresh)
            flux = noisy * noise_level[np.newaxis, :]
            det3 = (sed_union_detection(flux, noise_level, model_seds) > sed_union_th)
            det4 = (sed_mixture_detection(flux, noise_level, model_seds) > sed_mixture_th)
    
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
            pfn = '%i.png' % k
            plt.savefig(pfn)
            print('Wrote', pfn)
            k += 1

    noise_level = noise_levels[0]

    if False:
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
            flux = noisy * noise_level[np.newaxis, :]
            det3 = (sed_union_detection(flux, noise_level, model_seds) > sed_union_th)
            det4 = (sed_mixture_detection(flux, noise_level, model_seds) > sed_mixture_th)
            rates.append([100. * np.sum(d) / n_star for d in [det1,det2,det3,det4]])
        rates = np.array(rates)
    
        plt.clf()
        plt.plot(angles, rates, '-')
        plt.xlabel('SED angle (deg)')
        plt.ylabel('Detection rate (\\%)')
        plt.legend(['Chi2 (raw)', 'Chi2 (pos)', 'SED union', 'SED mixture'])
        plt.savefig('8.png')

    if do_plot('chisq-color'):
        bayes_figure.subplots()

        print('Detection-rate experiment 2...')
        n_star = 1_000_000
        #n_star = 100_000
        starnoise = np.random.normal(size=(n_star, n_bands))
        colors = np.linspace(-2, +4, 50)
        rates = []
    
        noise_level = noise_levels[0]
        # recompute SED thresholds
        flux_th = sn_th * noise_level[np.newaxis, :]
        sed_union_th = sed_union_threshold(flux_th, noise_level, model_seds, pbg, falsepos_rate)
        sed_mixture_th = sed_mixture_threshold(flux_th, noise_level, model_seds, pbg, falsepos_rate)
    
        print('chi2_thresh', chi2_thresh)
        
        for icolor,color in enumerate(colors):
            print(icolor, 'color', color)
            # g mag - r mag = color
            rmag = 0
            gmag = color
            rflux = 10.**(rmag / -2.5)
            gflux = 10.**(gmag / -2.5)
            sed = np.array([gflux, rflux]) / np.hypot(gflux, rflux)
    
            #print('color', color, 'sed:', sed)
            # sqrt(chi2_thresh) ~= 4.55
            #sns = g_sigma * sed
            #sns = np.sqrt(chi2_thresh) * sed
            #sns = 4.4 * sed # 48%
            sns = 4.44 * sed # 50%
            #sns = 4.45 * sed # 50.5?
            #sns = 4.5 * sed # 52.5
    
            noisy = sns[np.newaxis,:] + starnoise
            det1 = (chisq_detection_raw(noisy) > chi2_thresh)
            det2 = (chisq_detection_pos(noisy) > chi2_pos_thresh)
            flux = noisy * noise_level[np.newaxis, :]
            det3 = (sed_union_detection(flux, noise_level, model_seds) > sed_union_th)
            det4 = (sed_mixture_detection(flux, noise_level, model_seds) > sed_mixture_th)
            rates.append([100. * np.sum(d) / n_star for d in [det1,det2,det3,det4]])
    
            if icolor == 21:
                print('SED:', sed)
                print('S/N:', sns)
    
                plt.clf()
                for idet,det in enumerate([det1,det2,det3,det4]):
                    plt.subplot(2,2,idet+1)
                    plt.plot(noisy[det,0], noisy[det,1], 'b.', alpha=0.2)
                    plt.plot(noisy[~det,0], noisy[~det,1], 'r.', alpha=0.2)
                plt.savefig('9b.png')
            
        rates = np.array(rates)
    
        det_names = labels #['Chi2 (raw)', 'Chi2 (pos)', 'SED union', 'SED mixture']
    
        plt.clf()
        for i,rate in enumerate(rates.T):
            plt.plot(colors, rate,
                     color=xcolors[i], linestyle=linestyles[i],
                     linewidth=linewidths[i], alpha=alphas[i])
        plt.xlabel('Star $g - r$ color (mag)')
        plt.ylabel('Detection rate (\\%)')
        plt.legend(det_names)
        plt.title('Detection rate for stars of different colors')
        plt.axvline(0., linestyle='--', alpha=0.2, color='k')
        plt.axvline(1., linestyle=':', alpha=0.2, color='k')
        plt.xlim(colors[0], colors[-1])
        name = 'chisq-color'
        plt.savefig('%s.png' % (name))
        plt.savefig('%s.pdf' % (name))

    sys.exit(0)



    



    
    
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

if __name__ == '__main__':
    # default to making all plots
    plots = None
    import sys
    args = sys.argv[1:]
    # if any are specified on the command line, only make those
    if len(args):
        plots = args #[int(a) for a in args]
    main(plots=plots)
