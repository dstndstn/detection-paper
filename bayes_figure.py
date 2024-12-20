import matplotlib
matplotlib.rcParams['figure.figsize'] = (5,5)
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
import pylab as plt
import numpy as np
from scipy.special import erf
from scipy.stats import norm

'''

2-band example showing behavior of Bayesian SED-matched detection.

We first show the behavior of a single "red" detection filter.

(Next, we look at a two-SED prior and examine how the weights affect
the locus of the threshold.)

Then we look at a three-SED model ("flat", "red", and "r-only").

Finally, we look at an empirical (DECaLS) SED.

'''
def bayes_figures():
    plt.subplots_adjust(left=0.15, right=0.98, bottom=0.1, top=0.98)

    # Assuming unit variance on the detection maps; signal = S/N.
    sig_g = 1.
    sig_r = 1.
    sig_j = np.array([sig_g, sig_r])

    sed_red = np.array([1., 2.5])
    sed_red /= np.sum(sed_red)

    sed_ronly = np.array([0., 1.])

    sed_flat = np.array([1., 1.])
    sed_flat /= np.sum(sed_flat)

    # Grid of detection-map values
    dextent = [-5.5,11,-5.5,11]
    dgvals = np.linspace(dextent[0], dextent[1], 320)
    drvals = np.linspace(dextent[2], dextent[3], 320)
    # d_j: shape (2, N, N)
    # detection maps per band
    d_j = np.array(np.meshgrid(dgvals, drvals))

    # Plotting axes for some plots
    ax1 = [-2,10,-2,10]

    # Background probability (chi-squared)
    p_bg = (1./np.prod(np.sqrt(2.*np.pi)*sig_j) *
            np.exp(-0.5 * np.sum((d_j / sig_j[:,np.newaxis,np.newaxis])**2, axis=0)))

    pratio_red   = get_pratio(d_j, sig_j, sed_red)
    pratio_ronly = get_pratio(d_j, sig_j, sed_ronly)
    pratio_flat  = get_pratio(d_j, sig_j, sed_flat)

    figU = False

    # chi-squared equivalent
    if figU:

        uextent = [-11, 11, -11, 11]
        ugvals = np.linspace(uextent[0], uextent[1], 320)
        urvals = np.linspace(uextent[2], uextent[3], 320)
        ud_j = np.array(np.meshgrid(ugvals, urvals))

        p_bg_u = (1./np.prod(np.sqrt(2.*np.pi)*sig_j) *
                  np.exp(-0.5 * np.sum((ud_j / sig_j[:,np.newaxis,np.newaxis])**2, axis=0)))

        th = np.linspace(0., 2.*np.pi, 36, endpoint=False)
        #th = np.linspace(0., 0.5 * np.pi, 36, endpoint=True)
        #th = th[2:-2]
        seds_uniform = np.vstack((np.cos(th), np.sin(th))).T
        #seds_uniform /= np.sum(np.abs(seds_uniform), axis=1)[:,np.newaxis]
        print('uniform seds:', seds_uniform.shape)

        pratios_u = np.array([get_pratio(ud_j, sig_j, s) for s in seds_uniform])
        print('pratios_u:', pratios_u.shape)

        pratio_u = np.sum(pratios_u * 1./len(th), axis=0)
        p_fg_u = p_bg_u * pratio_u
        
        plt.clf()
        contour_plot(p_bg_u, p_fg_u, [(s, '0.5') for s in seds_uniform],
                     extent=uextent, sedlw=0.5)
        plt.axis(uextent)
        plt.savefig('prob-contours-u.pdf')

        plt.clf()
        rel_contour_plot(pratio_u, [(s, '0.5') for s in seds_uniform],
                         extent=uextent, sedlw=0.5)
        plt.axis(uextent)
        plt.savefig('prob-rel-u.pdf')

        return



    figA = True
    figB = True
    figB2 = True
    figC = True
    figD = False
    figE = True

    if figA:
        # First figure: using one SED-matched filter (red) as an illustration.
        pratio_a = pratio_red
        p_fg_a = p_bg * pratio_a

        # 5-sigma (one-sided) Gaussian false positive rate, for comparison
        # falsepos = norm.sf(5.)
        # print(falsepos)
        # print(falsepos * 4e3*4e3, 'false positives per 4k x 4k image')

        plt.clf()
        contour_plot(p_bg, p_fg_a, [(sed_red, 'r')], extent=dextent)
        axa = [-5.5,11, -5.5,11]
        plt.axis(axa)
        plt.savefig('prob-contours-a.pdf')

        plt.clf()
        rel_contour_plot(pratio_a, [(sed_red, 'r')], extent=dextent)
        plt.axis(axa)
        plt.savefig('prob-rel-a.pdf')

    if figB:
        pratio_b = 0.49 * pratio_red + 0.49 * pratio_flat + 0.02 * pratio_ronly
        p_fg_b = p_bg * pratio_b

        plt.clf()
        plotseds = [(sed_red, 'r'), (sed_flat, 'b'), (sed_ronly, 'm')]
        contour_plot(p_bg, p_fg_b, plotseds, extent=dextent)
        plt.axis(axa)
        plt.savefig('prob-contours-b.pdf')

        plt.clf()
        rel_contour_plot(pratio_b, plotseds, extent=dextent)
        plt.axis(axa)
        plt.savefig('prob-rel-b.pdf')

    if figB2:
        alpha = 2.
        sed_red2   = alpha * sed_red
        sed_ronly2 = alpha * sed_ronly
        sed_flat2  = alpha * sed_flat
        pratio_red2   = get_pratio(d_j, sig_j, sed_red2)
        pratio_ronly2 = get_pratio(d_j, sig_j, sed_ronly2)
        pratio_flat2  = get_pratio(d_j, sig_j, sed_flat2)

        pratio_b2 = 0.49 * pratio_red2 + 0.49 * pratio_flat2 + 0.02 * pratio_ronly2
        p_fg_b2 = p_bg * pratio_b2

        plt.clf()
        plotseds2 = [(sed_red2, 'r'), (sed_flat2, 'b'), (sed_ronly2, 'm')]
        contour_plot(p_bg, p_fg_b2, plotseds2, extent=dextent)
        plt.axis(axa)
        plt.savefig('prob-contours-b2.pdf')

        plt.clf()
        rel_contour_plot(pratio_b2, plotseds2, extent=dextent)
        plt.axis(axa)
        plt.savefig('prob-rel-b2.pdf')

    if figC:
        sed_red2 = sed_red * 2

        pratio_red2   = get_pratio(d_j, sig_j, sed_red2)
        pratio_c = pratio_red2
        p_fg_c = p_bg * pratio_c

        # plt.clf()
        # contour_plot(p_bg, p_fg_c, [(sed_red2, 'r')])
        # axa = [-5.5,11, -5.5,11]
        # plt.axis(axa)
        # plt.savefig('prob-contours-c.pdf')

        plt.clf()
        contour_plot(p_fg_a, p_fg_c, [(sed_red2, 'r')],
                     style1=dict(colors='b', linestyles='-'),
                     style2=dict(colors='r', linestyles='--'),
                     label1='Foreground model, faint luminosity function',
                     label2='Foreground model, bright luminosity function',
                     extent=dextent)
        axa = [-5.5,11, -5.5,11]
        plt.axis(axa)
        plt.savefig('prob-contours-c.pdf')

        plotseds = [(sed_red, 'r')]
        plt.clf()
        levs = np.arange(0, 5)
        rel_contour_plot(pratio_c, plotseds, extent=dextent, levs=levs)
        rel_contour_plot(pratio_a, [], extent=dextent, levs=levs, contour_linestyle='--')
        plt.axis(axa)
        plt.savefig('prob-rel-c.pdf')

    if figD:
        sed_red3 = sed_red
        pratio_red3 = get_pratio(d_j, sig_j, sed_red3, alpha=0.5)
        pratio_d = pratio_red3
        p_fg_d = p_bg * pratio_d
        plt.clf()
        contour_plot(p_fg_a, p_fg_c, [(sed_red2, 'r')],
                     style1=dict(colors='b', linestyles='-'),
                     style2=dict(colors='r', linestyles='--'),
                     label1='Foreground model',
                     label2='Foreground model, s * 2',
                     extent=dextent)
        axa = [-5.5,11, -5.5,11]
        plt.axis(axa)
        plt.savefig('prob-contours-d.pdf')

    if figE:
        d_one = np.linspace(-10, +30, 500)
        sed_1a = np.array([1.])
        sed_1b = np.array([2.])

        flux_1a = d_one / sed_1a
        flux_1b = d_one / sed_1b

        prior1a = np.exp(-flux_1a) * (flux_1a > 0)
        prior1b = np.exp(-flux_1b) * (flux_1b > 0)

        sig_one = np.array([1.])
        p_bg_one = 1./np.prod(np.sqrt(2.*np.pi)*sig_one) * np.exp(-0.5 * (d_one / sig_one)**2)

        pratio_1a = get_pratio(d_one, sig_one, sed_1a)
        p_fg_1a = p_bg_one * pratio_1a
        pratio_1b = get_pratio(d_one, sig_one, sed_1b)
        p_fg_1b = p_bg_one * pratio_1b

        plt.clf()
        plt.subplots_adjust(hspace=0.2)

        ax1 = plt.subplot2grid((2,1), (1, 0))
        plt.plot(d_one, p_bg_one, 'k-', lw=3, alpha=0.3, label='Background model')
        plt.plot(d_one, p_fg_1a[0,:], 'b-', label='Faint luminosity function')
        plt.plot(d_one, p_fg_1b[0,:], 'r--', label='Bright luminosity function')
        plt.axvline(0., color='k', alpha=0.1)
        plt.axhline(0., color='k', alpha=0.1)
        plt.xlim(-4,8)
        plt.yticks(np.arange(0, 0.41, 0.2))
        plt.xlabel(r'Observed flux ($\sigma$)')
        plt.legend()
        plt.ylabel('Posterior Probability')
        
        ax2 = plt.subplot2grid((2,1), (0, 0))
        plt.plot(d_one, prior1a, 'b-', label='Faint luminosity function')
        # /2 to normalize (~ d_d / d_flux)
        plt.plot(d_one, prior1b/2., 'r--', label='Bright luminosity function')
        plt.axhline(0., color='k', alpha=0.1)
        plt.axvline(0., color='k', alpha=0.1)
        plt.xlim(-4,8)
        plt.yticks([0, 0.5, 1.0])
        plt.ylabel('Prior probability')
        plt.xlabel('Prior flux (arb. units)')
        plt.xticks([])
        plt.legend()
        plt.savefig('prob-1d.pdf')


def get_pratio(d_j, sig_j, sed_i, alpha = 1.):
    '''
    Get the probability ratio (fg/bg) for given data points and SED.

    alpha: exponential prior on total flux

    d_j: (n_bands, nx, ny)
    sig_j: (n_bands)
    sed_i: (n_bands)
    '''
    a_i = alpha - np.sum(d_j * sed_i[:,np.newaxis,np.newaxis] / sig_j[:,np.newaxis,np.newaxis]**2, axis=0)
    b_i = 0.5 * np.sum(sed_i**2 / sig_j**2)
    beta_i = 2 * np.sqrt(b_i)
    c_i = a_i / beta_i
    pratio_i = alpha * np.sqrt(np.pi) / beta_i * np.exp(c_i**2) * (1. - erf(c_i))
    return pratio_i


def contour_plot(p_bg, p_fg, seds,
                 style1=dict(linestyles='-', alpha=0.3, linewidths=3, colors='k'),
                 style2=dict(linestyles='-', colors='b'),
                 label1='Background (noise) model',
                 label2='Foreground (source) model',
                 extent=None,
                 sedlw=3):
    levs = np.arange(-6, 0)
    #c1 = plt.contour(np.log10(p_bg), levels=levs, extent=extent, **style1)
    c1 = plt.contour(p_bg, levels=10.**np.array(levs), extent=extent, **style1)
    #c1.set_label(label1)
    #c2 = plt.contour(np.log10(p_fg), levels=levs, extent=extent, **style2)
    c2 = plt.contour(p_fg, levels=10.**np.array(levs), extent=extent, **style2)
    #c2.set_label(label2)

    plt.xlabel('g-band detection map S/N')
    plt.ylabel('r-band detection map S/N')
    plt.axhline(0, color='k', alpha=0.2)
    plt.axvline(0, color='k', alpha=0.2)
    xx = np.array([0,100])
    for sed,c in seds:
        plt.plot(xx * sed[0], xx * sed[1], '-', color=c, alpha=0.5, lw=sedlw)
    plt.axis('square')
    ax = plt.axis()

    pp = []
    for s in [style2,style1]:
        s = s.copy()
        ls = s['linestyles']
        del s['linestyles']
        s['linestyle'] = ls
        ls = s['colors']
        del s['colors']
        s['color'] = ls
        if 'linewidths' in s:
            ls = s['linewidths']
            del s['linewidths']
            s['linewidth'] = ls
        p = plt.plot([0,0],[0,0], **s)
        pp.append(p[0])

    plt.axis(ax)
    #plt.legend([c2,c1], [label2,label1])
    plt.legend(pp, [label2,label1])
    #plt.legend()
    #plt.legend([c2.collections[0], c1.collections[0]],
    #           [label2, label1],
    #           loc='lower right')

def rel_contour_plot(pratio, seds, extent=None,
                     sedlw=3, contour_linestyle='-', levs=None):
    if levs is None:
        levs = np.arange(0, 11)
    plt.contour(np.log10(pratio), levels=levs, linestyles=contour_linestyle,
                extent=extent, colors='k')
    plt.xlabel('g-band detection map S/N')
    plt.ylabel('r-band detection map S/N')
    plt.axhline(0, color='k', alpha=0.2)
    plt.axvline(0, color='k', alpha=0.2)
    xx = np.array([0,100])
    for sed,c in seds:
        plt.plot(xx * sed[0], xx * sed[1], '-', color=c, alpha=0.5, lw=sedlw)
    plt.axis('square');



if __name__ == '__main__':
    bayes_figures()
