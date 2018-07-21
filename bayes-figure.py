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

plt.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.98)

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
#dextent = [-5,10,-5,10]
#dextent = [-5,11,-5,11]
dextent = [-5.5,11,-5.5,11]
dgvals = np.linspace(dextent[0], dextent[1], 320)
drvals = np.linspace(dextent[2], dextent[3], 320)
d_j = np.array(np.meshgrid(dgvals, drvals))
# d_j: shape (2, N, N)
# Plotting axes for some plots
ax1 = [-2,10,-2,10]

def get_pratio(d_j, sig_j, sed_i, alpha = 1.):
    '''
    Get the probability ratio (fg/bg) for given data points and SED.

    alpha: exponential prior on total flux
    '''
    a_i = alpha - np.sum(d_j * sed_i[:,np.newaxis,np.newaxis] / sig_j[:,np.newaxis,np.newaxis]**2, axis=0)
    b_i = 0.5 * np.sum(sed_i**2 / sig_j**2)
    beta_i = 2 * np.sqrt(b_i)
    c_i = a_i / beta_i
    pratio_i = alpha * np.sqrt(np.pi) / beta_i * np.exp(c_i**2) * (1. - erf(c_i))
    return pratio_i


# Background probability
p_bg = 1./np.prod(np.sqrt(2.*np.pi)*sig_j) * np.exp(-0.5 * np.sum((d_j / sig_j[:,np.newaxis,np.newaxis])**2, axis=0))

pratio_red   = get_pratio(d_j, sig_j, sed_red)
pratio_ronly = get_pratio(d_j, sig_j, sed_ronly)
pratio_flat  = get_pratio(d_j, sig_j, sed_flat)

pratio_a = pratio_red
p_fg_a = p_bg * pratio_a

# 5-sigma (one-sided) Gaussian false positive rate, for comparison
# falsepos = norm.sf(5.)
# print(falsepos)
# print(falsepos * 4e3*4e3, 'false positives per 4k x 4k image')

def contour_plot(p_bg, p_fg, seds,
                 style1=dict(linestyles='-', alpha=0.3, linewidths=3, colors='k'),
                 style2=dict(linestyles='-', colors='b'),
                 label1='Background (noise) model',
                 label2='Foreground (source) model'):
    levs = np.arange(-6, 0)
    c1 = plt.contour(np.log10(p_bg), levels=levs, extent=dextent, **style1)
    c2 = plt.contour(np.log10(p_fg), levels=levs, extent=dextent, **style2)
    
    plt.xlabel('g-band detection map S/N')
    plt.ylabel('r-band detection map S/N')
    plt.axhline(0, color='k', alpha=0.5)
    plt.axvline(0, color='k', alpha=0.5)
    xx = np.array([0,100])
    for sed,c in seds:
        plt.plot(xx * sed[0], xx * sed[1], '-', color=c, alpha=0.1)
    plt.axis('square')
    plt.legend([c1.collections[0], c2.collections[0]],
               [label1, label2],
               loc='lower right')

def rel_contour_plot(pratio, seds):
    levs = np.arange(0, 11)
    plt.contour(np.log10(pratio), levels=levs, linestyles='-', extent=dextent, colors='k')
    plt.xlabel('g-band detection map S/N')
    plt.ylabel('r-band detection map S/N')
    plt.axhline(0, color='k', alpha=0.5)
    plt.axvline(0, color='k', alpha=0.5)
    xx = np.array([0,100])
    for sed,c in seds:
        plt.plot(xx * sed[0], xx * sed[1], '-', color=c, alpha=0.1)
    plt.axis('square');

plt.clf()
contour_plot(p_bg, p_fg_a, [(sed_red, 'r')])
axa = [-5.5,11, -5.5,11]
plt.axis(axa)
plt.savefig('prob-contours-a.pdf')

plt.clf()
rel_contour_plot(pratio_a, [(sed_red, 'r')])
plt.axis(axa)
plt.savefig('prob-rel-a.pdf')

pratio_b = 0.49 * pratio_red + 0.49 * pratio_flat + 0.02 * pratio_ronly
p_fg_b = p_bg * pratio_b

plt.clf()
plotseds = [(sed_red, 'r'), (sed_flat, 'b'), (sed_ronly, 'm')]
contour_plot(p_bg, p_fg_b, plotseds)
plt.axis(axa)
plt.savefig('prob-contours-b.pdf')

plt.clf()
rel_contour_plot(pratio_b, plotseds)
plt.axis(axa)
plt.savefig('prob-rel-b.pdf')

######

#sed_red2 = np.array([1., 2.5])
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
             label1='Foreground model',
             label2='Foreground model, s * 2')
axa = [-5.5,11, -5.5,11]
plt.axis(axa)
plt.savefig('prob-contours-c.pdf')


sed_red3 = sed_red
pratio_red3 = get_pratio(d_j, sig_j, sed_red3, alpha=0.5)
pratio_d = pratio_red3
p_fg_d = p_bg * pratio_d
plt.clf()
contour_plot(p_fg_a, p_fg_c, [(sed_red2, 'r')],
             style1=dict(colors='b', linestyles='-'),
             style2=dict(colors='r', linestyles='--'),
             label1='Foreground model',
             label2='Foreground model, s * 2')
axa = [-5.5,11, -5.5,11]
plt.axis(axa)
plt.savefig('prob-contours-d.pdf')


sed_1a = np.array([1.])
sed_1b = np.array([2.])

d_one = np.linspace(-10, +30, 500)
sig_one = np.array([1.])
p_bg_one = 1./np.prod(np.sqrt(2.*np.pi)*sig_one) * np.exp(-0.5 * (d_one / sig_one)**2)

pratio_1a = get_pratio(d_one, sig_one, sed_1a)
p_fg_1a = p_bg_one * pratio_1a
pratio_1b = get_pratio(d_one, sig_one, sed_1b)
p_fg_1b = p_bg_one * pratio_1b

#sed_1c = np.array([1.])
#pratio_1c = get_pratio(d_one, sig_one, sed_1c, alpha=0.5)
#p_fg_1c = p_bg_one * pratio_1c

# plt.clf()
# plt.plot(d_one, p_bg_one, 'k-', lw=3, alpha=0.3, label='Background model')
# plt.plot(d_one, p_fg_1a[0,:], 'b-', label='Foreground model')
# #plt.plot(d_one, np.exp(-d_one), 'b--')
# plt.plot(d_one, p_fg_1b[0,:], 'r--', label='Foreground model, s * 2')
# #plt.plot(d_one, np.exp(-d_one/4.), 'r--')
# #plt.ylim(0, p_bg_one.max())
# plt.axvline(0., color='k', alpha=0.1)
# plt.axhline(0., color='k', alpha=0.1)
# plt.xlim(-4,12)
# plt.legend()
# plt.yticks(np.arange(0, 0.41, 0.1))
# plt.xlabel('Detection map values')
# plt.ylabel('Probability')
# plt.savefig('prob-1a.pdf')
# 
# flux_1a = d_one / sed_1a
# flux_1b = d_one / sed_1b
# ia = np.argmax(p_fg_1a)
# ib = np.argmax(p_fg_1b)
# plt.clf()
# plt.plot(flux_1a, p_fg_1a[0,:], 'b-', label='Foreground model')
# plt.plot(flux_1b, p_fg_1b[0,:] * sed_1b, 'r--', label='Foreground model, s * 2')
# plt.axvline(flux_1a[ia], color='b', alpha=0.1)
# plt.axvline(flux_1b[ib], color='r', alpha=0.1)
# plt.axvline(0., color='k', alpha=0.1)
# plt.axhline(0., color='k', alpha=0.1)
# plt.legend()
# plt.xlabel('Flux (implied by detection map)')
# plt.ylabel('Probability')
# plt.xlim(-4,8)
# plt.savefig('prob-1b.pdf')


flux_1a = d_one / sed_1a
flux_1b = d_one / sed_1b

prior1a = np.exp(-flux_1a) * (flux_1a > 0)
prior1b = np.exp(-flux_1b) * (flux_1b > 0)

plt.clf()
plt.subplots_adjust(hspace=0)
ax1 = plt.subplot2grid((3,1), (0, 0), rowspan=2)
plt.plot(d_one, p_bg_one, 'k-', lw=3, alpha=0.3, label='Background model')
plt.plot(d_one, p_fg_1a[0,:], 'b-', label='Foreground model')
plt.plot(d_one, p_fg_1b[0,:], 'r--', label='Foreground model, s * 2')
#plt.plot(d_one, p_fg_1c[0,:], 'g:', label='Foreground model, alpha = 0.5')
plt.axvline(0., color='k', alpha=0.1)
plt.axhline(0., color='k', alpha=0.1)
plt.xlim(-4,12)
plt.yticks(np.arange(0, 0.41, 0.1))
plt.legend()
plt.ylabel('Probability')
plt.xticks([])

ax2 = plt.subplot2grid((3,1), (2, 0))
plt.plot(d_one, prior1a, 'b-', label='Foreground prior')
# /2 to normalize (~ d_d / d_flux)
plt.plot(d_one, prior1b/2., 'r--', label='Foreground prior, s * 2')
plt.axhline(0., color='k', alpha=0.1)
plt.xlim(-4,12)
plt.yticks([0, 0.5, 1.0])
plt.xlabel('Detection map values')
plt.legend();
plt.savefig('prob-1d.pdf')
