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

sed_red_g = 1./3.5
sed_red_r = 2.5/3.5
#sed_red_g = 1.
#sed_red_r = 2.5
sed_red = np.array([sed_red_g, sed_red_r])

sed_ronly = np.array([0., 1.])

sed_flat_g = 0.5
sed_flat_r = 0.5
sed_flat = np.array([sed_flat_g, sed_flat_r])

# Grid of detection-map values
#dextent = [-5,10,-5,10]
#dextent = [-5,11,-5,11]
dextent = [-5.5,11,-5.5,11]
dgvals = np.linspace(dextent[0], dextent[1], 160)
drvals = np.linspace(dextent[2], dextent[3], 160)
d_j = np.array(np.meshgrid(dgvals, drvals))
# d_j: shape (2, N, N)
# Plotting axes for some plots
ax1 = [-2,10,-2,10]

def get_pratio(d_j, sig_j, sed_i, alpha = 2.):
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

def contour_plot(p_bg, p_fg, seds):
    levs = np.arange(-6, 0)
    c1 = plt.contour(np.log10(p_bg), levels=levs, linestyles='-', alpha=0.3, linewidths=3, extent=dextent, colors='k')
    c2 = plt.contour(np.log10(p_fg), levels=levs, linestyles='-', extent=dextent, colors='b')
    plt.xlabel('g-band detection map S/N')
    plt.ylabel('r-band detection map S/N')
    plt.axhline(0, color='k', alpha=0.5)
    plt.axvline(0, color='k', alpha=0.5)
    xx = np.array([0,100])
    for sed,c in seds:
        plt.plot(xx * sed[0], xx * sed[1], '-', color=c, alpha=0.1)
    plt.axis('square')
    plt.legend([c1.collections[0], c2.collections[0]],
               ['Background (noise) model', 'Foreground (source) model'],
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

sys.exit(0)

levs = np.arange(-6, 0)
c1 = plt.contour(np.log10(p_bg), levels=levs, linestyles='-', alpha=0.3, linewidths=3, extent=dextent, colors='k')
c2 = plt.contour(np.log10(p_fg_b), levels=levs, linestyles='-', extent=dextent, colors='b')
plt.xlabel('g-band detection map S/N')
plt.ylabel('r-band detection map S/N')
plt.axhline(0, color='k', alpha=0.5)
plt.axvline(0, color='k', alpha=0.5)
xx = np.array([0,100]);
plt.plot(xx * sed_red[0], xx * sed_red[1], 'r-', alpha=0.1);
#plt.axis(ax);
plt.axis('square');
#plt.axis(dextent);
axa = [-5.5,11, -5.5,11]
plt.axis(axa);
plt.legend([c1.collections[0], c2.collections[0]],
           ['Background (noise) model', 'Foreground (source) model'],
    loc='lower right')
#plt.title('Likelihood contours for single-SED prior')
plt.savefig('prob-contours-a.pdf')

levs = np.arange(0, 11)
plt.clf()
plt.contour(np.log10(pratio_a), levels=levs, linestyles='-', extent=dextent, colors='k')
plt.xlabel('g-band detection map S/N')
plt.ylabel('r-band detection map S/N');
plt.axhline(0, color='k', alpha=0.5)
plt.axvline(0, color='k', alpha=0.5);
xx = np.array([0,100]);
plt.plot(xx * sed_red[0], xx * sed_red[1], 'r-', alpha=0.1);
#plt.plot(xx * sed_red[0], xx * sed_red[1], 'r-');
#plt.plot(xx * sed_3[0], xx * sed_3[1], 'm-');
plt.axis('square');
plt.axis(axa);
#plt.title('Likelihood ratio conours for single-SED prior')
plt.savefig('prob-rel-a.pdf')


sys.exit(0)

# In[180]:



# In[181]:

a_2 = alpha - np.sum(d_j * sed_red[:,np.newaxis,np.newaxis] / sig_j[:,np.newaxis,np.newaxis]**2, axis=0)
b_2 = 0.5 * np.sum(sed_red**2 / sig_j**2)
beta_2 = 2 * np.sqrt(b_2)
c_2 = a_2 / beta_2
pratio_2 = np.sqrt(np.pi) / beta_2 * np.exp(c_2**2) * (1. - erf(c_2))


# In[182]:

pratio_b = pratio_1 * 0.5 + pratio_2 * 0.5


# In[183]:

plt.imshow(np.log10(pratio_2), interpolation='nearest', origin='lower', extent=dextent, vmin=-1, vmax=9)
plt.colorbar();
plt.contour(np.log10(pratio_2), levels=[-np.log10(falsepos)], colors=['k'], extent=dextent);


# In[184]:

plt.imshow(np.log10(pratio_b), interpolation='nearest', origin='lower', extent=dextent, vmin=-1, vmax=9)
plt.colorbar();
plt.contour(np.log10(pratio_b), levels=[-np.log10(falsepos)], colors=['k'], extent=dextent);
sc = 5. / np.sqrt(np.sum(sed_flat**2))
plt.plot(sc * sed_flat[0], sc * sed_flat[1], 'ko');
sc = 5. / np.sqrt(np.sum(sed_red**2))
plt.plot(sc * sed_red[0], sc * sed_red[1], 'ro');
plt.xlabel('g-band detection map')
plt.ylabel('r-band detection map');
plt.axhline(0, color='k', alpha=0.5)
plt.axvline(0, color='k', alpha=0.5);


# In[229]:

pratio_c = pratio_1 * 0.2 + pratio_2 * 0.8
#plt.imshow(np.log10(pratio), interpolation='nearest', origin='lower', extent=dextent, vmin=-1, vmax=9)
#plt.colorbar();
c0 = plt.contour(np.log10(pratio_a), levels=[-np.log10(falsepos)], colors=['k'], linestyles=['--'], extent=dextent)
c1 = plt.contour(np.log10(pratio_b), levels=[-np.log10(falsepos)], colors=['k'], extent=dextent)
c2 = plt.contour(np.log10(pratio_c), levels=[-np.log10(falsepos)], colors=['b'], extent=dextent);
#sc = 5. / np.sqrt(np.sum(sed_flat**2))
#plt.plot(sc * sed_flat[0], sc * sed_flat[1], 'ko');
#sc = 5. / np.sqrt(np.sum(sed_red**2))
#plt.plot(sc * sed_red[0], sc * sed_red[1], 'ro');
xx = np.array([0,100]);
plt.plot(xx * sed_flat[0], xx * sed_flat[1], 'b-', alpha=0.1);
plt.plot(xx * sed_red[0], xx * sed_red[1], 'r-', alpha=0.1);

plt.xlabel('g-band detection map S/N')
plt.ylabel('r-band detection map S/N');
plt.axhline(0, color='k', alpha=0.5)
plt.axvline(0, color='k', alpha=0.5);
plt.axis('square');
plt.axis(ax1);
plt.legend([c0.collections[0], c1.collections[0], c2.collections[0]],
           ['100% flat SED', '50% flat + 50% red SED', '20% flat + 80% red SED'], loc='upper right')
plt.title('Likelihood-ratio contours for two-SED model');
plt.savefig('prob-rel-b.png')


# In[186]:

sed_3_g = 0.
sed_3_r = 1.
sed_3 = np.array([sed_3_g, sed_3_r])


# In[187]:

a_3 = alpha - np.sum(d_j * sed_3[:,np.newaxis,np.newaxis] / sig_j[:,np.newaxis,np.newaxis]**2, axis=0)
b_3 = 0.5 * np.sum(sed_3**2 / sig_j**2)
beta_3 = 2 * np.sqrt(b_3)
c_3 = a_3 / beta_3
pratio_3 = np.sqrt(np.pi) / beta_3 * np.exp(c_3**2) * (1. - erf(c_3))


# In[231]:

pratio_d = pratio_1 * 0.49 + pratio_2 * 0.49 + pratio_3 * 0.02


# In[232]:

plt.imshow(np.log10(pratio_d), interpolation='nearest', origin='lower', extent=dextent, vmin=-1, vmax=9)
plt.colorbar();
plt.axhline(0, color='k', alpha=0.5)
plt.axvline(0, color='k', alpha=0.5);
#plt.contour(np.log10(pratio_d), levels=[-np.log10(falsepos)], colors=['k'], extent=dextent);
plt.contour(pratio_d, levels=[1./falsepos], colors=['k'], extent=dextent);
sc = 5. / np.sqrt(np.sum(sed_flat**2))
plt.plot(sc * sed_flat[0], sc * sed_flat[1], 'ko');
sc = 5. / np.sqrt(np.sum(sed_red**2))
plt.plot(sc * sed_red[0], sc * sed_red[1], 'ro');
sc = 5. / np.sqrt(np.sum(sed_3**2))
plt.plot(sc * sed_3[0], sc * sed_3[1], 'mo');
plt.xlabel('g-band detection map')
plt.ylabel('r-band detection map');


# In[233]:

np.sum(pratio_d)


# In[234]:

p_fg_d = p_bg * pratio_d


# In[235]:

plt.imshow(p_bg, extent=dextent, interpolation='nearest', origin='lower');
plt.colorbar();


# In[236]:

plt.imshow(p_fg_d, extent=dextent, interpolation='nearest', origin='lower');
plt.colorbar();


# In[237]:

p_fg_a = p_bg * pratio_a
plt.imshow(p_fg_a, extent=dextent, interpolation='nearest', origin='lower');
plt.colorbar();


# In[238]:

plt.imshow(np.log10(p_fg_a), extent=dextent, interpolation='nearest', origin='lower');
plt.colorbar();


# In[242]:

levs = np.arange(-6, 0)
plt.contour(np.log10(p_bg), levels=levs, linestyles='-', extent=dextent, colors='k',
           label='Background model')
plt.contour(np.log10(p_fg_d), levels=levs, linestyles='-', extent=dextent, colors='r',
           label='Foreground model (3-SED)');
plt.xlabel('g-band detection map S/N')
plt.ylabel('r-band detection map S/N');
plt.axhline(0, color='k', alpha=0.5)
plt.axvline(0, color='k', alpha=0.5);
ax = plt.axis()
xx = np.array([0,100]);
plt.plot(xx * sed_flat[0], xx * sed_flat[1], 'b-', alpha=0.1);
plt.plot(xx * sed_red[0], xx * sed_red[1], 'r-', alpha=0.1);
plt.plot(xx * sed_3[0], xx * sed_3[1], 'm-', alpha=0.1);
plt.axis(ax);
plt.axis('square');
plt.axis(ax);
plt.title('Likelihood contours for 3-SED model');
plt.savefig('prob-countours-d.png')


# In[205]:

print(np.sum(p_bg), np.sum(p_fg_d))
(dgvals[1]-dgvals[0]) * (drvals[1]-drvals[0]) * np.sum(p_fg_d)


# In[244]:

levs = np.arange(0, 11)
plt.contour(np.log10(pratio_d), levels=levs, linestyles='-', extent=dextent, colors='k')
plt.xlabel('g-band detection map S/N')
plt.ylabel('r-band detection map S/N');
plt.axhline(0, color='k', alpha=0.5)
plt.axvline(0, color='k', alpha=0.5);
ax = plt.axis()
xx = np.array([0,100]);
plt.plot(xx * sed_flat[0], xx * sed_flat[1], 'b-', alpha=0.1);
plt.plot(xx * sed_red[0], xx * sed_red[1], 'r-', alpha=0.1);
plt.plot(xx * sed_3[0], xx * sed_3[1], 'm-', alpha=0.1);
plt.axis(ax);
plt.axis('square');
plt.axis(ax);
plt.title('Relative likelihood contours for 3-SED model')
plt.savefig('prob-rel-d.png')


# In[113]:

d_j_random = np.random.normal(size=(2,4000,4000))


# In[132]:

pratio_r1 = get_pratio(d_j_random, sig_j, sed_flat)
pratio_r2 = get_pratio(d_j_random, sig_j, sed_red)
pratio_r3 = get_pratio(d_j_random, sig_j, sed_3)
pratio_r_d = pratio_r1 * 0.45 + pratio_r2 * 0.45 + pratio_r3 * 0.1
p_r_bg = 1./np.prod(np.sqrt(2.*np.pi)*sig_j) * np.exp(-0.5 * (d_j_random / sig_j[:,np.newaxis,np.newaxis])**2)
p_r_fg = p_r_bg * pratio_r_d
pratio_r_d.shape


# In[139]:

plt.hist(pratio_r_d.ravel(), bins=100, range=(0,30000), log=True);


# In[246]:

plt.hist(np.log10(pratio_r_d.ravel()), bins=100, range=(0,5), log=True);


# In[247]:

1./falsepos


# In[248]:

levs = np.arange(0, 11)
plt.contour(np.log10(pratio_d), levels=levs, linestyles='-', extent=dextent, colors='k')
plt.xlabel('g-band detection map S/N')
plt.ylabel('r-band detection map S/N');
plt.axhline(0, color='k', alpha=0.5)
plt.axvline(0, color='k', alpha=0.5);
ax = plt.axis()
xx = np.array([0,100]);
plt.plot(xx * sed_flat[0], xx * sed_flat[1], 'b-', alpha=0.1);
plt.plot(xx * sed_red[0], xx * sed_red[1], 'r-', alpha=0.1);
plt.plot(xx * sed_3[0], xx * sed_3[1], 'm-', alpha=0.1);
plt.axis(ax);
plt.axis('square');
plt.axis(ax);
plt.title('Relative likelihood contours for 3-SED model')

plt.contour(np.log10(p_bg), levels=np.arange(-6, 0), linestyles='-', extent=dextent, colors='r')


# In[250]:

plt.hist(p_r_fg.ravel(), bins=100, log=True);


# In[251]:

p_bg.min()


# In[252]:

norm.isf(p_bg.min())


# In[253]:

logthreshs = np.linspace(0, 10, 100)
lothreshs = logthreshs[:-1]
hithreshs = logthreshs[1:]
nbin = np.zeros_like(lothreshs)
for i,(lo,hi) in enumerate(zip(lothreshs, hithreshs)):
    N = np.sum((pratio_r_d >= 10.**lo) * (pratio_r_d < 10.**hi))
    nbin[i] = N
plt.plot((lothreshs+hithreshs)/2., nbin, 'b-')


# In[256]:

t = (lothreshs+hithreshs)/2.
plt.semilogy(t, nbin, 'b.-')


# In[ ]:



