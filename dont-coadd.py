import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
import pylab as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter, correlate, correlate1d

plt.figure(figsize=(5,4))
plt.subplots_adjust(right=0.95, top=0.98)

W,H = 25,25
sig1 = 1.
sig2 = 2.
psfsig1 = 2.
psfsig2 = 1.02
# To make total S/N = 100
flux = 100. * 100./19.06/1.0364

cx = W//2
cy = H//2
xx,yy = np.meshgrid(np.arange(W), np.arange(H))
image1 = flux/(2.*np.pi*psfsig1**2) * np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2.*psfsig1**2))
image2 = flux/(2.*np.pi*psfsig2**2) * np.exp(-((xx-cx)**2 + (yy-cy)**2) / (2.*psfsig2**2))

psfnorm1 = 1./(2.*np.sqrt(np.pi)*psfsig1)
psfnorm2 = 1./(2.*np.sqrt(np.pi)*psfsig2)
detmap1 = gaussian_filter(image1, psfsig1) / (psfnorm1**2)
detmap2 = gaussian_filter(image2, psfsig2) / (psfnorm2**2)
detsig1 = sig1 / psfnorm1
detsig2 = sig2 / psfnorm2
detmap = (detmap1 * (1./detsig1**2) + detmap2 * (1./detsig2**2)) / (1./detsig1**2 + 1./detsig2**2)
detsig = np.sqrt(1./(1./detsig1**2 + 1./detsig2**2))

# detmap1.max()/detsig1, detmap2.max()/detsig2, detmap.max()/detsig

alphas = np.linspace(0, 1, 101)
codetsn = np.zeros(len(alphas), np.float32)

psfimg1 = 1./(2.*np.pi*psfsig1**2) * np.exp(-0.5 * ((xx-cx)**2 + (yy-cy)**2) / psfsig1**2)
psfimg2 = 1./(2.*np.pi*psfsig2**2) * np.exp(-0.5 * ((xx-cx)**2 + (yy-cy)**2) / psfsig2**2)
norm1 = np.sqrt(np.sum(psfimg1**2))
norm2 = np.sqrt(np.sum(psfimg2**2))

for ii,alpha in enumerate(alphas):
    beta = 1.-alpha
    coadd = alpha * image1 + beta * image2
    cosig = np.sqrt((alpha * sig1)**2 + (beta * sig2)**2)
    copsf = alpha * psfimg1 + beta * psfimg2
    conorm = np.sqrt(np.sum(copsf**2))
    codet = correlate(coadd, copsf) / conorm**2
    codetsig = cosig / conorm
    codetsn[ii] = codet.max() / codetsig

plt.clf()
plt.axhline(detmap.max()/detsig, color='k', linestyle='--',
            label='2-image detection map')
plt.plot(alphas, codetsn, 'b-',
         label='Coadd 2 images then detect')
plt.axhline(detmap1.max()/detsig1, color='r', linestyle=':',
            label='Single-image detection maps')
plt.axhline(detmap2.max()/detsig2, color='r', linestyle=':')
plt.legend(loc=(0.02, 0.75))
plt.xlim(0,1)
plt.xlabel('Coadd weight')
plt.ylabel('Detection S/N');
plt.savefig('dont-coadd.pdf')
