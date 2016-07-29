from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import pylab as plt

import scipy.stats

from astrometry.util.plotutils import *
from astrometry.util.fits import *


if __name__ == '__main__':
    ps = PlotSequence('bayes', suffixes=['png','pdf'])
    #ps = PlotSequence('bayes')

    from pgm import PGM, Node

    from matplotlib import rc
    rc('font',**{'family':'serif','size':24})
    
    pgm = PGM()
    pgm.add_node(Node('S', 'S', 0., 0.))
    pgm.add_node(Node('F', 'F', 2., 0.))

    pgm.add_node(Node('f_1', '$f_1$', -1., -2., observed=True))
    pgm.add_node(Node('f_2', '$f_2$',  1., -2., observed=True))
    pgm.add_node(Node('f_3', '$f_3$',  3., -2., observed=True))

    pgm.add_edge('S', 'f_1')
    pgm.add_edge('S', 'f_2')
    pgm.add_edge('S', 'f_3')
    pgm.add_edge('F', 'f_1')
    pgm.add_edge('F', 'f_2')
    pgm.add_edge('F', 'f_3')
    
    ax = pgm.render(fx=6, fy=4)
    ax.axis([-2,4, -3,1])
    ax.figure.savefig('pgm.pdf')

    #plt.figure(figsize=(6,4))
    rc('font',**{'family':'serif','size':12})
    plt.figure(figsize=(6,4))

    plt.subplots_adjust(left=0.1, right=0.95,
                        bottom=0.15, top=0.95)
    
    f1 = 5.
    f2 = 10.
    sig1 = 1.
    sig2 = 1.

    # p(S1) = U(0, 1)

    # p(F | f1,f2) \propto integral_0^1 \
    #                    N(f1 - F S_1, sig1^2) N(f2 - F (1 - S_1), sig2^2) \
    #                    f(F) dS1

    S1 = np.linspace(0, 1, 100)
    pS1 = np.ones_like(S1)
    #for S1 in SS1:

    iv = S1**2 / sig1**2 + (1 - S1)**2 / sig2**2
    muF = (
        ((f1 * S1/sig1**2) + (f2 * (1 - S1) / sig2**2)) /
        iv)
    varF = 1 / iv

    snF = muF / np.sqrt(varF)
    
    plt.clf()
    plt.plot(S1, muF, 'b-', label='mean(F)')
    plt.plot(S1, muF + np.sqrt(varF), 'b-', alpha=0.3)
    plt.plot(S1, muF - np.sqrt(varF), 'b-', alpha=0.3)
    #plt.plot(S1, muF / np.sqrt(varF), 'k--', label='S/N(F)')
    plt.plot(S1, snF, 'k-', alpha=0.5, lw=3, label='S/N(F)')
    plt.plot(S1, varF, 'r--', label='var(F)')
    plt.xlabel('S1')
    plt.legend(loc='upper right')

    plt.axvline(1/3., color='k', alpha=0.5)
    plt.axhline(f1, color='k', alpha=0.5)
    plt.axhline(f2, color='k', alpha=0.5)
    plt.axhline(np.hypot(f1,f2), color='k', alpha=0.5)
    
    ps.savefig()

    F1vals = np.linspace(0, 20, 101)
    #SS1,FF1 = np.meshgrid(S1, F1vals)
    #pF1 = np.zeros_like(SS1)
    pF1 = np.zeros((len(F1vals),len(S1)))
    for i,F1 in enumerate(F1vals):
        pF1[i,:] = np.exp(-(muF - F1)**2 / (2.*varF))

    if False:
        plt.clf()
        plt.imshow(pF1, interpolation='nearest', origin='lower', vmin=0,
                   extent=[S1.min(), S1.max(), F1vals.min(), F1vals.max()],
                   aspect='auto', cmap='hot')
        plt.xlabel('SED S')
        plt.ylabel('Flux F')
        plt.title('p(F)')
        ps.savefig()

    pF = np.sum(pF1, axis=1)

    alpha = 2.
    bet = 2.
    beta_prior = scipy.stats.beta.pdf(S1, alpha, bet)
    pFb = np.sum(pF1 * beta_prior[np.newaxis,:], axis=1)

    #imode = np.argmax(pFb)
    #icredup = np.argmin(np.cumsum(pFb[imode

    plt.clf()
    plt.plot(F1vals, pF, 'b-', label='Flat prior on $S$')
    plt.plot(F1vals, pFb, 'k-', alpha=0.5, lw=3, label='Beta prior on $S$')
    plt.axvline(f1+f2, color='k', alpha=0.2)

    plt.xlabel('Flux $F$')
    plt.ylabel('Posterior $p(F|\{f_i\})$')
    plt.legend(loc='upper left')
    plt.yticks([])
    ps.savefig()

    print('Mean of flux posterior (flat S prior):', np.sum(pF * F1vals) / np.sum(pF))
    print('Mean of flux posterior (beta S prior):', np.sum(pFb * F1vals) / np.sum(pFb))
    
    print('Mean S/N (flat prior):', np.sum(snF) / np.sum(np.ones_like(snF)))
    print('Mean S/N (Beta prior):', np.sum(snF * beta_prior) / np.sum(beta_prior))
    

    f1 = 5.
    f2 = 10.
    f3 = 10.
    sig1 = 1.
    sig2 = 1.
    sig3 = 1.

    S1 = np.linspace(0, 1, 100)
    S2 = np.linspace(0, 1, 101)
    #S3 = 1. - (S1 + S2)
    #pS1 = np.ones((len(S2), len(S1)))

    # pS = 1. * (S1[np.newaxis,:] + S2[:,np.newaxis] < 1.)
    # print('S1', len(S1))
    # print('S2', len(S2))
    # print('pS', pS.shape)

    SS1,SS2 = np.meshgrid(S1, S2)
    SS3 = 1. - (SS1 + SS2)
    pS = 1. * (SS1 + SS2 <= 1.)
    
    iv = SS1**2 / sig1**2 + SS2**2 / sig2**2 + SS3**2 / sig3**2
    muF = (
        ((f1 * SS1/sig1**2) + (f2 * SS2/sig2**2) + (f3 * SS3 / sig3**2)) /
        iv)
    varF = 1 / iv

    plt.clf()
    plt.imshow(muF * pS, interpolation='nearest', origin='lower',
               extent=[0,1,0,1], aspect='auto', cmap='hot')
    plt.title('mu(F)')
    plt.xlabel('S1')
    plt.ylabel('S2')
    ps.savefig()
    
    plt.clf()
    plt.imshow(muF / np.sqrt(varF) *pS, interpolation='nearest', origin='lower',
               extent=[0,1,0,1], aspect='auto', cmap='hot')
    plt.title('S/N(F)')
    plt.xlabel('S1')
    plt.ylabel('S2')

    plt.axvline(0.25, color=(0.5,0.5,1.))
    plt.axhline(0.5, color=(0.5,0.5,1.))
    
    ps.savefig()
    plt.clf()

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    I = (pS == 1)
    snF = muF / np.sqrt(varF)
    ax.scatter(SS1[I], SS2[I], SS3[I], c=snF[I], linewidths=0)
    ax.view_init(45, 20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('S1')
    ax.set_ylabel('S2')
    ax.set_zlabel('S3')
    ps.savefig()

    print('Max S/N:', snF[I].max())
    print('Expected S/N:', np.mean(snF[I]))

    SEDs = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 1., 1.],
        [1., 2.5, 2.5**2]])
    SEDs /= np.sum(SEDs, axis=1)[:,np.newaxis]

    s1 = SEDs[:,0]
    s2 = SEDs[:,1]
    s3 = SEDs[:,2]
    
    iv = s1**2 / sig1**2 + s2**2 / sig2**2 + s3**2 / sig3**2
    muF = (
        ((f1 * s1/sig1**2) + (f2 * s2/sig2**2) + (f3 * s3 / sig3**2)) /
        iv)
    varF = 1 / iv

    snF = muF / np.sqrt(varF)
    print('S/Nes for point SEDs:', snF)
    print('Max      SN:', snF.max())
    print('Expected SN:', np.mean(snF))
    
    #T = fits_table('sweep-000m010-010m005.fits')
    T = fits_table('sweep-240p005-250p010-cut.fits')
    print(len(T), 'sources')
    T.cut((T.decam_nobs[:,1] > 0) * (T.decam_nobs[:,2] > 0) *
          (T.decam_nobs[:,4] > 0))
    print(len(T), 'with all Nobs > 0')
    print('Nobs',
          np.unique(T.decam_nobs[:,1]),
          np.unique(T.decam_nobs[:,2]),
          np.unique(T.decam_nobs[:,4]))
          
    T.s1 = np.maximum(0, T.decam_flux[:,1])
    T.s2 = np.maximum(0, T.decam_flux[:,2])
    T.s3 = np.maximum(0, T.decam_flux[:,4])
    ss = T.s1 + T.s2 + T.s3
    T.s1 /= ss
    T.s2 /= ss
    T.s3 /= ss

    plt.clf()
    loghist(T.s1, T.s2, nbins=200, range=((0,1),(0,1)))
    #plt.plot(SEDs[:,0], SEDs[:,1], 'go')
    plt.xlabel('S1 (g)')
    plt.ylabel('S2 (r)')
    ps.savefig()

    T.gmag = -2.5 * (np.log10(T.decam_flux[:,1]) - 9)
    T.rmag = -2.5 * (np.log10(T.decam_flux[:,2]) - 9)
    T.zmag = -2.5 * (np.log10(T.decam_flux[:,4]) - 9)
    T.grzmag = -2.5 * (np.log10(T.decam_flux[:,1] +
                                T.decam_flux[:,2] +
                                T.decam_flux[:,4]) - 9)
    
    # plt.clf()
    # loghist(T.gmag - T.rmag, T.rmag - T.zmag, nbins=100,
    #         range=((-2,2),(-2,2)))
    # plt.xlabel('g - r (mag)')
    # plt.ylabel('r - z (mag)')
    # ps.savefig()

    plt.clf()
    loghist(T.zmag - T.gmag, T.zmag - T.rmag, nbins=100,
            range=((-5,5),(-5,5)))
    plt.xlabel('z - g (mag)')
    plt.ylabel('z - r (mag)')
    ps.savefig()

    
    plt.clf()
    loghist(T.grzmag - T.gmag, T.grzmag - T.rmag, nbins=100,
            range=((-5,5),(-5,5)))
    plt.xlabel('grz - g (mag)')
    plt.ylabel('grz - r (mag)')
    ps.savefig()

    
