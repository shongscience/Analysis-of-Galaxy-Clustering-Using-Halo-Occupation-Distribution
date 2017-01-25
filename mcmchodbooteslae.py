# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:30:15 2016

Info : A script to wrap up the hod fits for our Bootest LAEs 

1. plot the observed n(z) distribution

2. plot the observed clustering 

3. plot a guess-HOD function 

4. Generate RandomPairFunction for the inverse IC correction 

5. Now writing chi-square and likelihood function, to find the best fit parameter 

6. Finally generate a MCMC sample near the best fit parameter, to generate Bayesian posteriors



@author: shong
"""

import matplotlib.pyplot as plt
import numpy as np

import scipy.integrate as intg
from scipy.integrate import simps
from scipy.integrate import romberg
from scipy.interpolate import interp1d
from halomod.integrate_corr import AngularCF, angular_corr_gal, flat_z_dist, dxdz
from hmf.cosmo import Planck15
from mpmath import gamma as Gamma
import scipy.optimize as op


""" Phase 1 ============================
import zlae.nz ; this is a smoothed observed z distribution 
"""
obsz, obsnz = np.loadtxt("zlae.nz",unpack=True)

#print obsz
#print obsnz

obsnzfunc = interp1d(obsz,obsnz) # interpolated function for AngularCF
integ = romberg(obsnzfunc,2.5,2.9) # normalize the n(z)
obsnz = obsnz/integ 
obsnzfunc = interp1d(obsz,obsnz) # normalized interpolated function

testz = np.linspace(2.5,3.0,num=100,endpoint=True)
print romberg(obsnzfunc,2.5,2.9) # this should be "1" now


#plot the filter selections 
plt.rc('font', family='serif') 
plt.rc('font', serif='Times New Roman') 
plt.rcParams.update({'font.size': 18})

fig = plt.figure()
plt.plot(obsz,obsnz,'o',testz,obsnzfunc(testz),'-')
plt.axis([2.5,3.0,0,15])
#plt.axis('equal')
#plt.axes().set_aspect(1.0)
plt.xlabel(r'$z$')
plt.ylabel(r'$n_{obs}(z)$')
plt.show()


""" Phase 2 ============================ 
observed w(theta)
"""
logang, wang = np.loadtxt("plaelsoutcut.data",skiprows=2 ,usecols =(0,4), unpack=True)

ang = np.power(10.0,logang)

print logang
print ang
print wang

### Bootstrap error
wangerr = np.loadtxt("plaebooterror.data",skiprows=1 ,usecols = (2,), unpack=True)
print wangerr

print len(ang),len(wang),len(wangerr)

fig = plt.figure(figsize=(10,5))
#plt.plot(logang,wang,'s')
plt.errorbar(ang,wang,yerr=wangerr,fmt='o')
plt.axis([1,1000,0.001,50])
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\theta$ (arcsec)')
plt.ylabel(r'$\omega_{LS}(\theta)$')
plt.scatter(ang[:14],wang[:14],marker='x',color='r',s=100)
plt.show()



""" Phase 3 ========================
Exploring some HOD models for guessing a good staring search position. 

redshift selection function
try flatz
"""

#####
# To choose theta, set it in arcsec then convert
theta_min = 1.0 * np.pi/(180.0 * 3600.0)
theta_max = 10000.0 * np.pi/(180.0 * 3600.0)



#####
## interpolated obsz 
## Geach12

acf = AngularCF(hmf_model="Tinker08",bias_model="Tinker10",hod_model="Geach12",
                hod_params={"M_1":np.log10(1.0e+11),"alpha":0.5,"fca":0.8,
                "fs":0.8,"fcb":0.00},
                ng=0.0025,Mmin=8,z=2.66,
                p1=obsnzfunc,zmin=2.55,zmax=2.8,logu_min=-5,logu_max=2.5,
                rnum = 100,theta_min=theta_min,theta_max=theta_max,p_of_z=True)


plt.axis([1,1000,0.001,50])

plone = plt.errorbar(ang,wang,yerr=wangerr,fmt='o',label='Observed Clustering')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\theta$ (arcsec)')
plt.ylabel(r'$\omega_{LS}(\theta)$')

pltwo = plt.scatter(acf.theta * 3600.0 * 180.0/np.pi,acf.angular_corr_gal,
                    color='red', marker='+', label='Geach12')

plt.legend(handles=[plone,pltwo],loc=3)

plt.show()


#print acf.n_eff
print "hod model = ",acf.hod_model
#print acf.n_gal
print "mean_gal_den = ",acf.mean_gal_den
print "hod parameter = ",acf.hod_params
print "cosmo model = ",acf.cosmo_model



""" Phase 4 ========================
Integral Constraints for HOD models 

Random Points and their pairs from the Observing Mask

the angular scale = log10 arcsecs
"""

############ Random points distribution : Observing Mask
#load random points from the observing mask
randra, randdec  = np.loadtxt("short1000bootesrandom.dat",unpack=True)

#plt.axis([1,1000,0.001,50])

#plt.yscale('log')
#plt.xscale('log')
plt.xlabel(r'R.A.(deg)')
plt.ylabel(r'Decl. (deg)')
plt.scatter(randra[:500],randdec[:500],marker='.')
plt.show()





###########
# get acf.bins in the Log10 arcsec scale


print acf.theta * 3600.0 * 180.0/np.pi
print np.log10(acf.theta * 3600.0 * 180.0/np.pi)
logtheta = np.log10(acf.theta * 3600.0 * 180.0/np.pi)


#plt.axis([1,10000,0.000001,50])
plt.axis([0,4,0.000001,50])
#plone = plt.errorbar(ang,wang,yerr=wangerr,fmt='o',label='Observed Clustering')
plt.yscale('log')
plone = plt.errorbar(logang,wang,yerr=wangerr,fmt='o',label='Observed Clustering')
#plt.xscale('log')
plt.xlabel(r'Log$\theta$ (arcsec)')
plt.ylabel(r'$\omega_{LS}(\theta)$')
pltwo = plt.scatter(logtheta,acf.angular_corr_gal,
                    color='r', marker='+', label='Zehavi05')
plt.show()


###########
# load randompairs in the log10 arcsec scale for "log bin"
rpairdist  = np.log10(np.loadtxt("rpair1000.edge",usecols=(2,),unpack=True))
totalrpair = np.double(len(rpairdist))


rhist, rhistbin = np.histogram(rpairdist,bins=logtheta)

print rpairdist.mean(),rpairdist.max(),rpairdist.min()
print totalrpair,rhist.sum()



# show all variables for sanity checks
print rhist
print rhistbin #histogram edges, number of edges = number of hist "+1"
print logtheta
print acf.angular_corr_gal
print rhist.sum()

print len(rhist),len(logtheta),len(acf.angular_corr_gal)
print np.sum(rhist*acf.angular_corr_gal[:len(acf.angular_corr_gal)-1])
print np.sum(rhist*acf.angular_corr_gal[1:])
print np.mean([np.sum(rhist*acf.angular_corr_gal[:len(acf.angular_corr_gal)-1]),
               np.sum(rhist*acf.angular_corr_gal[1:])])


# rough IC
iccorr = np.mean([np.sum(rhist*acf.angular_corr_gal[:len(acf.angular_corr_gal)-1]),
               np.sum(rhist*acf.angular_corr_gal[1:])])/totalrpair
print iccorr


angiccorr = np.copy((acf.angular_corr_gal - iccorr)/(1.0+iccorr))

plt.axis([0,4,0.000001,50])
#plone = plt.errorbar(ang,wang,yerr=wangerr,fmt='o',label='Observed Clustering')
plt.yscale('log')
#plt.xscale('log')
plt.xlabel(r'log $\theta$ (arcsec)')
plt.ylabel(r'$\omega_{LS}(\theta)$')
pltwo = plt.scatter(logtheta,acf.angular_corr_gal,
                    color='r', marker='+', label='Zehavi05')

plthree = plt.scatter(logtheta,angiccorr,
                    color='b', marker='+', label='Zehavi05')
plt.show()



""" Phase 5 ===================================
1. Chi-squares and likelihood function 
2. find a rough minimum-chisquare (maximum likelihood) spot for the next MCMC run. 
"""

# 1. generate a basic AngularCF 
acf = AngularCF(hmf_model="Tinker08",bias_model="Tinker10",hod_model="Geach12",
                hod_params={"M_1":np.log10(1.0e+12),"alpha":0.5,"fca":0.8,
                "fs":0.8,"fcb":0.00},
                ng=0.0025,Mmin=8,z=2.66,
                p1=obsnzfunc,zmin=2.55,zmax=2.8,logu_min=-5,logu_max=2.5,
                rnum = 100,theta_min=theta_min,theta_max=theta_max,p_of_z=True)
# 2. implemeting the inverse IC (the above, phase 4) into likelihood function

#scipy.optimize uses "x","*args"; parameter vector + args tuples 
#   "x" -> x[0],x[1], etc : the model parameters
#   "*args" -> x,y,yerr, etc : any data combined with "x" for a final return value

import scipy.special as sp # for "sp.erf"

def lnlike(paras, x, y, yerr, hodacf, rhist, rhistbin,debug=False,writeeps=False):
    #read parameters
    logm1, alpha, fca, fcb, fs, sig_logm = np.double(paras)
    #sig_logm = 0.26 # others are defaults
    delta=1.0
    xfrac=1.0
    if debug:
        print "paras = ",paras
        print "logm1 alpha fca fcb fs sig_logm = ",logm1,alpha,fca,fcb,fs,sig_logm
    
    # if the paras are "insane", return minus infinity for the likelihood
    if alpha < 0.0 or alpha > 3.0 or logm1 < 6 or logm1 > 15 or np.isnan(alpha) or np.isnan(logm1) or fca < 0.0 or fca > 0.5 or fs < 0.0 or fs > 1.0 or fcb < 0.0 or fcb > 1.0 or sig_logm < 0.0 or sig_logm > 3.0 or np.isnan(fca) or np.isnan(fcb) or np.isnan(fs) or np.isnan(sig_logm):
        if debug:
            print "paras = ",paras
            print hodacf.hod_model,hodacf.hod_params
            print "len(hodacf.theta) = ",len(hodacf.theta)
            print "lnlike : a negive inf is returned"
        return -np.inf
    
    #calculate hod from the input paras
    try:    
        hodacf.update(hod_params={"M_1":logm1,"alpha":alpha,"fca":fca,"fcb":fcb,"fs":fs,"sig_logm":sig_logm,"delta":delta,"x":xfrac},ng=0.0025)
    except:
        print "Exception : issue found during matching density. Returns a negative infinity likelyhood." 
        return -np.inf

    if debug:
        print hodacf.hod_model,hodacf.hod_params
        print "len(hodacf.theta) = ",len(hodacf.theta)

    #Now, get some IC using randompairs, rhist and rhistbin
    interpacf = interp1d(np.log10(hodacf.theta * 3600.0 * 180.0/np.pi),hodacf.angular_corr_gal)
    wangrhistbin = interpacf(rhistbin)
    if debug:
        print "lnlike : interpolating randompair functions"

    #crude average of left and right-edges sum
    ic = np.mean([np.sum(rhist*wangrhistbin[:len(wangrhistbin)-1]),
                  np.sum(rhist*wangrhistbin[1:])])/np.double(rhist.sum()) #crude, left sum + right sum
                  
    #Now, get model values with inver IC
    ymodel = interpacf(x)
    ymodel = (ymodel - ic)/(1.0+ic) # inverse correction of IC
    
    reval = -0.5*(np.sum(((y - ymodel)/yerr)**2))
    #debug : plot the ic effect
    if debug:
        print hodacf.hod_model,hodacf.hod_params
        print "acf.hod_params.values() = ",hodacf.hod_params.values()
        print "IC = ",ic
        print "lnlike = ",reval
        print "acf.ng = ",hodacf.ng
        print "acf.z = ",hodacf.z
        print str("acf.m[0] = %g \n"%acf.m[0])
        
        #plot angular correlation and hod
        plt.figure(figsize=(10,4))
        plt.subplot(121)
        plt.axis([0,3,0.001,50])
        plt.yscale('log')
        plt.xlabel(r'log $\theta$ (arcsec)')
        plt.ylabel(r'$\omega_{LS}(\theta)$')
        plt.errorbar(x,y,yerr=yerr,fmt='o',color='b')
        plt.plot(np.log10(hodacf.theta * 3600.0 * 180.0/np.pi),hodacf.angular_corr_gal,
                    'r--')
        plt.plot(np.log10(hodacf.theta * 3600.0 * 180.0/np.pi),
                (hodacf.angular_corr_gal-ic)/(1.0+ic),'r-')
        plt.plot(x,ymodel,'gx',markersize=14)        
        #hod
        plt.subplot(122)
        logmmin = np.double(hodacf.hod_params['M_min']) # in the Geach12, 8th(hence, 7) keyparam is the Mmin
        xlogm = np.linspace(9,15,num=50)
        geachhodcen = fcb*(1.0 - fca)*np.exp(-1.0*(xlogm - logmmin)**2/(2.0*sig_logm**2)) + fca*(1.0 + sp.erf((xlogm-logmmin)/sig_logm))
        geachhodsat = fs*(1.0 + sp.erf((xlogm-logm1)/delta))*(10**xlogm / 10 ** logm1)**alpha
        plt.axis([9,15,0.001,100])
        plt.yscale('log')
        plt.xlabel(r'log $M_h (h^{-1}M_\odot)$')
        plt.ylabel(r'$<N_g>$')
        plt.plot(xlogm,geachhodcen,'b-')
        plt.plot(xlogm,geachhodsat,'b--')
        plt.plot(xlogm,geachhodsat+geachhodcen,'r-')
        
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
                
        plt.show()
        if writeeps:
            plt.savefig("temphod.eps")
    return reval
# testing .. call some lnlike_s 
#logm1, alpha, fca, fcb, fs, sig_logm 
#dummy = lnlike([11.0,0.8,0.5,0.8,0.1, 0.5],logang[:14],wang[:14],wangerr[:14],acf,rhist,rhistbin,debug=True)


""" Phase 5-5 ##################################################
# find maximum likelihood using scipy.optimize
"""
#acf.update(Mmin=8,z=2.66)
#nll = lambda *args: -lnlike(*args)
#result = op.minimize(nll, [13.126,0.739,0.00291,0.929,0.985,0.0965],method='Nelder-Mead',
#                     args=(logang[:14],wang[:14],wangerr[:14],acf,rhist,rhistbin))
##################### NEW : Got the result [12.9555492864,0.924511261957,0.00284121093717,0.949734642878,0.943569448523,0.214739807923]
# old bugged minimum
#dummy = lnlike([13.1256887522,0.739391492272,0.00290919073314,0.929586997362,0.984839010247,0.096483155068],logang[:14],wang[:14],wangerr[:14],acf,rhist,rhistbin,debug=True)
dummy = lnlike([13.126,0.739,0.00291,0.929,0.985,0.0965],logang[:14],wang[:14],wangerr[:14],acf,rhist,rhistbin,debug=True)



""" Phase 6 =================================================
1. prior function and MCMC run; then get the posteriors 
"""

def lnprior(paras):
    logm1, alpha, fca, fcb, fs, sig_logm = np.double(paras)
    if 9.0 < logm1 < 14.0 and 0.0 < alpha < 3.0 and 0.0 <= fca <= 0.5 and 0.0 <= fcb <= 1.0 and 0.0 <= fs <= 1.0 and 0.0 < sig_logm < 2.0:
        return 0.0
    return -np.inf

def lnprob(paras, x, y, yerr, hodacf, rhist, rhistbin):
    lp = lnprior(paras)
    ll = lnlike(paras, x, y, yerr, hodacf, rhist, rhistbin)
    if not np.isfinite(lp):
        return -np.inf
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll






#"""
## MCMC

ndim, nwalkers = 6, 120
pos0 = [[13.126,0.739,0.00291,0.929,0.985,0.0965]+ 1e-2*np.random.randn(ndim) for i in range(nwalkers)]


#import multiprocessing
from pathos.multiprocessing import Pool
import pickle
import emcee




sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                args=[logang[:14],wang[:14],wangerr[:14],acf,rhist,rhistbin],pool = Pool(4))
print "The MCMC Run has started ..."
#print "pos0 = ",pos0
print "Initial 10 steps ..."
posnow, probnow, statenow = sampler.run_mcmc(pos0, 10) 
print "10 steps are done ..."
for i in range(79):
    #print "posnow: ",posnow
    postmp, probnow, statenow = sampler.run_mcmc(posnow, 10)
    posnow = postmp
    print "%f percent is done.." % (100.0*np.double(10.0+i*10.0)/800.0)
    samplechain = np.copy(sampler.chain)
    with open('mcmcGeach.pickle','wb') as f:
        pickle.dump([samplechain,posnow,ndim,nwalkers],f)
    f.close() #keep dumping the current results to overwrite the pickle. 
print "The MCMC Run is done."
#"""



"""
Run this sequence when MCMC die during the run ..

posnow!! is the important variable to resume!!
"""
ndim, nwalkers = 6, 120

import pickle
import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                args=(logang[:14],wang[:14],wangerr[:14],acf,rhist,rhistbin))

with open('mcmcGeach550.pickle') as f:
    samplechain,posnow,ndim,nwalkers = pickle.load(f)


print "Resume the MCMC Run ..."

print "Initial 10 steps ..."
postmp, probnow, statenow = sampler.run_mcmc(posnow, 10) 
posnow = postmp
print "10 steps are done ..."
for i in range(39):
    #print "posnow: ",posnow
    postmp, probnow, statenow = sampler.run_mcmc(posnow, 10)
    posnow = postmp
    print "%f percent is done.." % (100.0*np.double(10.0+i*10.0)/400.0)
    samplechain = np.copy(sampler.chain)
    with open('mcmcGeach.pickle','wb') as f:
        pickle.dump([samplechain,posnow,ndim,nwalkers],f)
    f.close() #keep dumping the current results to overwrite the pickle. 
print "The MCMC Run is done."
#"""








#restore from pickle 
import pickle

"""
with open('mcmcGeach.pickle') as f:
    samplechain,posnow,ndim,nwalkers = pickle.load(f)


print len(samplechain)
print samplechain.shape
#nwalks = 800
nwalks = np.int(samplechain.shape[1])

"""

with open('mcmcGeach550.pickle') as f:
    samplechain,posnow,ndim,nwalkers = pickle.load(f)


print len(samplechain)
print samplechain.shape
#nwalks = 800
nwalks = np.int(samplechain.shape[1])



with open('mcmcGeach+400.pickle') as fa:
    samplechaina,posnowa,ndima,nwalkersa = pickle.load(fa)

print len(samplechaina)
print samplechaina.shape

nwalksa = 370


#print xwalk


fig = plt.figure(figsize=(6,30))

xwalk = np.arange(nwalks)
plt.subplot(611)
plt.axis([0,nwalks,9,14])
#plt.yscale('log')
#plt.xscale('log')
for iwalker in range(0, nwalkers):
    plt.plot(xwalk[0:],samplechain[iwalker,0:,0],color='grey')
#plt.xlabel(r'walks')
plt.ylabel(r'$M_1$')



plt.subplot(612)
plt.axis([0,nwalks,0,2])
#plt.axis([0,10,-10,10])
#plt.yscale('log')
#plt.xscale('log')
for iwalker in range(0, nwalkers):
    plt.plot(xwalk[0:],samplechain[iwalker,0:,1],color='grey')
#plt.xlabel(r'walks')
plt.ylabel(r'$\alpha$')


plt.subplot(613)
plt.axis([0,nwalks,0,1])
#plt.axis([0,10,-10,10])
#plt.yscale('log')
#plt.xscale('log')
for iwalker in range(0, nwalkers):
    plt.plot(xwalk[0:],samplechain[iwalker,0:,2],color='grey')
#plt.xlabel(r'walks')
plt.ylabel(r'fca')

plt.subplot(614)
plt.axis([0,nwalks,0,1.5])
#plt.axis([0,10,-10,10])
#plt.yscale('log')
#plt.xscale('log')
for iwalker in range(0, nwalkers):
    plt.plot(xwalk[0:],samplechain[iwalker,0:,3],color='grey')
#plt.xlabel(r'walks')
plt.ylabel(r'fcb')

plt.subplot(615)
plt.axis([0,nwalks,0,1.5])
#plt.axis([0,10,-10,10])
#plt.yscale('log')
#plt.xscale('log')
for iwalker in range(0, nwalkers):
    plt.plot(xwalk[0:],samplechain[iwalker,0:,4],color='grey')
#plt.xlabel(r'walks')
plt.ylabel(r'fs')


plt.subplot(616)
plt.axis([0,nwalks,0,1.5])
#plt.axis([0,10,-10,10])
#plt.yscale('log')
#plt.xscale('log')
for iwalker in range(0, nwalkers):
    plt.plot(xwalk[0:],samplechain[iwalker,0:,5],color='grey')
#plt.xlabel(r'walks')
plt.ylabel(r'$\sigma_{\logM}$')



plt.show()






# collapse the chain outputs
samplesb = samplechain[:, 450:, :].reshape((-1, ndim))
samplesa = samplechaina[:, 0:, :].reshape((-1, ndim))
samples = np.concatenate((np.double(samplesb),np.double(samplesa)),axis=0)

#total 950 walks, removing 450 walks as burnins 

#samples = samplechain[:, 400:, :].reshape((-1, ndim))



import corner
fig = corner.corner(samples, labels=[r'$\log M_1$',r'$\alpha$',r'$F_c^A$',r'$F_c^B$',r'$F_s$',r'$\sigma_{\log M}$'], 
                                     truths=[13.126,0.739,0.00291,0.929,0.985,0.0965])
fig.savefig("mcmcGeach.eps")


""" plot each mcmc fracture
fig = corner.corner(samplesa, labels=[r'$\log M_1$',r'$\alpha$',r'$F_c^A$',r'$F_c^B$',r'$F_s$',r'$\sigma_{\log M}$'], 
                                     truths=[13.126,0.739,0.00291,0.929,0.985,0.0965])
fig = corner.corner(samplesb, labels=[r'$\log M_1$',r'$\alpha$',r'$F_c^A$',r'$F_c^B$',r'$F_s$',r'$\sigma_{\log M}$'], 
                                     truths=[13.126,0.739,0.00291,0.929,0.985,0.0965])
"""




m1mc, alphamc, fcamc, fcbmc, fsmc, sigmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                    zip(*np.percentile(samples, [16, 50, 84],axis=0)))

print m1mc, alphamc, fcamc, fcbmc, fsmc, sigmc

with open('mcmcParameters.pickle','wb') as f:
    pickle.dump([m1mc, alphamc, fcamc, fcbmc, fsmc, sigmc],f)
f.close() #keep dumping the current results to overwrite the pickle. 


fig = corner.corner(samples, labels=[r'$\log M_1 (h^{-1} M_{\odot})$',r'$\alpha$',r'$F_c^A$',r'$F_c^B$',r'$F_s$',r'$\sigma_{\log M}$'], 
                                     truths=[12.97, 0.79, 0.11, 0.35,0.84, 0.63],truth_color='b',
                                     alts=[13.126,0.739,0.00291,0.929,0.985,0.0965],
                                     alt_color='g',linewidth=10,truth_lw=1,alt_lw=3,truth_ms=10,alt_ms=10)


fig.savefig("mcmcGeachMCMC.eps")

print m1mc, alphamc, fcamc, fcbmc, fsmc, sigmc




#dummy = lnlike([m1mc[0], alphamc[0], fcamc[0], fcbmc[0], fsmc[0],sigmc[0]],logang[:14],wang[:14],wangerr[:14],acf,rhist,rhistbin,debug=True)
#dummy = lnlike([m1mc[0], alphamc[0], fcamc[0], fcbmc[0]+fcbmc[1], fsmc[0],sigmc[0]],logang[:14],wang[:14],wangerr[:14],acf,rhist,rhistbin,debug=True)
#dummy = lnlike([m1mc[0], alphamc[0], fcamc[0], fcbmc[0]-fcbmc[2], fsmc[0],sigmc[0]],logang[:14],wang[:14],wangerr[:14],acf,rhist,rhistbin,debug=True)


dummy = lnlike([12.97, 0.79, 0.11, 0.35,0.84, 0.63],logang[:14],wang[:14],wangerr[:14],acf,rhist,rhistbin,debug=True)

dummy = lnlike([13.126,0.739,0.00291,0.929,0.985,0.0965],logang[:14],wang[:14],wangerr[:14],acf,rhist,rhistbin,debug=True)




fig = plt.figure(figsize=(6,5))
#plot angular correlation and hod
plt.figure(figsize=(10,8))

delta=1.0
xfrac=1.0

## MODEL 1
logm1, alpha, fca, fcb, fs, sig_logm = np.double([12.96, 0.92, 0.0028,0.95, 0.9, 0.21])
dummy = lnlike([12.96, 0.92, 0.0028,0.95, 0.9, 0.21],logang[:14],wang[:14],wangerr[:14],acf,rhist,rhistbin,debug=False)

x=logang[:14]
y=wang[:14]
yerr=wangerr[:14]
acf.update(hod_params={"M_1":logm1,"alpha":alpha,"fca":fca,"fcb":fcb,"fs":fs,"sig_logm":sig_logm,"delta":delta,"x":xfrac})
#Now, get some IC using randompairs, rhist and rhistbin
interpacf = interp1d(np.log10(acf.theta * 3600.0 * 180.0/np.pi),acf.angular_corr_gal)
wangrhistbin = interpacf(rhistbin)

#crude average of left and right-edges sum
ic = np.mean([np.sum(rhist*wangrhistbin[:len(wangrhistbin)-1]),
              np.sum(rhist*wangrhistbin[1:])])/np.double(rhist.sum()) #crude, left sum + right sum
                  
#Now, get model values with inver IC
ymodel = interpacf(x)
ymodel = (ymodel - ic)/(1.0+ic) # inverse correction of IC
    
reval = -0.5*(np.sum(((y - ymodel)/yerr)**2))

print "ic chi ",ic, (-2.0*reval)

plt.subplot(221)
plt.axis([0,3,0.001,50])
plt.yscale('log')
plt.xlabel(r'log $\theta$ (arcsec)')
plt.ylabel(r'$\omega_{LS}(\theta)$')
plt.errorbar(x,y,yerr=yerr,fmt='o',color='b')
plt.plot(np.log10(acf.theta * 3600.0 * 180.0/np.pi),acf.angular_corr_gal,'r--')
plt.plot(np.log10(acf.theta * 3600.0 * 180.0/np.pi),(acf.angular_corr_gal-ic)/(1.0+ic),'r-')
plt.plot(x,ymodel,'gx',markersize=14)        
#hod
plt.subplot(222)
logmmin = acf.hod_params.values()[7] # in the Geach12, 8th(hence, 7) keyparam is the Mmin
xlogm = np.linspace(9,15,num=50)
geachhodcen = fcb*(1.0 - fca)*np.exp(-1.0*(xlogm - logmmin)**2/(2.0*sig_logm**2)) + fca*(1.0 + sp.erf((xlogm-logmmin)/sig_logm))
geachhodsat = fs*(1.0 + sp.erf((xlogm-logm1)/delta))*(10**xlogm / 10 ** logm1)**alpha
plt.axis([9,15,0.001,100])
plt.yscale('log')
plt.xlabel(r'log $M_h (h^{-1}M_\odot)$')
plt.ylabel(r'$<N_g>$')
plt.plot(xlogm,geachhodcen,'b-')
plt.plot(xlogm,geachhodsat,'b--')
plt.plot(xlogm,geachhodsat+geachhodcen,'r-')


## MODEL 2
logm1, alpha, fca, fcb, fs, sig_logm = np.double([13.32, 0.48, 0.051, 0.18, 0.8, 0.22])
dummy = lnlike([13.32, 0.48, 0.051, 0.18, 0.8, 0.22],logang[:14],wang[:14],wangerr[:14],acf,rhist,rhistbin,debug=False)

acf.update(hod_params={"M_1":logm1,"alpha":alpha,"fca":fca,"fcb":fcb,"fs":fs,"sig_logm":sig_logm,"delta":delta,"x":xfrac})
#Now, get some IC using randompairs, rhist and rhistbin
interpacf = interp1d(np.log10(acf.theta * 3600.0 * 180.0/np.pi),acf.angular_corr_gal)
wangrhistbin = interpacf(rhistbin)

#crude average of left and right-edges sum
ic = np.mean([np.sum(rhist*wangrhistbin[:len(wangrhistbin)-1]),
              np.sum(rhist*wangrhistbin[1:])])/np.double(rhist.sum()) #crude, left sum + right sum
                  
#Now, get model values with inver IC
ymodel = interpacf(x)
ymodel = (ymodel - ic)/(1.0+ic) # inverse correction of IC
    
reval = -0.5*(np.sum(((y - ymodel)/yerr)**2))

print "ic chi ",ic, (-2.0*reval)

plt.subplot(223)
plt.axis([0,3,0.001,50])
plt.yscale('log')
plt.xlabel(r'log $\theta$ (arcsec)')
plt.ylabel(r'$\omega_{LS}(\theta)$')
plt.errorbar(x,y,yerr=yerr,fmt='o',color='b')
plt.plot(np.log10(acf.theta * 3600.0 * 180.0/np.pi),acf.angular_corr_gal,'r--')
plt.plot(np.log10(acf.theta * 3600.0 * 180.0/np.pi),(acf.angular_corr_gal-ic)/(1.0+ic),'r-')
plt.plot(x,ymodel,'gx',markersize=14)        
#hod
plt.subplot(224)
logmmin = acf.hod_params.values()[7] # in the Geach12, 8th(hence, 7) keyparam is the Mmin
xlogm = np.linspace(9,15,num=50)
geachhodcen = fcb*(1.0 - fca)*np.exp(-1.0*(xlogm - logmmin)**2/(2.0*sig_logm**2)) + fca*(1.0 + sp.erf((xlogm-logmmin)/sig_logm))
geachhodsat = fs*(1.0 + sp.erf((xlogm-logm1)/delta))*(10**xlogm / 10 ** logm1)**alpha
plt.axis([9,15,0.001,100])
plt.yscale('log')
plt.xlabel(r'log $M_h (h^{-1}M_\odot)$')
plt.ylabel(r'$<N_g>$')
plt.plot(xlogm,geachhodcen,'b-')
plt.plot(xlogm,geachhodsat,'b--')
plt.plot(xlogm,geachhodsat+geachhodcen,'r-')



plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

plt.show()


print acf.angular_corr_gal
print acf.angular_corr_matter


dummy = lnlike([12.96, 0.92, 0.0028,0.95, 0.9, 0.21],logang[:14],wang[:14],wangerr[:14],acf,rhist,rhistbin,debug=True)
print acf.corr_mm
dummy = lnlike([13.32, 0.48, 0.051, 0.18, 0.8, 0.22],logang[:14],wang[:14],wangerr[:14],acf,rhist,rhistbin,debug=True)
print acf.corr_mm




"""
Check 3d mocks from orsi 



x1, y1, z1, mh1, cenflag1 = np.loadtxt("./orsi/geach12_model1.cube",skiprows=1 ,usecols =(0,1,2,6,7), unpack=True)
print len(x1)
print np.max(x1)
print np.max(y1)
print np.max(z1)
with open('model1cube.pickle','wb') as f:
    pickle.dump([x1, y1, z1, mh1, cenflag1],f)

x2, y2, z2, mh2, cenflag2 = np.loadtxt("./orsi/geach12_model2.cube",skiprows=1 ,usecols =(0,1,2,6,7), unpack=True)
print len(x2)
print np.max(x2)
print np.max(y2)
print np.max(z2)
with open('model2cube.pickle','wb') as f:
    pickle.dump([x2, y2, z2, mh2, cenflag2],f)


with open('model1cube.pickle') as f:
    x1, y1, z1, mh1, cenflag1 = pickle.load(f)

with open('model2cube.pickle') as f:
    x2, y2, z2, mh2, cenflag2 = pickle.load(f)

ibox1 = np.where((100 < x1) & (x1 < 200) & (100 < y1) & (y1 < 200) & (200 < z1) & (z1 < 300))
ibox2 = np.where((100 < x2) & (x2 < 200) & (100 < y2) & (y2 < 200) & (200 < z2) & (z2 < 300))

print len(x1[ibox1])
print len(x2[ibox2])

ng1 = str(len(x1[ibox1]))
ng2 = str(len(x2[ibox2]))

## plot
plt.figure(figsize=(16,7))
plt.subplot(121)
plt.axis([100,200,100,200])
plt.title("Model #1 (Ng = "+ng1+")")
plt.xlabel(r'x ($h^{-1}$ Mpc)')
plt.ylabel(r'y ($h^{-1}$ Mpc)')
plt.scatter(x1[ibox1],y1[ibox1],marker='.')

plt.subplot(122)
plt.axis([100,200,100,200])
plt.title("Model #2 (Ng = "+ng2+")")
plt.xlabel(r'x ($h^{-1}$ Mpc)')
plt.ylabel(r'y ($h^{-1}$ Mpc)')
plt.scatter(x2[ibox2],y2[ibox2],marker='.')

print len(x1[ibox1])
print len(x2[ibox2])


"""