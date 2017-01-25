# Analysis-of-Galaxy-Clustering-Using-Halo-Occupation-Distribution

I have been using the "halomod" python package for investigating the clustering properties of our Bootest LAEs.  

=================================
1. HOW to install
=================================
Python Modules (Python 2.x) for Halo Occupation Models from Dr. Steven Murray (http://hmf.icrar.org/)

== Install : 
$ conda install numpy scipy matplotlib astropy ipython numba
$ pip install cached_property
$ CAMBURL=http://camb.info/CAMB_Mar13.tar.gz pip install git+git://github.com/steven-murray/pycamb.git        #need "wget"
$ pip install git+git://github.com/steven-murray/hmf.git@develop
$ pip install git+git://github.com/steven-murray/halomod.git@develop

== check : pip list |grep halomod (or hmf, pycamb)

== editable dev-install for "halomod" from a local dir (setup.py dir)

[shong@beethoven:~/work/pywork/halomodels/halomod-develop]$ pip install -e .


===================================
2. I have put my all-in-one python script, as an example to how to use the 'halomod' package. 
For the best fit parameters, this script also utilizes the Bayesian MCMC sampler, 'emcee' too. 

