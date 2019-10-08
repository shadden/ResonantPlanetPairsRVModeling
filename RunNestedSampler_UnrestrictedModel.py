import numpy as np
import radvel
import pickle
import pandas as pd
import sys
import re
import dynesty
DATA = "./data/"
I = int(sys.argv[1])
AllObservations=pd.read_pickle("All_Observations.pkl")

# Get observation data
system = AllObservations.system.unique()[I]
Observations=AllObservations.query('system==@system')

# Get MCMC posterior sample file
dname = re.sub("HD ","HD",system)
dname = re.sub("BD\+20 ","BD+20_",dname)
datadir = DATA+dname
with open(datadir+"/full_model_post.pkl","rb") as fi:
    posterior_samples = pickle.load(fi)

# Add some additional columns to the posterior file
posterior_samples.columns
for i in range(1,3):
    rtx = posterior_samples["secosw{}".format(i)]
    rty = posterior_samples["sesinw{}".format(i)]    
    posterior_samples["e{}".format(i)] = rtx**2+rty**2
    posterior_samples["w{}".format(i)] = np.arctan2(rty,rtx)

# Set up radvel likelihood object
fitbasis = 'per tc e w k'
params = radvel.Parameters(2,basis=fitbasis)
mdl = radvel.RVModel(params,time_base=Observations.time.median())
likes=[]
instruments = Observations.instrument.unique()
for i,instrument in enumerate(instruments):
    data = Observations.query('instrument==@instrument')[['time','velocity','uncertainty']].values
    inst_like = radvel.likelihood.RVLikelihood(mdl,*(data.T),suffix=instrument)
    # ... setting pars to 0 is for some reason 
    # neccessary to stop the code from eating shit...
    for key,par in inst_like.params.items():
        par.value = 0
    likes.append(inst_like)
like = radvel.likelihood.CompositeLikelihood(likes)
like.params['dvdt'].vary = False
like.params['curv'].vary = False


# Rename the posterior columns according to new instrument naming scheme
like_gamma_keys = [ k for k in like.params.keys() if re.search('gamma',k)]
post_gamma_cols = [ k for k in posterior_samples.columns if re.search('gamma',k)]
rename_dict = {x:y for x,y in zip(post_gamma_cols,like_gamma_keys)}
like_jit_keys = [ k for k in like.params.keys() if re.search('jit',k)]
post_jit_cols = [ k for k in posterior_samples.columns if re.search('jit',k)]
rename_dict.update({x:y for x,y in zip(post_jit_cols,like_jit_keys)})

posterior_samples.rename(columns=rename_dict,inplace=True)


# set likelihood values to 
for par in like.list_vary_params():
    if par not in posterior_samples.columns:
        errmsg = "Posterior data is missing parameter {}".format(par)
        errmsg+= "\n(Posterior data columns may simply need to be renamed)"
        raise KeyError(errmsg)
    else:
        like.params[par].value = posterior_samples[par].median()

radvel.maxlike_fitting(like)
if False:
    # Set min and max
    from collections import OrderedDict
    maxdict=OrderedDict({col:np.inf for col in like.list_vary_params()})
    mindict=OrderedDict({col:-np.inf for col in like.list_vary_params()})
    meddict=OrderedDict({col:-np.inf for col in like.list_vary_params()})
    for col in like.list_vary_params():
        x = posterior_samples[col]
        xmin,xmed,xmax = x.min(),x.median(),x.max()
        maxdict[col] = xmed + 2 * (xmax-xmed)
        mindict[col] = xmed + 2 * (xmin-xmed)
        if mindict[col] < 0:
            mindict[col] = 1e-4 * xmed
        meddict[col] = xmed
    for inst in instruments:
        inst_obs = Observations.query('instrument==@inst')
        gammastr="gamma{}".format(inst)
        jitstr="jit{}".format(inst)
        mindict[jitstr] = 1e-3 * inst_obs['uncertainty'].median()
        maxdict[jitstr] = 30   * inst_obs['uncertainty'].median()
        mindict[gammastr] = inst_obs['velocity'].min()
        maxdict[gammastr] = inst_obs['velocity'].max()
    
    # Define prior transform function
    def prior_transform(u,mindict,meddict,maxdict,suffixes):
        ####################
        ### first planet ###
        ####################
        i=0
        per1 = mindict['per1'] * (1-u[i]) + maxdict['per1'] * u[i]  # within range p1min to p2max
    
        i+=1
        tc1 = meddict['tc1'] + (u[i]-0.5) * per1
    
        i+=1
        e1 = u[i] # eccentricity from 0 to 1
    
        i+=1
        w1 = np.mod(2 * np.pi * u[i],2*np.pi) # omega from 0 to 2pi
    
        i+=1
        logk1_min = np.log(mindict['k1'])
        logk1_max = np.log(maxdict['k1'])
        logk1 = logk1_min * (1-u[i]) + logk1_max * u[i] # K log-uniform between Kmin and Kmax
        k1 = np.exp(logk1)
    
        #####################
        ### second planet ###
        #####################
        i+=1
        per2 = mindict['per2']  * (1-u[i]) + maxdict['per2'] * u[i] # within range p2min to p2max
    
        i+=1
        tc2 = meddict['tc2'] + (u[i]-0.5) * per2
    
        i+=1
        e2 = u[i] # eccentricity from 0 to 1
    
        i+=1
        w2 = np.mod(2 * np.pi * u[i],2*np.pi) # omega from 0 to 2pi
    
        i+=1
        logk2_min = np.log(mindict['k2'])
        logk2_max = np.log(maxdict['k2'])
        logk2 = logk2_min * (1-u[i]) + logk2_max * u[i] # K log-uniform between Kmin and Kmax
        k2 = np.exp(logk2)
    
        #########################
        ### instrument params ###
        #########################
    
        Ninst = len(np.atleast_1d(suffixes))
        gamma = np.zeros(Ninst)
        jit = np.zeros(Ninst)
    
        for k,sfx in enumerate(np.atleast_1d(suffixes)):
    
            i+=1
            gamma[k] =  mindict['gamma{}'.format(sfx)] * (1-u[i]) + maxdict['gamma{}'.format(sfx)] * u[i]
    
            i+=1
            logjit_min = np.log(mindict['jit{}'.format(sfx)])
            logjit_max = np.log(maxdict['jit{}'.format(sfx)])
            logjit = logjit_min * (1-u[i]) + logjit_max * u[i]
            jit[k] = np.exp(logjit)
    
        return np.append(
            np.array([per1,tc1,e1,w1,k1,per2,tc2,e2,w2,k2]),
            np.vstack((gamma,jit)).T.reshape(-1)
        )
    
    ######################
    ### Set up sampler ###
    ######################
    def pt(u):
        return prior_transform(u,mindict,meddict,maxdict,suffixes)
    Npars = len(like.list_vary_params())
    suffixes = like.suffixes
    modpars = [like.list_vary_params().index('w1'),like.list_vary_params().index('w2')]
    
    sampler_full_model = dynesty.NestedSampler(
        like.logprob_array,
        pt,
        Npars,
        periodic=modpars,
        sample='rwalk'
    )
    
    ####################
    ### run  sampler ###
    ####################
    full_model_results = sampler_full_model.run_nested()
    full_model_results=sampler_full_model.results
    with open(datadir+"/full_model_nested_sampling_results_v1.0.pkl","wb") as fi:
        pickle.dump(full_model_results,fi)
