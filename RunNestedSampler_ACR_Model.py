import numpy as np
from ResonantPairModel import ACRModel, ACRModelPrior
from scipy.interpolate import interp2d,RectBivariateSpline
import radvel
import pickle
import pandas as pd
import sys
import re
import dynesty
DATA = "./data/"

I = int(sys.argv[1])
order = int(sys.argv[2])
NthBestAngle = int(sys.argv[3])
DRYRUN = False

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
pratio = posterior_samples['per2']/posterior_samples['per1']
res_j = int(np.round(np.median(order + order / (pratio-1))))
assert not (res_j is 2 and order is 1), "2:1 MMR ACR data not available!"

# Add some additional columns to the posterior file
posterior_samples.columns
for i in range(1,3):
    rtx = posterior_samples["secosw{}".format(i)]
    rty = posterior_samples["sesinw{}".format(i)]
    posterior_samples["e{}".format(i)] = rtx**2+rty**2
    posterior_samples["w{}".format(i)] = np.arctan2(rty,rtx)

# Load ACR Model
ACR_datadir="./ACRData/"
with open(ACR_datadir + "acr_points_{}_to_{}.pkl".format(res_j,res_j-order),'rb') as fi:
    acr_loc_dict = pickle.load(fi)
gammas = np.array(list(acr_loc_dict.keys()))
Nt = len(acr_loc_dict[gammas[0]])
t = np.linspace(0,1,Nt)
if acr_loc_dict[gammas[0]][0,0]==0:
    e1_arr = np.array([acr_loc_dict[gamma][::-1,0] for gamma in gammas])
    e2_arr = np.array([np.abs(acr_loc_dict[gamma][::-1,1]) for gamma in gammas])
else:
    e1_arr = np.array([acr_loc_dict[gamma][:,0] for gamma in gammas])
    e2_arr = np.array([np.abs(acr_loc_dict[gamma][:,1]) for gamma in gammas])
e1_rbs = RectBivariateSpline(gammas,t,e1_arr)
e2_rbs = RectBivariateSpline(gammas,t,e2_arr)
def acr_fn(gamma,t):
    return e1_rbs(gamma,1-t)[0,0],e2_rbs(gamma,1-t)[0,0]

# Set up radvel likelihood object
mdl = ACRModel(res_j, order, acr_fn, time_base=Observations.time.median())
likes=[]
instruments = Observations.instrument.unique()
for i,instrument in enumerate(instruments):
    data = Observations.query('instrument==@instrument')[['time','velocity','uncertainty']].values
    inst_like = radvel.likelihood.RVLikelihood(mdl,*(data.T),suffix=instrument)
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

# Initialize parameters to median values from full model MCMC
posterior_samples.rename(columns=rename_dict,inplace=True)
for key,par in like.params.items():
    if key in posterior_samples.columns:
        par.value = posterior_samples[key].median()


# Determine the best angle_n parameter
from scipy.optimize import minimize
bnds = [(-np.inf,np.inf) for x in like.list_vary_params()]
bnds[like.list_vary_params().index('stcosw')] = (-0.99,0.99)
bnds[like.list_vary_params().index('stsinw')] = (-0.99,0.99)
bnds[like.list_vary_params().index('k1')] = (0,np.inf)
bnds[like.list_vary_params().index('m2_by_m1')] = (.1,10)
for inst in instruments:
    med_unc = Observations.query('instrument==@inst').uncertainty.median()
    bnds[like.list_vary_params().index('jit{}'.format(inst))] = (1e-2 * med_unc,np.inf)
maxlike = -np.infty
for n in range(like.params['jres'].value):
    like.params['angle_n'].value = n
    minesult = minimize(like.neglogprob_array,like.get_vary_params(),bounds=bnds)
    loglike = like.logprob()
    if loglike > maxlike:
        maxlike = loglike
        nbest = n
    print(n,loglike)
print("nbest: {}".format(nbest))
like.params['angle_n'] = radvel.Parameter(value=nbest,vary=False)

# Set min and max
from collections import OrderedDict
maxdict=OrderedDict({col:np.inf for col in like.list_vary_params()})
mindict=OrderedDict({col:-np.inf for col in like.list_vary_params()})
meddict=OrderedDict({col:-np.inf for col in like.list_vary_params()})
for col in like.list_vary_params():
    if col in posterior_samples.columns:
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
    ###############
    ### planets ###
    ###############
    i=0
    per1 = mindict['per1'] * (1-u[i]) + maxdict['per1'] * u[i]  # within range p1min to p2max

    i+=1
    tc1 = meddict['tc1'] + (u[i]-0.5) * per1

    i+=1
    logk1_min = np.log(mindict['k1'])
    logk1_max = np.log(maxdict['k1'])
    logk1 = logk1_min * (1-u[i]) + logk1_max * u[i] # K log-uniform between Kmin and Kmax
    k1 = np.exp(logk1)

    i+=1
    log10_m2_by_m1 = 2 * (u[i] - 0.5)
    m2_by_m1 = 10**log10_m2_by_m1

    i+=1
    t = u[i] # t from 0 to 1

    i+=1
    w1 = np.mod(2 * np.pi * u[i],2*np.pi) # omega from 0 to 2pi

    stcosw,stsinw = np.sqrt(t) * np.array((np.cos(w1),np.sin(w1)))
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
        np.array([per1,tc1,k1,m2_by_m1,stcosw,stsinw]),
        np.vstack((gamma,jit)).T.reshape(-1)
    )


Npars = len(like.list_vary_params())
suffixes = like.suffixes


def pt(u):
    return prior_transform(u,mindict,meddict,maxdict,suffixes)

sampler = dynesty.NestedSampler(
    like.logprob_array,
    pt,
    Npars,
    sample='rwalk'
)
if not DRYRUN:
        sampler.run_nested()
        results=sampler.results
        with open(datadir+"/acr_model_nested_sampling_results_{:d}to{:d}res.pkl".format(res_j,res_j-order),"wb") as fi:
            pickle.dump(results,fi)
