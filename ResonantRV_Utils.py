import numpy as np
from ResonantPairModel import ACRModel, ACRModelPrior,ACRModelPriorTransform,RadvelModelPriorTransform
from scipy.interpolate import interp2d,RectBivariateSpline
import radvel
import pandas as pd
import pickle
import rebound as rb
import matplotlib.pyplot as plt

class acr_function():
    """
    Class for computing planet eccentricities
    for different ACR configurations.
    """
    def __init__(self,res_j,res_k,ACR_data_file="./ACR_locations_data.pkl"):
        """
        Parameters
        ----------
        res_j : int
            Together with res_k specifies the particular MMR
            as the j_res:j_res-k_res resonance.
        res_k : int
            Order of the resonance
        ACR_data_file : str
            File path for binary file containting dictionary of ACR curve data.
        """
        with open(ACR_data_file,'rb') as fi:
            all_acr_loc_dict = pickle.load(fi)
        acr_loc_dict = all_acr_loc_dict[(res_j,res_k)]
        gammas = np.array(list(acr_loc_dict.keys()))
        Nt = len(acr_loc_dict[gammas[0]][0])
        t = np.linspace(0,1,Nt)

        e1_arr = np.array([acr_loc_dict[gamma][0] for gamma in gammas])
        e2_arr = np.array([np.abs(acr_loc_dict[gamma][1]) for gamma in gammas])
        
        self.e1_rbs = RectBivariateSpline(gammas,t,e1_arr)
        self.e2_rbs = RectBivariateSpline(gammas,t,e2_arr)

    def __call__(self,gamma,t):
        """
        Arguments
        ---------
        gamma : float
            Planets' mass ratio = M_out / M_in
        t : float
            Paramterization of the ACR configuration's AMD.
            t=0 corresponds to circular orbits and t=1 corresponds
            to the point where ACR curves of different mass ratios
            all meet.
        """
        return self.e1_rbs(gamma,t)[0,0],self.e2_rbs(gamma,t)[0,0]

def get_acr_like(observations_df,j,k):
    """
    Get a radvel likelihood object for an ACR radial velocity model.
    
    Arguments
    ---------
    observations_df : pandas DataFrame
        Dataframe containing the radial observations.
        Required columns:
            'instrument','time','velocity','uncertainty'
    j : int
        Speficies the j:j-k resonance for the ACR model
    k : int
        Order of resonance for ACR model.

    Returns
    -------
    radvel CompositeLikelihood
    """
    acrfn = acr_function(j,k)
    mdl = ACRModel(j,k , acrfn, time_base=observations_df.time.median())
    likes=[]
    instruments = observations_df.instrument.unique()
    for i,instrument in enumerate(instruments):
        data = observations_df.query('instrument==@instrument')[['time','velocity','uncertainty']].values
        inst_like = radvel.likelihood.RVLikelihood(mdl,*(data.T),suffix=instrument)
        gammainst = 'gamma{}'.format(instrument)
        jitinst = 'jit{}'.format(instrument)
        inst_like.params[gammainst].value=0
        inst_like.params[jitinst].value=0
        likes.append(inst_like)
    like = radvel.likelihood.CompositeLikelihood(likes)
    like.params['dvdt'].vary = False
    like.params['curv'].vary = False
    return like

def get_radvel_like(observations_df):
    """
    Get a radvel likelihood object for a two-planet radial velocity model.
    
    Arguments
    ---------
    observations_df : pandas DataFrame
        Dataframe containing the radial observations.
        Required columns:
            'instrument','time','velocity','uncertainty'

    Returns
    -------
    radvel CompositeLikelihood
    """
    # Set up radvel likelihood object
    fitbasis = 'per tc e w k'
    params = radvel.Parameters(2,basis=fitbasis)
    mdl = radvel.RVModel(params,time_base=observations_df.time.median())
    likes=[]
    instruments = observations_df.instrument.unique()
    for i,instrument in enumerate(instruments):
        data = observations_df.query('instrument==@instrument')[['time','velocity','uncertainty']].values
        inst_like = radvel.likelihood.RVLikelihood(mdl,*(data.T),suffix=instrument)
        # ... setting pars to 0 is for some reason 
        # neccessary to stop the code from eating shit...
        for key,par in inst_like.params.items():
            par.value = 0
        likes.append(inst_like)
    like = radvel.likelihood.CompositeLikelihood(likes)
    like.params['dvdt'].vary = False
    like.params['curv'].vary = False
    return like

def synthpars_to_sim(spars,t0=0,Mstar = 1,DataInMetersPerSecond=False):
    """
    Get a rebound simulations from a set of radvel parameters in the 'synth' basis.

    Arguments
    ---------
    spars : radvel.Parameters
        A set of radvel parameters in the synth basis. 
        These can be generated from a likelihood object using
        Likelihood.model.get_synthparams()
    t0 : float, optional
        Simulation's epoch (stored as sim.t).
        Default is 0
    Mstar : float, optional
        Stellar mass in solar masses.
        Default is 1.
    DataInMetersPerSecond : boole, (default=False)
        Semi-amplitudes are in meters per second.

    Returns
    -------
    rebound.Simulation object
    """
    if DataInMetersPerSecond:
        Kfactor = 1731*1e3
    else:
        Kfactor = 1731
    sim = rb.Simulation()
    sim.units = ('day','AU','Msun')
    sim.add(m=Mstar)
    sim.t = t0
    for i in range(1,3):
        P = spars['per{}'.format(i)].value
        e = spars['e{}'.format(i)].value
        omega = spars['w{}'.format(i)].value
        Tp = spars['tp{}'.format(i)].value
        M = np.mod(2 * np.pi * (t0 - Tp) / P,2*np.pi)
        k = spars['k{}'.format(i)].value
        a = P**(2/3) * (sim.G * (Mstar) * (2*np.pi)**(-2))**(1/3)
        mass = Mstar * np.sqrt(1-e*e) * (k / 1e3 / Kfactor) *  (P / 2 / np.pi / a)
        sim.add(m=mass,P=P,e=e,inc = np.pi / 2, omega=omega,M=M)
    sim.move_to_com()
    return sim

def plot_fit(like,ax=None,Npt = 200):

    if ax is None:
        fig,ax = plt.subplots(1,figsize=(12,4))
        fig.set_tight_layout(True)

    x = like.x
    y = like.model(like.x)+like.residuals()
    yerr = like.yerr
    for tel in set(like.telvec):
        msk = like.telvec==tel
        ax.errorbar(x[msk],y[msk],yerr=yerr[msk], fmt='o',label=tel)
    min_time= np.min(like.x)
    max_time = np.max(like.x)
    ti = np.linspace(min_time,max_time,Npt)
    ax.plot(ti, like.model(ti))
    ax.set_xlabel('Time')
    ax.set_ylabel('RV')


