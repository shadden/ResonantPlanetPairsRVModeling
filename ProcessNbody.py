import sys
import numpy as np
import radvel
import pickle
import pandas as pd
import re
from ResonantRV_Utils import get_acr_like, get_full_like, synthpars_to_sim

# Number of samples to generate simulations for:
Nsamples = 4500

# Perform stability test for simulations?
stability_test = True

# Stellar mass to adopt for N-body simulation
HostStarDictionary = {
  'BD+20 2475':[10.83,"Stassun2017"], 
  'HD 102329':[3.21, "Stassun2017"],
  'HD 116029':[0.83, "Stassun2017"],
  'HD 200964':[1.39, "Luhn2019"],
  'HD 202696':[1.91,"Trifonov2019"],
  'HD 204313':[1.03,"Stassun2017"],
  'HD 33844':[1.84,"Stassun2017"],
  'HD 45364':[0.82,"Correia2009"],
  'HD 5319':[1.27,"Luhn2019"],
  'HD 99706':[1.46,"Luhn2019"]
}

# Read in observations
DATADIR = "./saves/"

AllObservations = pd.read_pickle("/Users/shadden/DropboxSmithsonian/ResonantPlanetPairsRVModeling/"+"./data/All_Observations.pkl")
I = int(sys.argv[1])
system = AllObservations.system.unique()[I]
Observations = AllObservations.query('system==@system')
Mstar,citekey = HostStarDictionary[system]

# Set up file names for saved data
dname = re.sub("HD ","HD",system)
dname = re.sub("BD\+20 ","BD+20_",dname)
savedir = DATADIR+dname + "/"
full_model_mcmc_posterior_file = savedir + "full_model_mcmc_posterior.pkl"
full_model_nested_sampling_results_file = savedir + "full_model_nested_sampling_result.pkl"
# acr_model_mcmc_posterior_file = savedir +  "acr_model_mcmc_posterior_{}to{}_angle_n_{}.pkl"
acr_model_nested_sampling_results_file_string = savedir +  "acr_model_nested_sampling_{}to{}_angle_n_{}.pkl"

full_model_simulations_file = savedir + "full_model_simulation_sample_summary.pkl"
acr_model_simulations_file = savedir + "acr_model_simulation_sample_summary_{}to{}_angle_n_{}.pkl"

def synthesize_nbody_rv(sim,times,DataInKilometersPerSecond=False):
    rvs = np.zeros(len(times))
    star = sim.particles[0]
    for i,t in enumerate(times):
        sim.integrate(t)
        rvs[i] = -1 * star.vz
    if DataInKilometersPerSecond:
        return 1731.46 * rvs
    else:
        return 1731.46 * 1e3 * rvs

def set_timestep(sim,dtfactor):
        ps=sim.particles[1:]
        tperi=np.min([p.p * (1-p.e)**1.5 / np.sqrt(1+p.e) for p in ps])
        dt = tperi * dtfactor
        sim.dt = dt
def set_min_distance(sim,rhillfactor):
        ps=sim.particles[1:]
        rhill = np.min([ p.a * (p.m/3)**(1/3.) for p in ps if p.m > 0])
        mindist = rhillfactor * rhill
        sim.exit_min_distance=mindist

def test_stability(sim,tmax):
    sim.integrator = "whfast"
    set_timestep(sim,1 / 50)
    set_min_distance(sim,5)
    try:
        sim.integrate(tmax)
        return tmax
    except:
        return sim.t


######################
# Process full model #
######################

# 1. Generate simulations
#########################


# Likelihood function
full_model_like = get_full_like(Observations)

# MCMC posterior
full_model_posterior = pd.read_pickle(full_model_mcmc_posterior_file)
posterior_length = len(full_model_posterior)

# Random indicies to select for generating simulations
random_indicies = np.random.randint(0,posterior_length - 1,size = Nsamples)

# List to hold parameters, corresponding rebound simulation, etc.
full_model_simulations = []

vary_pars = full_model_like.list_vary_params()

# Check if data is in m/s or km/s on the first pass
pars = full_model_posterior.iloc[0]
vary_par_vals = pars[vary_pars].values
full_model_like.set_vary_params(vary_par_vals)
model_pars = full_model_like.params
spars = model_pars.basis.to_synth(model_pars)
y = full_model_like.model( full_model_like.x )
# this should sepearate m/s data vs km/s data 
KilometerPerSecondQ = np.max(np.abs(y)) < 3

for i in random_indicies:
    pars = full_model_posterior.iloc[i]
    vary_par_vals = pars[vary_pars].values
    full_model_like.set_vary_params(vary_par_vals)
    model_pars = full_model_like.params
    spars = model_pars.basis.to_synth(model_pars)
    simulation = synthpars_to_sim(
            spars,
            t0 = full_model_like.model.time_base,
            Mstar = Mstar,
            DataInKilometersPerSecond = KilometerPerSecondQ
    )
    
    full_model_simulations.append(dict({"index":i,"parameters":pars,"simulation":simulation}))

# 2. Compare Nbody RVs to radvel model
#####################################
times = full_model_like.x
for sample in full_model_simulations:
    pars = sample['parameters']
    sim = sample['simulation']
    rv_nb = synthesize_nbody_rv(sim,times,KilometerPerSecondQ)
    vary_par_vals = pars[vary_pars].values
    full_model_like.set_vary_params(vary_par_vals)
    yerr = full_model_like.errorbars()
    resids = full_model_like.residuals()
    y = full_model_like.model(times) + resids

    o_minus_c_Nbody = (y - rv_nb) / yerr
    o_minus_c_radvel = resids / yerr

    chi_sq_Nbody = o_minus_c_Nbody.dot(o_minus_c_Nbody)
    chi_sq_radvel = o_minus_c_radvel.dot(o_minus_c_radvel)
    
    sample.update({
        "ChiSquared_Nbody":chi_sq_Nbody,
        "ChiSquared_Radvel":chi_sq_radvel,
        "Nbody_RV":rv_nb
        })
if stability_test:
    # 3. Test long-term stability
    #####################################
    for sample in full_model_simulations:
        simulation = sample["simulation"]
        Tmax = simulation.t +  1e4 * simulation.particles[2].P 
        tstable = test_stability(simulation,Tmax)
        sample.update({"max_time":tstable})

# Convert 'full_model_simulations' to a dataframe
cols = list(full_model_simulations[0]['parameters'].keys()) + ['ChiSquared_Radvel','ChiSquared_Nbody']
cols += [ 'time' ] + [s + "{}".format(d) for d in range(3) for s in 'mxyzuvw']
rows = []
for sim in full_model_simulations:
    pars = sim['parameters']
    datarow = pars.values.tolist()
    datarow.append(sim['ChiSquared_Radvel'])
    datarow.append(sim['ChiSquared_Nbody'])
    simulation = sim['simulation']
    datarow.append(simulation.t)
    for p in simulation.particles:
        datarow.append(p.m)
        datarow += p.xyz
        datarow += p.vxyz
    rows.append(datarow)
dataframe = pd.DataFrame(rows,columns=cols)
dataframe.to_pickle(full_model_simulations_file)
    

######################
# Process ACR models #
######################
print("Proccessing ACR posteriors....")
print("")
import glob
import re
# "acr_model_mcmc_posterior_{}to{}_angle_n_{}.pkl"

acr_posterior_files = glob.glob(savedir + "acr_model_mcmc_posterior_*.pkl")
for acr_posterior_file in acr_posterior_files:

    # parse file name to determine resonance and angle_n values
    res_j,res_j_minus_k,angle_n = [int(x) for x in re.findall("\D(\d)\D",acr_posterior_file)]
    res_k = res_j - res_j_minus_k

    # 1. Generate simulations
    #########################
    
    # Likelihood function
    acr_model_like = get_acr_like(Observations,res_j,res_k)
    acr_model_like.params['angle_n'].value = angle_n
    
    # MCMC posterior
    acr_model_posterior = pd.read_pickle(acr_posterior_file)
    posterior_length = len(acr_model_posterior)
    
    # Random indicies to select for generating simulations
    random_indicies = np.random.randint(0,posterior_length - 1,size = Nsamples)
    
    # List to hold parameters, corresponding rebound simulation, etc.
    acr_model_simulations = []
    
    vary_pars = acr_model_like.list_vary_params()
    
    # Check if data is in m/s or km/s on the first pass
    pars = acr_model_posterior.iloc[0]
    vary_par_vals = pars[vary_pars].values
    acr_model_like.set_vary_params(vary_par_vals)
    spars = acr_model_like.model.get_synthparams()
    y = acr_model_like.model( acr_model_like.x )
    # This cut should sepearate m/s data vs km/s data 
    KilometerPerSecondQ = np.max(np.abs(y)) < 3
    
    for i in random_indicies:
        pars = acr_model_posterior.iloc[i]
        vary_par_vals = pars[vary_pars].values
        acr_model_like.set_vary_params(vary_par_vals)
        spars = acr_model_like.model.get_synthparams()
        simulation = synthpars_to_sim(
                spars,
                t0 = acr_model_like.model.time_base,
                Mstar = Mstar,
                DataInKilometersPerSecond = KilometerPerSecondQ
        )
        acr_model_simulations.append(dict({"index":i,"parameters":pars,"simulation":simulation}))
    
    # 2. Compare Nbody RVs to radvel model
    #####################################
    times = acr_model_like.x
    for sample in acr_model_simulations:
        pars = sample['parameters']
        sim = sample['simulation']
        rv_nb = synthesize_nbody_rv(sim,times,KilometerPerSecondQ)
        vary_par_vals = pars[vary_pars].values
        acr_model_like.set_vary_params(vary_par_vals)
        yerr = acr_model_like.errorbars()
        resids = acr_model_like.residuals()
        y = acr_model_like.model(times) + resids
    
        o_minus_c_Nbody = (y - rv_nb) / yerr
        o_minus_c_radvel = resids / yerr
    
        chi_sq_Nbody = o_minus_c_Nbody.dot(o_minus_c_Nbody)
        chi_sq_radvel = o_minus_c_radvel.dot(o_minus_c_radvel)
        
        sample.update({
            "ChiSquared_Nbody":chi_sq_Nbody,
            "ChiSquared_Radvel":chi_sq_radvel,
            "Nbody_RV":rv_nb
            })
    if stability_test:
        # 3. Test long-term stability
        #####################################
        for sample in acr_model_simulations:
            simulation = sample["simulation"]
            Tmax = simulation.t +  1e4 * simulation.particles[2].P 
            tstable = test_stability(simulation,Tmax)
            sample.update({"max_time":tstable})

    # Convert 'acr_model_simulations' to a dataframe
    cols = list(acr_model_simulations[0]['parameters'].keys()) + ['ChiSquared_Radvel','ChiSquared_Nbody']
    cols += [ 'time' ] + [s + "{}".format(d) for d in range(3) for s in 'mxyzuvw']
    rows = []
    for sim in acr_model_simulations:
        pars = sim['parameters']
        datarow = pars.values.tolist()
        datarow.append(sim['ChiSquared_Radvel'])
        datarow.append(sim['ChiSquared_Nbody'])
        simulation = sim['simulation']
        datarow.append(simulation.t)
        for p in simulation.particles:
            datarow.append(p.m)
            datarow += p.xyz
            datarow += p.vxyz
        rows.append(datarow)
    dataframe = pd.DataFrame(rows,columns=cols)
    dataframe.to_pickle(acr_model_simulations_file.format(res_j,res_j-res_k,angle_n) )
