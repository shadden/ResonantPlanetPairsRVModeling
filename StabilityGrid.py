import numpy as np
import rebound as rb
import sys
from ResonantRV_Utils import acr_function
if __name__=="__main__":
    res_j = int(sys.argv[1])
    res_k = int(sys.argv[2])
else:
    res_j = 5
    res_k = 2 

acr_curve_fn = acr_function(res_j,res_k)

def set_timestep(sim,dtfactor):
        ps=sim.particles[1:]
        tperi=np.min([p.P * (1-p.e)**1.5 / np.sqrt(1+p.e) for p in ps])
        dt = tperi * dtfactor
        sim.dt = dt

def set_min_distance(sim,rhillfactor):
        ps=sim.particles[1:]
        mstar = sim.particles[0].m
        rhill = np.min([ p.a * (p.m/3/mstar)**(1/3.) for p in ps if p.m > 0])
        mindist = rhillfactor * rhill

def getsim(Delta,t,m1,m2):
    gamma = m2 / m1
    e1,e2=acr_curve_fn(gamma,t)
    if e1<0:
        e1=0
    if e2<0:
        e2=0
    sim = rb.Simulation()
    sim.add(m=1)
    sim.add(m=m1,e=e1,P=1,l=0,pomega=0)
    P2 = 5/3 * (1 + Delta)
    M2 = ((1-res_j+res_k) / res_j) * np.pi
    l2 = np.mod( M2 + np.pi , 2 * np.pi)
    sim.add(m=m2,e=e2,P=P2,l=l2,pomega=np.pi)
    return sim

def run_megno_sim(par):
    Delta,t = par
    sim = getsim(Delta,t,1e-4,1e-4)
    sim.integrator = "whfast"
    set_timestep(sim,1/50)
    set_min_distance(sim,2)
    sim.init_megno()
    try:
        sim.integrate(3e3 * sim.particles[2].P,exact_finish_time=0  )
        Y = sim.calculate_megno()
        if np.isnan(Y):
            return 10
        return Y
    except rebound.Encounter:
        return 10


if __name__=="__main__":
    Ngrid = 100
    par_Delta = np.linspace(-0.02,0.02,Ngrid)
    par_t = np.linspace(0.5,1,Ngrid)
    parameters = []
    for t in par_t:
        for Delta in par_Delta:
            parameters.append((Delta,t))
    from rebound.interruptible_pool import InterruptiblePool
    pool = InterruptiblePool()
    results = pool.map(run_megno_sim,parameters)
