import radvel
import rebound as rb
import numpy as np

_AU_PER_DAY_IN_KM_PER_SECOND=1731.456836805555556

def _nb_params_to_sim(params,time_base):
    sim = rb.Simulation()
    sim.units=('Msun','days','AU')
    sim.t = time_base
    mstar = params['Mstar'].value
    sim.add(m=mstar,hash='star')
    star = sim.particles['star']
    for i in range(1,params.num_planets + 1):
        mass = params['m{}'.format(i)].value
        per = params['per{}'.format(i)].value
        mean_anom = params['M{}'.format(i)].value
        secosw = params['secosw{}'.format(i)].value
        sesinw = params['sesinw{}'.format(i)].value
        e = secosw * secosw + sesinw * sesinw
        pomega =np.arctan2(sesinw,secosw)
        sim.add(primary = star, m=mass,P=per,e=e,M=mean_anom,pomega=pomega,inc=np.pi/2,hash=i)
    sim.move_to_com()
    sim.exit_min_distance = 3 * np.max([p.rhill for p in sim.particles[1:]])
    return sim

def _nb_rv_calc(time,params,time_base,meters_per_second =True):
    vel = np.zeros(len(time))
    sim = _nb_params_to_sim(params,time_base)
    star = sim.particles['star']
    for i,t in enumerate(time):
        try:
            sim.integrate(t)
        except rb.Encounter:
            return np.inf * np.ones(len(vel))
        vel[i] = -1 * star.vz
    vel *= _AU_PER_DAY_IN_KM_PER_SECOND
    if meters_per_second:
        vel *=1e3
    return vel

class NbodyModel(radvel.GeneralRVModel):
    def __init__(self,Nplanets,Mstar = 1.0, time_base=0,meters_per_second=True):
        params = radvel.Parameters(Nplanets,basis ='per tc secosw sesinw k')
        self.num_planets = params.num_planets
        self.meters_per_second = meters_per_second
        for i in range(1,Nplanets+1):
            params.pop("k{}".format(i))
            params.pop("tc{}".format(i))
            params.update(
             {"m{}".format(i):radvel.Parameter()}
            )
            params.update(
             {"M{}".format(i):radvel.Parameter()}
            )

        params.update(
            {"Mstar":radvel.Parameter(value=Mstar,vary=False)}
        )
        super(NbodyModel,self).__init__(params,_nb_rv_calc,time_base)

    def __call__(self,t,*args,**kwargs):
        return super(NbodyModel,self).__call__(t,self.time_base,*args,**kwargs)
    def get_sim(self):
        return _nb_params_to_sim(self.params,self.time_base)