import rebound as rb
import reboundx
import numpy as np
import theano
import theano.tensor as T
from exoplanet.theano_ops.kepler import KeplerOp
from warnings import warn
DEBUG = False

def getOmegaMatrix(n):
    """
    Get the 2n x 2n skew-symmetric block matrix:
          [0 , I_n]
          [-I_n, 0 ]
    that appears in Hamilton's equations.

    Arguments
    ---------
    n : int
        Determines matrix dimension

    Returns
    -------
    numpy.array
    """
    return np.vstack(
        (
         np.concatenate([np.zeros((n,n)),np.eye(n)]).T,
         np.concatenate([-np.eye(n),np.zeros((n,n))]).T
        )
    )

def calc_DisturbingFunction_with_sinf_cosf(alpha,e1,e2,w1,w2,sinf1,cosf1,sinf2,cosf2):
    """
    Compute the value of the disturbing function
    .. math::
        \frac{a'}{|r-r'|} - a'\frac{r.r'}{|r'^3|}
    from a set of input orbital elements for coplanar planets.

    Arguments
    ---------
    alpha : float
        semi-major axis ratio
    e1 : float
        inner eccentricity
    e2 : float
        outer eccentricity
    w1 : float
        inner long. of peri
    w2 : float
        outer long. of peri
    sinf1 : float
        sine of inner planet true anomaly
    cosf1 : float
        cosine of inner planet true anomaly
    sinf2 : float
        sine of outer planet true anomaly
    cosf2 : float
        cosine of outer planet true anomaly

    Returns
    -------
    float :
        Disturbing function value
    """
    r1 = alpha * (1-e1*e1) /(1 + e1 * cosf1)
    _x1 = r1 * cosf1
    _y1 = r1 * sinf1
    Cw1 = T.cos(w1)
    Sw1 = T.sin(w1)
    x1 = Cw1 * _x1  - Sw1 * _y1
    y1 = Sw1 * _x1  + Cw1 * _y1

    r2 = (1-e2*e2) /(1 + e2 * cosf2)
    _x2 = r2 * cosf2
    _y2 = r2 * sinf2
    Cw2 = T.cos(w2)
    Sw2 = T.sin(w2)
    x2 = Cw2 * _x2  - Sw2 * _y2
    y2 = Sw2 * _x2  + Cw2 * _y2

    # direct term
    dx = (x2 - x1)
    dy = (y2 - y1)
    dr2 = dx*dx + dy*dy
    direct = 1 / T.sqrt(dr2)

    # indirect term
    r1dotr = (x2 * x1 + y2 * y1)
    r1sq = x2*x2 + y2*y2
    r1_3 = 1 / r1sq / T.sqrt(r1sq)
    indirect = -1 * r1dotr * r1_3

    return direct+indirect

def cart_vars_to_a1e1s1_a2e2s2(cart_vars,gamma,j,k):
    y1,y2,x1,x2,amd = cart_vars
    s1 = np.arctan2(y1,x1)
    s2 = np.arctan2(y2,x2)
    I1 = (x1*x1 + y1*y1) / 2
    I2 = (x2*x2 + y2*y2) / 2
    a1,e1,a2,e2 = I1I2_to_a1e1_a2e2(I1,I2,amd,gamma,j,k)
    return a1,e1,s1,a2,e2,s2

def I1I2_to_a1e1_a2e2(I1,I2,amd,gamma,j,k):
    """
    Convert canonical momenta to orbital
    elements for resonant planet pair. 

    Arguments
    ---------
    I1 : float
        Inner planet's momentum I=beta * sqrt(a) * (1-sqrt(1-e^2)). 

    I2 : float
        Outer planet's momentum I=beta * sqrt(a) * (1-sqrt(1-e^2)). 

    amd : float
        Quantity defining the value of planet's AMD, I1+I2, when
        the planets are at the  location of nominal resonance,
        a2 = [j/(j-k)]^(2/3) * a1. This quantity is conserved in 
        the absence of dissipation.

    gamma : float
        Planet's mass ratio, gamma = m2/m1.

    j : int
        Integer that, together with k, defines the j:j-k resonance.

    k : int
        Order of the resonance.

    Returns
    -------
    tuple of floats :
     Returns semi-major axes and eccentricities in the order:
        (a1,e1,a2,e2)
    """
    beta1 = 1/(1+gamma)
    beta2 = 1-beta1
    Gamma1 = I1
    Gamma2 = I2

    s = (j-k)/k
    alpha_res = ((j-k)/j)**(2/3)
    P0 = ( beta2 - beta1 * np.sqrt(alpha_res) ) / 2
    P = P0 - (s+1/2) * amd
    
    Ltot = beta1 * np.sqrt(alpha_res) + beta2 - amd
    L1 = Ltot/2 - P - s * (I1 + I2)
    L2 = Ltot/2 + P + (1 + s) * (I1 + I2)


    a1 = (L1 / beta1 )**2
    e1 = np.sqrt(1-(1-(Gamma1 / L1))**2)

    a2 = (L2 / beta2 )**2
    e2 = np.sqrt(1-(1-(Gamma2 / L2))**2)

    return a1,e1,a2,e2

def get_compiled_theano_functions(N_QUAD_PTS):
        # resonance j and k
        j,k = T.lscalars('jk')
        s = (j-k) / k

        # Planet masses: m1,m2
        m1,m2 = T.dscalars(2)
        Mstar = 1
        mu1 = m1 / (Mstar + m1)
        mu2 = m2 / (Mstar + m2)
        eps = m1 * mu2 / (mu1 + mu2) / Mstar
        beta1 = mu1 / (mu1+mu2)
        beta2 = mu2 / (mu1+mu2)
        gamma = mu2/mu1
        
        # Angle variable for averaging over
        Q = T.dvector('Q')

        # Dynamical variables:
        dyvars = T.vector()
        sigma1, sigma2, I1, I2, amd = [dyvars[i] for i in range(5)]


        # Quadrature weights
        quad_weights = T.dvector('w')
        
        # Set lambda2=0
        l2 = T.constant(0.)
        
        l1 = k*Q 
        w1 = (1+s) * l2 - s * l1 - sigma1
        w2 = (1+s) * l2 - s * l1 - sigma2
        
        Gamma1 = I1
        Gamma2 = I2
        
        # Resonant semi-major axis ratio
        alpha_res = ((j-k)/j)**(2/3) * ((Mstar + m1) / (Mstar+m2))**(1/3)
        P0 = ( beta2 - beta1 * T.sqrt(alpha_res) ) / 2
        P = P0 - (s+1/2) * amd
        
        Ltot = beta1 * T.sqrt(alpha_res) + beta2 - amd
        L1 = Ltot/2 - P - s * (I1 + I2)
        L2 = Ltot/2 + P + (1 + s) * (I1 + I2)
        
        a1 = (L1 / beta1 )**2
        e1 = T.sqrt(1-(1-(Gamma1 / L1))**2)
        
        a2 = (L2 / beta2 )**2
        e2 = T.sqrt(1-(1-(Gamma2 / L2))**2)
        
        Hkep = -beta1 / (2 * a1) - beta2 / (2 * a2)
        
        alpha = a1 / a2
        
        ko = KeplerOp()
        
        M1 = l1 - w1
        M2 = l2 - w2
        
        sinf1,cosf1 =  ko( M1, e1 + T.zeros_like(M1) )
        sinf2,cosf2 =  ko( M2, e2 + T.zeros_like(M2) )
        
        R = calc_DisturbingFunction_with_sinf_cosf(alpha,e1,e2,w1,w2,sinf1,cosf1,sinf2,cosf2)
        Rav = R.dot(quad_weights)
        
        Hpert = -eps * Rav / a2
        Htot = Hkep + Hpert

        ######################
        # Dissipative dynamics
        ######################
        tau_alpha_0, K1, K2, p = T.dscalars(4)
        sigma1dot_dis,sigma2dot_dis,I1dot_dis,I2dot_dis,amddot_dis = T.dscalars(5)
        sigma1dot_dis,sigma2dot_dis = T.as_tensor(0.),T.as_tensor(0.)
        
        # Define timescales
        tau_e1 = tau_alpha_0 / K1
        tau_e2 = tau_alpha_0 / K2
        tau_a1_0 = -1 * tau_alpha_0 * (1+alpha_res * gamma)/ (alpha_res * gamma)
        tau_a2_0 = -1 * alpha_res * gamma * tau_a1_0
        tau_a1 = 1 / (1/tau_a1_0 + 2 * p * e1*e1 / tau_e1 )
        tau_a2 = 1 / (1/tau_a2_0 + 2 * p * e2*e2 / tau_e2 )

        # Time derivative of orbital elements
        e1dot_dis = -1*e1 / tau_e1
        e2dot_dis = -1*e2 / tau_e2
        a1dot_dis = -1*a1 / tau_a1
        a2dot_dis = -1*a2 / tau_a2

        # Time derivatives of canonical variables
        I1dot_dis = L1 * e1 * e1dot_dis / ( T.sqrt(1-e1*e1) ) - I1 / tau_a1 / 2
        I2dot_dis = L2 * e2 * e2dot_dis / ( T.sqrt(1-e2*e2) ) - I2 / tau_a2 / 2
        Pdot_dis = -1 * ( L2 / tau_a2 - L1 / tau_a1) / 4 - (s + 1/2) * (I1dot_dis + I2dot_dis)
        amddot_dis = Pdot_dis / T.grad(P,amd)

        #####################################################
        # Set parameters for compiling functions with Theano
        #####################################################
        
        # Get numerical quadrature nodes and weights
        nodes,weights = np.polynomial.legendre.leggauss(N_QUAD_PTS)
        
        # Rescale for integration interval from [-1,1] to [-pi,pi]
        nodes = nodes * np.pi
        weights = weights * 0.5
        
        # 'givens' will fix some parameters of Theano functions compiled below
        givens = [(Q,nodes),(quad_weights,weights)]

        # 'ins' will set the inputs of Theano functions compiled below
        #   Note: 'extra_ins' will be passed as values of object attributes
        #   of the 'ResonanceEquations' class 'defined below
        extra_ins = [m1,m2,j,k,tau_alpha_0,K1,K2,p]
        ins = [dyvars] + extra_ins
        

        # Define flows and jacobians.

        #  Conservative flow
        gradHtot = T.grad(Htot,wrt=dyvars)
        hessHtot = theano.gradient.hessian(Htot,wrt=dyvars)
        Jtens = T.as_tensor(np.pad(getOmegaMatrix(2),(0,1),'constant'))
        H_flow_vec = Jtens.dot(gradHtot)
        H_flow_jac = Jtens.dot(hessHtot)
        
        #  Dissipative flow
        dis_flow_vec = T.stack(sigma1dot_dis,sigma2dot_dis,I1dot_dis,I2dot_dis,amddot_dis)
        dis_flow_jac = theano.gradient.jacobian(dis_flow_vec,dyvars)
        
        
        # Extras
        dis_timescales = [tau_a1_0,tau_a2_0,tau_e1,tau_e2]
        orbels = [a1,e1,sigma1*k,a2,e2,sigma2*k]
        ##########################
        # Compile Theano functions
        ##########################
        
        if not DEBUG:
            # Note that compiling can take a while
            #  so I've put a debugging switch here 
            #  to skip evaluating these functions when
            #  desired.
            Rav_fn = theano.function(
                inputs=ins,
                outputs=Rav,
                givens=givens,
                on_unused_input='ignore'
            )
            Htot_fn = theano.function(
                inputs=ins,
                outputs=Htot,
                givens=givens,
                on_unused_input='ignore'
            )
            
            H_flow_vec_fn = theano.function(
                inputs=ins,
                outputs=H_flow_vec,
                givens=givens,
                on_unused_input='ignore'
            )
            
            H_flow_jac_fn = theano.function(
                inputs=ins,
                outputs=H_flow_jac,
                givens=givens,
                on_unused_input='ignore'
            )
            
            dis_flow_vec_fn = theano.function(
                inputs=ins,
                outputs=dis_flow_vec,
                givens=givens,
                on_unused_input='ignore'
            )
            
            dis_flow_jac_fn = theano.function(
                inputs=ins,
                outputs=dis_flow_jac,
                givens=givens,
                on_unused_input='ignore'
            )

            dis_timescales_fn =theano.function(
                inputs=extra_ins,
                outputs=dis_timescales,
                givens=givens,
                on_unused_input='ignore'
            )

            orbels_fn = theano.function(
                inputs=ins,
                outputs=orbels,
                givens=givens,
                on_unused_input='ignore'
            )

        else:
            return  [lambda x: x for _ in range(8)]
        
        return Rav_fn,Htot_fn,H_flow_vec_fn,H_flow_jac_fn,dis_flow_vec_fn,dis_flow_jac_fn,dis_timescales_fn,orbels_fn
        

class ResonanceEquations():
    """
    A class for the model describing the dynamics of a pair of planar planets
    in/near a mean motion resonance.

    Includes the effects of dissipation.

    Attributes
    ----------
    j : int
        Together with k specifies j:j-k resonance
    
    k : int
        Order of resonance.
    
    alpha : float
        Semi-major axis ratio a_1/a_2

    eps : float
        Mass parameter m1*mu2 / (mu1+mu2)

    m1 : float
        Inner planet mass

    m2 : float
        Outer planet mass

    """
    def __init__(self,j,k, n_quad_pts = 40, m1 = 1e-5 , m2 = 1e-5,K1=100, K2=100, tau_alpha = 1e5, p = 1):
        self.j = j
        self.k = k
        self.m1 = m1
        self.m2 = m2
        self.K1 = K1
        self.K2 = K2
        self.tau_alpha = tau_alpha
        self.p = p 
        funcs = get_compiled_theano_functions(n_quad_pts)

        self._Rav_fn,self._H,self._H_flow,self._H_jac,self._dis_flow,self._dis_jac,self._times_scales,self._orbels_fn = funcs
    
    @property
    def extra_args(self):
        return [self.m1,self.m2,self.j,self.k,self.tau_alpha,self.K1,self.K2,self.p]

    @property
    def mu1(self):
        return self.m1 / (1 + self.m1)

    @property
    def mu2(self):
        return self.m2 / (1 + self.m2)

    @property
    def beta1(self):
        return self.mu1 / (self.mu1 + self.mu2)

    @property
    def beta1(self):
        return self.mu1 / (self.mu1 + self.mu2)

    @property
    def beta2(self):
        return self.mu2 / (self.mu1 + self.mu2)
    
    @property
    def alpha(self):
        return ((self.j - self.k) / self.j)**(2/3)

    @property
    def timescales(self):
        tscales_array = self._times_scales(*self.extra_args)
        return {name:scale for name,scale in zip(['tau_a1','tau_a2','tau_e1','tau_e2'],tscales_array)}

    def Rav(self,z):
        """
        Calculate the value of the averaged disturbing function

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        float : 
            The value of the Hamiltonian evaluated at z.
        """
        return self._Rav_fn(z,*self.extra_args)

    def H(self,z):
        """
        Calculate the value of the Hamiltonian.

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        float : 
            The value of the Hamiltonian evaluated at z.
        """
        return self._H(z,*self.extra_args)
    
    def H_flow(self,z):
        """
        Calculate flow induced by the Hamiltonian
        .. math:
            \dot{z} = \Omega \cdot \nablda_{z}H(z)

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            Flow vector
        """
        return self._H_flow(z,*self.extra_args)


    def H_flow_jac(self,z):
        """
        Calculate the Jacobian of the flow induced by the Hamiltonian
        .. math:
             \Omega \cdot \Delta H(z)

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            Jacobian matrix
        """
        return self._H_jac(z,*self.extra_args)

    def dis_flow(self,z):
        """
        Calculate flow induced by dissipative forces
        .. math:
            \dot{z} = f_{dis}(z)

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            Flow vector
        """
        return self._dis_flow(z,*self.extra_args)

    def dis_flow_jac(self,z):
        """
        Calculate the Jacobian of the flow induced by dissipative forces
        .. math:
            \dot{z} = \nabla f_{dis}(z)

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            Flow Jacobian
        """
        return self._dis_jac(z,*self.extra_args)

    def flow(self,z):
        """
        Calculate flow 
        .. math:
            \dot{z} = \Omega \cdot \nablda_{z}H(z) + f_{dis}(z)

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            Flow vector
        """
        return self._H_flow(z,*self.extra_args) + self._dis_flow(z,*self.extra_args)

    def flow_jac(self,z):
        """
        Calculate flow Jacobian
        .. math:
            \Omega \cdot \Delta H(z) + \\nabla f_{dis}(z)

        Arguments
        ---------
        z : ndarray
            Dynamical variables

        Returns
        -------
        ndarray : 
            Jacobian
        """
        return self._H_jac(z,*self.extra_args) + self._dis_jac(z,*self.extra_args)

    def dyvars_to_orbels(self,z):
        r"""
        Convert dynamical variables
        .. math:
            z = (\sigma_1,\sigma_2,I_1,I_2,{\cal C}
        to orbital elements
        .. math:
            (a_1,e_1,\\theta_1,a_2,e_2,\\theta_2)
        """
        return self._orbels_fn(z,*self.extra_args)

    def orbels_to_dyvars(self,orbels):
        r"""
        Convert orbital elements
        .. math:
            (a_1,e_1,\theta_1,a_2,e_2,\theta_2)
        to dynamical variables
        .. math:
            z = (\sigma_1,\sigma_2,I_1,I_2,{\cal C}
        """
        # Total angular momentum constrained by
        # Ltot = beta1 * sqrt(alpha_res) + beta2 - amd
        a1,e1,theta1,a2,e2,theta2 = orbels
        sigma1 = theta1 / self.k
        sigma2 = theta2 / self.k
        I1 = self.beta1 * np.sqrt(a1/a2) * (1 - np.sqrt(1 - e1*e1) ) 
        I2 = self.beta2 * (1 - np.sqrt(1 - e2*e2) ) 
        s = (self.j - self.k) / self.k
        P0 = ( self.beta2 - self.beta1 * np.sqrt( self.alpha ) ) / 2
        P = 0.5 * (self.beta2 - self.beta1 * np.sqrt(a1/a2)) - (s+0.5) * (I1+I2)
        amd = (P0 - P) / (s + 0.5)
        return np.array((sigma1,sigma2,I1,I2,amd))

    def dyvars_to_rebound_simulation(self,z,pomega1=0,Q=0,include_dissipation = False):
        r"""
        Convert dynamical variables
        .. math:
            z = (\sigma_1,\sigma_2,I_1,I_2,{\cal C}
        to a Rebound simulation.
        """
        orbels = self.dyvars_to_orbels(z)
        j,k = self.j, self.k
        a1,e1,theta1,a2,e2,theta2 = orbels
        pomega2 = np.mod( pomega1 +  (theta2 - theta1) / k, 2*np.pi)
        M1 = np.mod( (theta1 - j*Q) / k ,2*np.pi )
        M2 = np.mod( (theta2 - (j-k) * Q) / k ,2*np.pi )
        sim = rb.Simulation()
        sim.add(m=1)
        sim.add(m = self.m1,a=a1,e=e1,M=M1,pomega=pomega1)
        sim.add(m = self.m2,a=a2,e=e2,M=M2,pomega=pomega2)
        rebx = None
        sim.move_to_com()
        if include_dissipation:
            ps = sim.particles
            rebx = reboundx.Extras(sim)
            mod = rebx.load_operator("modify_orbits_direct")
            rebx.add_operator(mod)
            mod.params["p"] = self.p
            timescales = self.timescales
            ps[1].params["tau_a"]=-1*timescales['tau_a1']
            ps[2].params["tau_a"]=-1*timescales['tau_a2']
            ps[1].params["tau_e"]=-1*timescales['tau_e1']
            ps[2].params["tau_e"]=-1*timescales['tau_e2']
        return sim,rebx
