import rebound
import numpy as np
import theano
import theano.tensor as T
from exoplanet.theano_ops.kepler import KeplerOp
import matplotlib.pyplot as plt
from celmech.disturbing_function import get_fg_coeffs
from IntegrableModelUtils import getOmegaMatrix, calc_DisturbingFunction_with_sinf_cosf
from IntegrableModelUtils import get_secular_f2_and_f10
from scipy.optimize import root_scalar,root
from warnings import warn
DEBUG = False

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
        beta2 = mu2 / (mu2+mu2)
        gamma = mu2/mu1
        
        # Resonant semi-major axis ratio
        alpha_res = ((j-k)/j)**(2/3) * ((Mstar + m1) / (Mstar+m2))**(1/3)
        
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
        Jtens = T.as_tensor(np.pad(getJmatrix(2),(0,1),'constant'))
        H_flow_vec = Jtens.dot(gradHtot)
        H_flow_jac = Jtens.dot(hessHtot)
        
        #  Dissipative flow
        dis_flow_vec = T.stack(sigma1dot_dis,sigma2dot_dis,I1dot_dis,I2dot_dis,amddot_dis)
        dis_flow_jac = theano.gradient.jacobian(dis_flow_vec,dyvars)

        ##########################
        # Compile Theano functions
        ##########################
        
        if not DEBUG:
            # Note that compiling can take a while
            #  so I've put a debugging switch here 
            #  to skip evaluating these functions when
            #  desired.

            Htot_fn = theano.function(
                inputs=ins,
                outputs=H_tot,
                givens=givens,
                on_unused_input='ignore'
            )
            
            H_flow_vec_fn = theano.function(
                inputs=ins,
                outputs=H_tot,
                givens=givens,
                on_unused_input='ignore'
            )
            
            H_flow_jac_fn = theano.function(
                inputs=ins,
                outputs=H_tot,
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
            dis_timescales_fn =function(
                inputs=ins,
                outputs=dis_flow_jac,
                givens=givens,
                on_unused_input='ignore'
            )
        else:
            Htot_fn,H_flow_fn,H_flow_jac_fn,dis_flow_vec_fn,dis_flow_jac_fn,dis_timescales_fn = [lambda x: x for x range(6)]
        
        return Htot_fn,H_flow_fn,H_flow_jac_fn,dis_flow_vec_fn,dis_flow_jac_fn,dis_timescales_fn
        

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
        funcs = get_compiled_theano_functions(N_QUAD_PTS)

        self._H,self._H_flow,self._H_jac,self._dis_flow,self._dis_jac,self._times_scales = funcs
    
    @property
    def extra_args(self):
        return [self.m1,self.m2,self.j,self.k,self.tau_alpha,self.K1,self.K2,self.p]

    @property
    def mu1(self):
        self.m1 / (1 + self.m1)

    @property
    def mu2(self):
        self.m2 / (1 + self.m2)

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

    def flow(self,z)
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

    def flow_jac(self,z)
        """
        Calculate flow Jacobian
        .. math:
            \Omega \cdot \Delta H(z) + \nabla f_{dis}(z)

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
        """
        Convert dynamical variables
        .. math:
            z = (\sigma_1,\sigma_2,I_1,I_2,{\cal C}
        to orbital elements
        .. math:
            (a_1,e_1,\theta_1,a_2,e_2,\theta_2)
        """
        sigma1,sigma2,I1,I2,amd = z
        a1,e1,a2,e2 = I1I2_to_a1e1_a2e2(I1,I2,amd,self.m2 /self.m1 ,self.j,self.k)
        return np.array((a1,e1,sigma1 * self.k, a2, e2, sigma2 * self.k))

    def orbels_to_dyvars(self,orbels):
        """
        Convert dynamical variables
        .. math:
            z = (\sigma_1,\sigma_2,I_1,I_2,{\cal C}
        to orbital elements
        .. math:
            (a_1,e_1,\theta_1,a_2,e_2,\theta_2)
        """
       # Total angular momentum constrained by
       # Ltot = beta1 * sqrt(alpha_res) + beta2 - amd
       a1,e1,theta1,a2,e2,theta2 = orbels
       sigma1 = theta1 / self.k
       sigma2 = theta2 / self.k
       I1 = self.beta1 * np.sqrt(a1/a2) * (1 - np.sqrt(1 - e1*e1) ) 
       I2 = self.beta2 * (1 - np.sqrt(1 - e2*e2) ) 
       s = (self.j - self.k) / self.k
       P0 = ( beta2 - beta1 * np.sqrt( self.alpha ) ) / 2
       P = 0.5 * (self.beta2 - self.beta1 * np.sqrt(a1/a2)) - (s+0.5) * (I1+I2)
       amd = (P0 - P) / (s + 0.5)
       return np.array((sigma1,sigma2,I1,I2,amd))
