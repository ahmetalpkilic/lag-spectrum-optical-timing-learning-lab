"""Two independent stationary temporal modes mixed into two clock errors."""
import numpy as np
from repeat_model import correlation, residual_precision

BASE = np.array([[0., 0.], [.4, .4]])
SPLIT = np.array([[-.3, .3], [0., .8]])
AFFINE = np.array([[1.5, .4], [-.2, .8]])


def mode_sequence(normal, rhos, factor, mean):
    """e_k = L x_k; x_k,j = rho_j*x_(k-1),j + sqrt(1-rho_j²)*z_k,j.

    Stationary initial x_0=z_0 gives Gamma_h=L diag(rho**h) L.T.
    L is a physical mixing factor, not necessarily triangular after transform.
    """
    normal=np.asarray(normal,float);rhos=np.asarray(rhos,float)
    factor=np.asarray(factor,float);mean=np.asarray(mean,float)
    if normal.ndim!=2 or normal.shape[1]!=2 or len(normal)<1 or rhos.shape!=(2,) or factor.shape!=(2,2) or mean.shape!=(2,):
        raise ValueError('Require normal(n,2), rhos(2), factor(2,2), mean(2)')
    if not all(np.all(np.isfinite(x)) for x in (normal,rhos,factor,mean)) or np.any(abs(rhos)>=1) or abs(np.linalg.det(factor))<1e-15:
        raise ValueError('Require finite inputs, |rho|<1 and nonsingular factor')
    x=np.empty_like(normal);x[0]=normal[0]
    for k in range(1,len(x)):
        x[k]=rhos*x[k-1]+np.sqrt(1-rhos*rhos)*normal[k]
    return mean+x@factor.T


def mode_repeats(normal, rhos, factors, means):
    return np.array([mode_sequence(z,r,l,mu) for z,r,l,mu in zip(normal,rhos,factors,means)])


def lag_covariance(rhos, factor, lag):
    return (factor*np.asarray(rhos)**lag)@factor.T


def expected_scatter(m, assumed_rho, rhos, factor):
    """Fixed assumed rho, not a data-dependent fit on these same repeats.

    E[Z.T Q Z/(m-1)]=L diag(tr(Q R_j)/(m-1)) L.T.
    Independent characterization makes conditioning on fitted rho legitimate.
    """
    q=residual_precision(m,float(assumed_rho))
    weights=np.array([np.trace(q@correlation(m,float(r)))/(m-1) for r in rhos])
    return (factor*weights)@factor.T
