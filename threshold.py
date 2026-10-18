"""Rank calibration reused from threshold-transfer lab (MIT)."""
import math
import numpy as np
from characterization import ar_sequence, fit_rho
import clock_model as c
ALPHA=.05

def rank_threshold(scores, alpha=ALPHA):
    """Strict exceedance of kth order statistic; never interpolate quantiles.

    Exchangeability gives P(new > kth) <= (n+1-k)/(n+1), marginally
    over calibration and new draws. Ties make strict rejection conservative.
    """
    scores=np.asarray(scores,float)
    if scores.ndim != 1 or len(scores)<1 or not np.all(np.isfinite(scores)):
        raise ValueError('Require a nonempty finite score vector')
    if not 0<alpha<1: raise ValueError('Require 0<alpha<1')
    k=math.ceil((len(scores)+1)*(1-alpha))
    if k>len(scores): raise ValueError('Too few calibration scores for finite threshold')
    return float(np.sort(scores)[k-1]),k


def null_scores(n, seed):
    """Each draw regenerates BOTH old (80/class) and pilot (40/class).

    This controls a new old+pilot pair, not all pilots conditional on a single
    deployed old estimate. These synthetic offline draws cost no acquisitions.
    """
    rng=np.random.default_rng(seed); scores=[]
    cov=c.covariances(4.)
    for _ in range(n):
        old=[];pilot=[]
        for g,rho in enumerate((0.,.4)):
            old.append(fit_rho(ar_sequence(rng.normal(size=(80,2)),rho,cov[g],np.zeros(2))))
            pilot.append(fit_rho(ar_sequence(rng.normal(size=(40,2)),rho,cov[g],np.zeros(2))))
        scores.append(float(np.max(np.abs(np.array(old)-pilot))))
    return scores
