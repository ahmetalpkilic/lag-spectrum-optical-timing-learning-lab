"""Affine-invariant lag spectrum; original diagnostic for this lesson."""
import numpy as np
from scipy.linalg import eigvalsh
from characterization import fit_rho


def lag_spectrum(z):
    """Symmetric lag-one moment against lag-zero moment, both normalized by n.

    Common normalization bounds eigenvalues in [-1,1]. Centering biases finite
    samples: calibrate the entire old/pilot procedure, not asymptotic eigenvalues.
    No ridge is added: an identity ridge would break general affine invariance.
    """
    z=np.asarray(z,float)
    if z.ndim!=2 or z.shape[1]!=2 or len(z)<6 or not np.all(np.isfinite(z)):
        raise ValueError('Require finite (n>=6,2) sequence')
    x=z-z.mean(axis=0)
    c=x.T@x/len(x)
    cross=x[:-1].T@x[1:]/len(x)
    lag=(cross+cross.T)/2
    try:
        return eigvalsh(lag,c)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Full-rank lag-zero covariance required') from exc


def features(blocks):
    """Return two class rhos and two ordered eigenvalues per class."""
    return dict(scalar=np.array([fit_rho(z) for z in blocks]),
                spectrum=np.array([lag_spectrum(z) for z in blocks]))


def scores(old,pilot):
    # Max includes both known quality classes: thresholds calibrate family alarms.
    return {name:float(np.max(np.abs(pilot[name]-old[name]))) for name in ('scalar','spectrum')}
