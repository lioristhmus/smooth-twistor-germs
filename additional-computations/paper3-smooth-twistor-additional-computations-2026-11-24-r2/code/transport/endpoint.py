"""Source-free readback of the saved free-space path endpoint, with separate oracles."""
from pathlib import Path
import numpy as np
from multipoles import Grid
from geometry import grad,divergence,tf,geometry,sup,I
from pp import york,LW
from continuum import cfg
from seed import field
from scipy.interpolate import BarycentricInterpolator
def run():
    g=Grid(cfg['profiles']['fine'],cfg);eps=.008
    path=Path('/up')
    z=[];k=[];V=[]
    for f in sorted(path.glob('field-*.npz')):
        with np.load(f) as record:
            z.append(record['z']);k.append(record['K']);V.append(record['V'])
    z=np.stack(z);k=np.stack(k)
    K=g.sample(k).reshape(*g.x.shape[:2],3,3)
    psi=1+eps**2*g.sample(z[...,None])[...,0];h=psi[...,None,None]**4*I
    H,M,_,_=geometry(g,h,K)
    decoded_K=K-LW(g,york(g,divergence(g,K)))
    # Third implementation of C via boundary field only (all rho and C2 are withheld).
    deg=len(g.v);bound=BarycentricInterpolator(g.r[-deg:],z[-deg:],rng=7)(g.outer)
    C=float(4*eps**2*bound[0]*g.outer/np.sqrt(4*np.pi))
    ref=Path('/far/fine');zr=[];vr=[]
    for f in sorted(ref.glob('selected-fields-*.npz')):
        with np.load(f) as d:zr.append(d['z']);vr.append(d['V'])
    refpsi=1+eps**2*g.sample(np.stack(zr)[...,None])[...,0]
    p0=field(g.x,cfg)
    refK=eps*(p0+eps**2*LW(g,g.sample(np.stack(vr))))
    return dict(H=sup(H),M=sup(M),seed=sup(decoded_K-eps*p0),ref_K=sup(K-refK),
                ref_h=sup(psi**4-refpsi**4),C=C,ratio=C/eps**2,scale_control=sup(decoded_K+eps*p0))
