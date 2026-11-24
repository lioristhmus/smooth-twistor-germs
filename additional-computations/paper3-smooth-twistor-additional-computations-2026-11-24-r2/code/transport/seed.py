"""Cartesian fourth-order TT operator on the compact radial potential, at physical points."""
import numpy as np
from numpy.polynomial import Polynomial

def field(x,cfg):
    y=x/cfg['radius']
    s=np.sum(y*y,axis=-1)
    q=np.zeros_like(s)
    inside=s<1.
    q[inside]=1/(1-s[inside])
    base=np.zeros_like(q)
    base[inside]=np.exp(cfg['bump_shape']*(1-q[inside]))
    # If b^(j)=b P_j(q), d/ds sends P to q²(P'−cP).
    pol=Polynomial([1.])
    derivatives=[]
    for j in range(4):
        pol=Polynomial([0.,0.,1.])*(pol.deriv()-cfg['bump_shape']*pol)
        derivatives.append((base*pol(q))[...,None,None])
    _,b2,b3,b4=derivatives
    eye=np.eye(3)
    E=np.array([[1.,1.,0.],[1.,-1.,0.],[0.,0.,0.]])
    yy=y[...,None]*y[...,None,:]
    v=y@E
    vv=y[...,None]*v[...,None,:]+v[...,None]*y[...,None,:]
    contraction=np.sum(y*v,axis=-1)[...,None,None]
    s=s[...,None,None]
    lap2=60*b2+80*s*b3+16*s*s*b4
    hess_lap=(20*b2+8*s*b3)*eye+(56*b3+16*s*b4)*yy
    fourth=8*b2*E+8*b3*(eye*contraction+2*vv)+16*b4*yy*contraction
    return (cfg['gain']/cfg['radius']**4)*(lap2*E-E@hess_lap-hess_lap@E+
            .5*fourth+.5*np.sum(hess_lap*E,axis=(-2,-1))[...,None,None]*eye)
