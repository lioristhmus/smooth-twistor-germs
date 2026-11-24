"""Read the exact node fields used in the exterior residual test."""
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import legvander
from modal import Panels
from geometry import geometry,slice_values,sup,I
from pp import Grid,pull,tf,york,LW,divergence,roots
def load(base):
    g=Panels([8,8,10,16],32)
    arr=[[],[]]
    files=sorted(base.glob('native-*.npz'));assert len(files)==len(g.r)
    for i,path in enumerate(files):
        with np.load(path) as d:
            assert d['r']==g.r[i]
            for bins,key in zip(arr,['h','K']):bins.append(d[key])
    return g,np.stack(arr[0]),np.stack(arr[1])
def check(base):
    g,h,K=load(base)
    H,M,_,_=geometry(g,h,K)
    mask=g.r>1
    hi,Ki,*_=slice_values(g.x[~mask,...,2],.012)
    import json
    previous=json.loads((base/'results.json').read_text())
    original=previous['history'][-1]
    result=dict(H=sup(H[mask]),M=sup(M[mask]),inner_h=sup(h[~mask]-hi),
                inner_K=sup(K[~mask]-Ki))
    assert result['H']==original[0] and result['M']==original[1]
    # The portable coarse interpolation is an explicitly measurable representation change.
    from pp import extension
    small=Grid([8,8,10,16],dict(radius=1))
    h0,K0=extension(small,base)
    H0,M0,_,_=geometry(small,h0,K0)
    result['resampled_H']=sup(H0[small.r>1]);result['resampled_M']=sup(M0[small.r>1])
    return result
def packet(base,cap):
    g,h,K=load(base);g.kernels=g.integration(g.r)
    h,K,dis,hist=pull(g,h,K)
    q=tf(h-I)
    kap=K-(np.trace(K,axis1=-2,axis2=-1)/np.trace(h,axis1=-2,axis2=-1))[...,None,None]*h
    p=kap-LW(g,york(g,divergence(g,kap)))
    tag='band'+str(cap)
    k,wk=roots(g,q,p,cap,tag,True)
    target=Grid([8,8,10,16],dict(radius=1))
    m=g.modes(dis)
    for i,r in enumerate(target.r):
        j=np.searchsorted(g.edges,r)-1;sl=slice(j*len(g.v),(j+1)*len(g.v))
        row=legvander([(r-g.mid[j])/g.half[j]],len(g.v)-1)[0]@g.basis_inv
        np.savez_compressed(f'{tag}-mark-{i:03}.npz',X=np.einsum('r,rhc->hc',row,m[sl]))
    return {str(cap):dict(k=k.tolist(),wk=wk.tolist(),gauge=hist,norm_q=sup(q[g.r<1]),
        norm_p=sup(p[g.r<1]),spec=[8,8,10,16],cap=cap)}
