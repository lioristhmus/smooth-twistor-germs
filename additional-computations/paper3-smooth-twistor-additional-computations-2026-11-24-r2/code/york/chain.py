"""Explicit fourth-order symbol, three calibrated contact responses, STF contractions."""
import numpy as np
def symbols(v):
    k=np.asarray(v,float);r=np.linalg.norm(k,axis=-1)
    assert np.min(r)>0
    D=k[...,None]*k[...,None,:]-r[...,None,None]**2*np.eye(3)
    X=np.stack([np.cross(k,e) for e in np.eye(3)],axis=-1)
    def curl(b):
        c=X@b
        return .5j*(c+c.swapaxes(-1,-2))
    def response(b):
        root_sum=2j*curl(b)
        dual=root_sum/(16*r[...,None,None]**4)
        return 4j*curl(dual)
    basis=[np.diag([1,-1,0]),np.diag([1,1,-2])]
    for i,j in ((0,1),(0,2),(1,2)):
        e=np.zeros((3,3));e[i,j]=e[j,i]=1;basis.append(e)
    N=np.zeros_like(r);M=np.zeros_like(D)
    for e in basis:
        b=D@e@D-.5*D*np.einsum('...ij,ij->...',D,e)[...,None,None]
        for j in range(3): b=response(b)
        b=b.real/(e*e).sum()
        N+=4*np.einsum('ij,...ij->...',e,b);M+=b@e
    return N,.75*N[...,None,None]*np.eye(3)+2*M
