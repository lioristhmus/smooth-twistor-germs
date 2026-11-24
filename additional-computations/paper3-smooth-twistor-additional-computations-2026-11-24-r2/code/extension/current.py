"""Implicit horizontal lift of the finite full-metric constraint chart, for nonzero q,p."""
from pathlib import Path
import json
import numpy as np
from smooth_packet import read_coeff,integrate
from pp import I,Grid,geometry,grad,divergence,LW,york,sup,wave_read,forward
def run(base,steps):
    rec=json.loads((base/'results.json').read_text())['32']
    g=Grid(rec['spec'],dict(radius=1));k=np.array(rec['k'])
    coeff,_=read_coeff(base,k)
    q,p=integrate(g,k,coeff,16,32)
    d=g.sample(np.stack([np.load(f)['X'] for f in sorted(base.glob('band32-mark-*.npz'))]))
    def fields(s,y):
        t1,t2=s;u,v=y
        h=I+t1*q+u[...,None,None]*I
        inv=np.linalg.inv(h);kap=t2*p+v
        K=kap-(np.sum(inv*kap,axis=(-2,-1))/3)[...,None,None]*h
        return h,K
    def fixed(s,y):
        h,K=fields(s,y)
        H,M,_,_=geometry(g,h,K)
        u,v=y
        lap=np.trace(grad(g,grad(g,u)),axis1=-2,axis2=-1)
        m,_=g.potential(g.modes((H+2*lap)[...,None]))
        return -.5*g.sample(m)[...,0],LW(g,york(g,divergence(g,v)-M))
    counters=[]
    def tangent(s,y,ds):
        # Differentiate T(s,y) without differencing two Ricci tensors.
        dy=tuple(np.zeros_like(a) for a in y);small=1.e-20
        for i in range(9):
            t=fixed(s+1j*small*ds,tuple(a+1j*small*b for a,b in zip(y,dy)))
            nxt=tuple(a.imag/small for a in t)
            delta=max(sup(a-b) for a,b in zip(nxt,dy))
            dy=nxt
            if delta<2e-11:break
        else:raise ArithmeticError(('linear inverse',delta))
        counters.append(dict(iterations=i+1,update=delta))
        return dy
    def path(segments):
        y=(np.zeros(q.shape[:2]),0*p);s=np.zeros(2)
        for seg in segments:
            ds=np.array(seg)/steps
            for i in range(steps):
                k1=tangent(s,y,ds)
                mid=tuple(a+.5*b for a,b in zip(y,k1))
                k2=tangent(s+.5*ds,mid,ds)
                y=tuple(a+b for a,b in zip(y,k2));s+=ds
        h,K=fields(s,y)
        return h,K,max(sup(a-b) for a,b in zip(y,fixed(s,y)))
    h0,K0,_=forward(g,q,p)
    h,K,defect=path([(1.,1.)])
    result=dict(steps=steps,fixed_residual=defect,direct_h=sup(h-h0),direct_K=sup(K-K0),
        readout=wave_read(g,h,K,d))
    h2,K2,defect2=path([(1.,0.),(0.,1.)])
    result.update(path_h=sup(h-h2),path_K=sup(K-K2),broken_residual=defect2,current_inverse=counters)
    return result
