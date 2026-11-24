"""Recover the retained compact radial family from its stored paired roots."""
from pathlib import Path
import json
import numpy as np
from spectral import Grid,bump_seed
from read_cr import decode
import records
from seed import field
from numpy.polynomial import Polynomial
from itertools import combinations_with_replacement
from collections import Counter
cfg=json.loads(Path('/src/far-plan.json').read_text())
def extract():
    base=Path('/cr')
    mark=json.loads((base/'marking.json').read_text())
    basis=[np.diag([1.,-1.,0]),np.diag([1.,1.,-2.])]
    for i,j in [(0,1),(0,2),(1,2)]:
        e=np.zeros((3,3)); e[i,j]=e[j,i]=1;basis.append(e)
    out=[]
    for n in mark['sizes']:
        g=Grid(n,mark['L'])
        columns=np.stack([bump_seed(g,e)[0].ravel() for e in basis],axis=1)
        for j in [0,1]:
            with records.load(base/f'cr{n}_{j}.npz') as c:
                q,p=decode(g,c)
                bad=decode(g,dict(positive=c['positive'],horizontal=-c['horizontal']))[1]
            a=np.linalg.lstsq(columns,p.ravel()/mark['eps'],rcond=None)[0]
            pred=(columns@a).reshape(p.shape)*mark['eps']
            residual=np.linalg.norm(pred-p)/np.linalg.norm(p)
            out.append(dict(n=n,j=j,coeff=a.real.tolist(),relative=residual,
                            q=float(np.max(abs(q))),bad=float(np.linalg.norm(bad-p)/np.linalg.norm(p))))
            assert residual<1e-11 and np.max(abs(a.imag))<1e-11
            if n==max(mark['sizes']):
                E=np.einsum('i,ijk->jk',a.real,basis)
                # Save continuum root functions: the common radial factor is
                # gain*r^4 times the radial Fourier transform of the compact bump.
                # At unit r the angular coefficients below are the actual paired roots.
                dirs=Grid(6,4).nvec.reshape(-1,3)
                axes=np.eye(3)[np.argmin(abs(dirs),axis=-1)]
                v=np.cross(axes,dirs);v/=np.linalg.norm(v,axis=-1)[:,None]
                z=v+1j*np.cross(dirs,v)
                pos=[];neg=[]
                for Z in [z,z.conj()]:
                    pos.append(np.einsum('ai,ij,aj->a',Z,E,Z)/4)
                    neg.append(np.einsum('ai,ij,aj->a',Z.conj(),E,Z.conj()))
                np.savez(f'roots{j}.npz',dirs=dirs,positive=np.array(pos).T,horizontal=np.array(neg).T)
    return out

def seed_from_roots(x,index):
    # The homogeneous quartic symbol recovered below acts as a local differential
    # operator on b. This gives its inverse transform without periodizing space.
    with np.load(f'/up/roots{index}.npz') as c:
        n=c['dirs'];axes=np.eye(3)[np.argmin(abs(n),axis=-1)]
        v=np.cross(axes,n);v/=np.linalg.norm(v,axis=-1)[:,None]
        z=v+1j*np.cross(n,v)
        class Frame:
            r=np.ones(len(n))
            def frames(self):return z
        q,s=decode(Frame(),c)
    quads=list(combinations_with_replacement(range(3),4))
    design=np.stack([np.prod(n[:,idx],axis=1) for idx in quads],axis=-1)
    coeff,_,rank,_=np.linalg.lstsq(design,s.real.reshape(-1,9),rcond=None)
    assert rank==15 and np.max(abs(design@coeff-s.reshape(-1,9)))<1e-11
    y=x/cfg['radius']; t=np.sum(y*y,axis=-1)
    u=np.zeros_like(t); mask=t<1;u[mask]=1/(1-t[mask])
    bump=np.zeros_like(t);bump[mask]=np.exp(cfg['bump_shape']*(1-u[mask]))
    poly=Polynomial([1]);der=[]
    for _ in range(4):
        poly=Polynomial([0,0,1])*(poly.deriv()-cfg['bump_shape']*poly);der.append(bump*poly(u))
    fields=[]
    for idx in quads:
        d=16*der[3]*np.prod(y[...,idx],axis=-1)
        for i in range(4):
            for j in range(i):
                if idx[i]==idx[j]:
                    rest=[idx[k] for k in range(4) if k not in (i,j)]
                    d+=8*der[2]*y[...,rest[0]]*y[...,rest[1]]
        for j,k,l in [(1,2,3),(2,1,3),(3,1,2)]:
            d+=4*der[1]*(idx[0]==idx[j])*(idx[k]==idx[l])
        fields.append(d)
    return (np.stack(fields,axis=-1)@coeff).reshape(x.shape[:-1]+(3,3))*cfg['gain']/cfg['radius']**4
