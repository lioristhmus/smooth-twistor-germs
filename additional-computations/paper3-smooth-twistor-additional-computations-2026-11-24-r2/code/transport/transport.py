import json
from pathlib import Path
import numpy as np
from multipoles import Grid
from continuum import cfg,seed_from_roots
from chain import symbols
def sup(x):return float(np.max(abs(x)))
def dot(B,D):return np.sum(B*D,axis=(-1,-2))
def mv(B,v):return np.einsum('...ij,...j->...i',B,v)
def diff(g,m):return g.gradient(m,g.radial_diff(m))
class Evolution:
    def __init__(self,profile):
        self.g=g=Grid(profile,cfg)
        self.p=[seed_from_roots(g.x,i) for i in (0,1)]
        k=np.array([[1.,0.,0.]])
        sc,v=symbols(k)
        self.scalar=-sc.item();self.trans=v[0,1,1];self.long=v[0,0,0]
        self.outer=g.integration([28.,56.])
    def vector(self,f):
        g=self.g
        packed=np.concatenate([f,np.sum(g.x*f,axis=-1,keepdims=True)],axis=-1)
        m,dm=g.potential(g.modes(packed))
        p=g.sample(m)[...,:3];a=g.gradient(m,dm)
        v=self.trans*p+(self.long-self.trans)/2*(p+np.einsum('rai,raij->raj',g.x,a[...,:3,:])-a[...,3,:])
        vm=g.modes(v);dv=diff(g,vm)
        return vm,dv+dv.swapaxes(-1,-2)-2/3*np.trace(dv,axis1=-2,axis2=-1)[...,None,None]*np.eye(3)
    def scalar_field(self,rho):
        g=self.g
        m,dm=g.potential(g.modes(rho[...,None])*self.scalar)
        return g.sample(m)[...,0],g.gradient(m,dm)[...,0,:],m
    def jet(self,S):
        z,grad,m=self.scalar_field(S[...,0])
        v,L=self.vector(S[...,1:])
        return z,grad,m,v,L
    def tangent(self,S,p,ds,e):
        z,grad,_,_,L=self.jet(S)
        psi=1+e*z;B=p+e*L;rho=S[...,0];f=S[...,1:]
        ugrad=np.zeros_like(grad);dz=np.zeros_like(z);dv=0.
        for _ in range(14):
            dB=ds+e*dv
            dr=.25*dot(B,dB)/psi**3+e*(5*rho/psi-dot(B,B)/psi**4)*dz
            nz,ng,_=self.scalar_field(dr)
            df=-2*(mv(dB,grad)+mv(B,ng))/psi[...,None]+e*(4*f/psi[...,None]+10*mv(B,grad)/psi[...,None]**2)*nz[...,None]
            _,nv=self.vector(df)
            change=max(sup(nz-dz),sup(ng-ugrad),sup(nv-dv))
            dz,ugrad,dv=nz,ng,nv
            if change<2e-13:break
        else:raise RuntimeError('current inverse')
        return np.concatenate([dr[...,None],df],axis=-1)
    def march(self,eps,steps,broken):
        p=0*self.p[0];S=np.zeros(p.shape[:2]+(4,))
        for segment in self.p if broken else [sum(self.p)]:
            ds=segment/steps
            for _ in range(steps):
                mid=S+.5*self.tangent(S,p,ds,eps**2)
                S+=self.tangent(mid,p+.5*ds,ds,eps**2)
                p+=ds
        return S
    def inspect(self,S,eps,save=False):
        g=self.g;e=eps**2
        z,grad,m,v,L=self.jet(S);psi=1+e*z
        A=psi**4;K=eps*(sum(self.p)+e*L)
        mK=g.modes(K.reshape(*z.shape,9))
        dK=diff(g,mK).reshape(*z.shape,3,3,3)
        div=np.einsum('...iji->...j',dK)
        _,Lread=self.vector(div)
        decoded=K-Lread
        lap=g.radial_diff(g.radial_diff(m))+2*g.radial_diff(m)/g.r[:,None,None]
        ell=np.repeat(np.arange(g.ell+1),2*np.arange(g.ell+1)+1)
        lap-=m*(ell*(ell+1))[None,:,None]/g.r[:,None,None]**2
        H=-8*e*g.sample(lap)[...,0]/psi**5-dot(K,K)/psi**8
        M=(div+2*e*mv(K,grad)/psi[...,None])/psi[...,None]**4
        far,_=g.potential(g.modes(S[...,0,None]),self.outer)
        w=g.sample(far)[...,0]*e
        mean=(w*(4+w*(6+w*(4+w))))@(g.dw/(4*np.pi))*[28,56]
        if save:
            for i in range(len(g.r)):
                np.savez_compressed(f'field-{i:03}.npz',z=m[i,:,0],V=v[i],K=mK[i])
        return dict(C=float(2*mean[1]-mean[0]),H=sup(H),M=sup(M),trace=sup(np.trace(K,axis1=-2,axis2=-1)/A),
               seed=sup(decoded-eps*sum(self.p)),seed_norm=sup(eps*sum(self.p)),maxu=sup(A-1)),m,v,A,K
def run(stage):
    name={'pilot':'coarse','outer':'exterior'}.get(stage,stage)
    op=Evolution(cfg['profiles'][name])
    eps=.008
    ref=json.loads((Path('/far')/name/'results.json').read_text())['rows']
    expected=next(r for r in ref if r['eps']==eps and not r['zero'] and r['multiplier']==1)
    rows=[]
    stages=[1] if stage=='pilot' else ([2,4,8] if stage=='medium' else [4])
    for steps in stages:
        ends=[]
        for broken in (False,True):
            S=op.march(eps,steps,broken)
            r,m,v,A,K=op.inspect(S,eps,stage=='fine' and not broken)
            ends.append((A,K));r.update(steps=steps,broken=broken,Cref=expected['coefficient'],ratio=r['C']/eps**2)
            if stage=='fine':
                dirs=sorted(Path('/far/fine').glob('selected-fields*.npz'))
                zz=[];vv=[]
                for f in dirs:
                    with np.load(f) as d:zz.append(d['z']);vv.append(d['V'])
                vm=np.stack(vv);zm=np.stack(zz)[...,None]
                dv=diff(op.g,vm)
                refK=eps*(sum(op.p)+eps**2*(dv+dv.swapaxes(-1,-2)-2/3*np.trace(dv,axis1=-2,axis2=-1)[...,None,None]*np.eye(3)))
                r.update(ref_h=sup(A-(1+eps**2*op.g.sample(zm)[...,0])**4),ref_K=sup(K-refK))
            rows.append(r)
        rows[-1].update(path_h=sup(ends[0][0]-ends[1][0]),path_K=sup(ends[0][1]-ends[1][1]))
    return rows
