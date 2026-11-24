"""Finite current-linearization transport on the q=0 two-polarization section."""
import json
from pathlib import Path
import numpy as np
import records
from chain import symbols
from read_cr import decode
from spectral import Grid,maxabs
cfg=json.loads(Path('/src/input.json').read_text())['transport']
import sys
tight=Path(sys.argv[0]).stem=='transport_tight'
solve_tolerance=1e-15 if tight else 2e-12
cr=Path('/in/cochains')
assert json.loads((cr/'results.json').read_text())['passed']
mark=json.loads((cr/'marking.json').read_text())
if tight:
    cfg={**cfg,'epsilons':[min(cfg['epsilons'])]}
    mark={**mark,'sizes':[cfg['size']]}
assert mark['spin']=='trivial Cartesian spin frame, positive identity lift'
def dot(a,b):return np.sum(a*b,axis=-1)
def mat(a):return a[...,None,None]
def vec(a):return a[...,None]
class Current:
    def __init__(self,g):
        self.g=g;self.N,self.V=symbols(g.k)
    def jet(self,p,y):
        g=self.g;k=g.k;v=g.fft(y);A=1+y[...,0];Ah=v[...,0];Wh=v[...,1:]
        grad=g.inv(1j*k*vec(Ah)).real
        lap=g.inv(-g.r**2*Ah).real
        wdot=dot(k,Wh)
        LW=1j*(k[...,None]*Wh[...,None,:]+Wh[...,None]*k[...,None,:]-2/3*mat(wdot)*np.eye(3))
        K=p+g.inv(LW).real
        mom=g.inv(-vec(g.r**2)*Wh-k*vec(wdot)/3).real
        return A,grad,lap,K,mom
    def residual(self,j):
        A,d,l,K,m=j
        H=-2*l/A**2+1.5*dot(d,d)/A**3-np.sum(K*K,axis=(-2,-1))/A**2
        M=m/vec(A)+np.einsum('...ij,...j->...i',K,d)/vec(2*A**2)
        return np.concatenate((vec(H),M),axis=-1)
    def tangent(self,j,dj):
        A,d,l,K,m=j;u,g,h,B,w=dj
        H=(-2*h/A**2+4*l*u/A**3+3*dot(d,g)/A**3-4.5*dot(d,d)*u/A**4
           -2*np.sum(K*B,axis=(-2,-1))/A**2+2*np.sum(K*K,axis=(-2,-1))*u/A**3)
        mul=lambda X,v:np.einsum('...ij,...j->...i',X,v)
        M=w/vec(A)-m*vec(u/A**2)+(mul(B,d)+mul(K,g))/vec(2*A**2)-mul(K,d)*vec(u/A**3)
        return np.concatenate((vec(H),M),axis=-1)
    def green(self,f):
        v=self.g.fft(f)
        u=-.5*self.N*v[...,0];W=np.einsum('...ij,...j->...i',self.V,v[...,1:])
        return self.g.inv(np.concatenate((vec(u),W),axis=-1)).real
    def lift(self,p,y,ds):
        j=self.jet(p,y);A,d,l,K,m=j
        rhs=-self.tangent(j,(0.,d*0,0.,ds,m*0));v=self.green(rhs)
        for count in range(8):
            a,b,c,B,w=self.jet(ds*0,v)
            residual=rhs-self.tangent(j,(v[...,0],b,c,B,w))
            if maxabs(residual)<solve_tolerance:return v
            v+=self.green(residual)
        raise ArithmeticError(('current inverse',maxabs(residual)))
    def march(self,segs,n):
        p=segs[0]*0;y=np.zeros(p.shape[:-2]+(4,))
        for ds in segs:
            inc=ds/n
            for _ in range(n):
                k1=self.lift(p,y,inc)
                y+=self.lift(p+inc/2,y+k1/2,inc)
                p+=inc
        return y
def save_source(stem,op,p,y):
    A,g,l,K,m=op.jet(p,y)
    assert np.min(A)>.9
    b=1/np.sqrt(A); boost=K*mat(b)
    # One scalar b specifies b I. For conformal h, spatial spin-frame connection is (δ_ai grad_b A−δ_ib grad_a A)/(2A).
    grad=g/vec(2*A)
    records.save(stem,b=b,boost=boost,rotation_vector=grad)
    return maxabs(op.residual((A,g,l,K,m)))
rows=[]
for n in mark['sizes']:
    g=Grid(n,mark['L']);op=Current(g)
    ps=[];read=[]
    for j in [0,1]:
        with records.load(cr/f'cr{n}_{j}.npz') as c:q,p=decode(g,c)
        assert maxabs(q)<1e-10
        ps.append(g.inv(p).real);read.append(maxabs(p))
    pairs=[(eps,step) for eps in cfg['epsilons'] for step in cfg['steps']] if n==cfg['size'] else [(cfg['epsilons'][0],cfg['steps'][-1])]
    for eps,nstep in pairs:
        a,b=[v*(eps/mark['eps']) for v in ps];p=a+b
        # High-accuracy discrete comparator uses the same isolated symbol quadrature and nonlinear residual.
        yref=np.zeros(p.shape[:-2]+(4,))
        for j in range(15):
            f=op.residual(op.jet(p,yref))
            if maxabs(f)<solve_tolerance:break
            yref-=op.green(f)
        assert maxabs(f)<solve_tolerance
        if nstep==cfg['steps'][-1]:save_source(f'n{n}_e{eps}_reference',op,p,yref)
        ends=[]
        for name,path in [('straight',[p]),('broken',[a,b])]:
            y=op.march(path,nstep);ends.append(y)
            stem=f'n{n}_e{eps}_s{nstep}_{name}'
            defect=save_source(stem,op,p,y)
            rows.append(dict(n=n,eps=eps,steps=nstep,path=name,file=stem+'.npz',defect=defect,endpoint_error=maxabs(y-yref),nonlinear_correction=maxabs(yref),reference_floor=maxabs(f)))
        rows[-1]['path_difference']=maxabs(ends[1]-ends[0])
Path('results.json').write_text(json.dumps(rows,indent=2))
Path('marking.json').write_text(json.dumps(mark,indent=2))
print(len(rows),'paths')
