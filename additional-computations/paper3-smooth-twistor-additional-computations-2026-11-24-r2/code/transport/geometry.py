"""Finite radial and spherical charts for protected maximal data."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import sph_harm_y
from multipoles import Grid
I=np.eye(3)
def sup(a):return float(np.max(abs(a)))
def symmetric(a):return .5*(a+a.swapaxes(-1,-2))
def tf(a):return a-np.trace(a,axis1=-2,axis2=-1)[...,None,None]*I/3
def grad(g,a):
    orig=a.shape
    m=g.modes(a.reshape(*orig[:2],-1))
    return g.gradient(m,g.radial_diff(m)).reshape(*orig,3)
def divergence(g,k):return np.einsum('...iji->...j',grad(g,k))
def geometry(g,h,K):
    # Γ^a_ij and its Cartesian derivatives in the fixed physical frame.
    dh=grad(g,h-I)
    Glo=(np.einsum('...aji->...aij',dh)+np.einsum('...aij->...aij',dh)-np.einsum('...ija->...aij',dh))/2
    inv=np.linalg.inv(h);G=np.einsum('...ab,...bij->...aij',inv,Glo)
    dG=grad(g,G)
    ric=(np.einsum('...kijk->...ij',dG)-np.einsum('...kikj->...ij',dG)
         +np.einsum('...kkl,...lij->...ij',G,G)-np.einsum('...kjl,...lik->...ij',G,G))
    raised=inv@K
    H=np.sum(inv*ric,axis=(-2,-1))-np.einsum('...ij,...ji->...',raised,raised)
    dK=grad(g,K)
    cov=dK-np.einsum('...lik,...lj->...ijk',G,K)-np.einsum('...ljk,...il->...ijk',G,K)
    M=np.einsum('...ik,...ijk->...j',inv,cov)
    electric=ric-K@inv@K  # Gauss readout at maximality, Ric−K_i^k K_kj + (trK)K.
    return H,M,ric,electric
def profile(u,amp):
    u=np.asarray(u)
    return amp*np.exp(-1/np.maximum(u,.0001))*(u>0)
def slice_values(z,amp):
    # In Rosen coordinates 2du dv−a²dX²−b²dY², v'=−(ab)²/2.
    def fun(zz,y):
        u,v,a,b,da,db=y
        uz=np.sqrt((a*a+b*b)/2)/(a*b);H=profile(u,amp)
        return uz*np.array([1,-(a*b)**2/2,da,db,H*a,-H*b])
    sol=solve_ivp(fun,(0,3.5),[0,0,1,1,0,0],rtol=3e-13,atol=3e-15,dense_output=True)
    y=sol.sol(np.clip(z,0,3.5).ravel()).reshape((6,)+z.shape)
    u,v,a,b,da,db=y
    u=np.where(z<0,z,u);v=np.where(z<0,-z/2,v)
    diag=np.stack([a*a,b*b,(a*a+b*b)/2],axis=-1)
    first=a*da/(a*b);second=b*db/(a*b)
    ks=np.stack([first,second,-diag[...,2]*(first/a**2+second/b**2)],axis=-1)
    return diag[...,None]*I,ks[...,None]*I,u,v
def cutoff(r):
    t=np.clip((r-1.15)/1.65,0,1)
    l=np.exp(-1/np.maximum(t,1e-100));h=np.exp(-1/np.maximum(1-t,1e-100))
    return h/(h+l)
def gram(g):
    n=g.n;theta=np.arccos(n[:,2]);phi=np.arctan2(n[:,1],n[:,0])
    co=np.cos(theta);si=np.sin(theta)
    et=np.stack([co*np.cos(phi),co*np.sin(phi),-si],axis=-1)
    ep=np.stack([-np.sin(phi),np.cos(phi),si*0],axis=-1)
    hh=[]
    for l in range(g.ell+1):
        for m in range(-l,l+1):
            val,d,h=sph_harm_y(l,abs(m),theta,phi,diff_n=2)
            def real(x):return x.real if m==0 else np.sqrt(2)*(x.imag if m<0 else x.real)
            a=real(h[:,0,0]);b=real(h[:,1,1])/si**2+co/si*real(d[:,0])
            c=(real(h[:,0,1])-co/si*real(d[:,1]))/si
            outer=lambda u,v:u[...,None]*v[...,None,:]
            hh.append(.5*(a-b)[:,None,None]*(outer(et,et)-outer(ep,ep))+c[:,None,None]*(outer(et,ep)+outer(ep,et)))
    return np.array(hh).reshape(len(hh),-1)
class Protected:
    def __init__(self,g):
        self.g=g;self.hs=gram(g)
        self.lam=np.repeat(np.arange(g.ell+1),2*np.arange(g.ell+1)+1)
        self.cache={}
    def Volterra(self,d):
        if d not in self.cache:
            g=self.g;deg=len(g.v);out=[]
            from numpy.polynomial.legendre import legvander
            for R in g.r:
                row=g.dr*np.minimum(g.r/R,1)**d*(g.r<R)*(g.r>=1)
                j=np.searchsorted(g.edges,R)-1
                sl=slice(j*deg,(j+1)*deg)
                if g.edges[j]>=1-1e-12:
                    left=g.edges[j];h=(R-left)/2;nodes=left+h*(1+g.v)
                    interp=legvander((nodes-g.mid[j])/g.half[j],deg-1)@g.basis_inv
                    row[sl]=(h*g.w*(nodes/R)**d)@interp
                else:row[:]=0
                out.append(row)
            self.cache[d]=np.array(out)
        return self.cache[d]
    def solve(self,H,M):
        g=self.g;n=g.n;nn=n[...,None]*n[...,None,:];tang=I-nn
        # DR_flat(a dr²)=(2/r)a'+(2−Δ_S)a/r².
        hm=g.modes(H[...,None])[...,0]
        a=np.empty_like(hm)
        for l in range(g.ell+1):
            s=slice(l*l,(l+1)**2)
            a[:,s]=self.Volterra(1+l*(l+1)/2)@(g.r[:,None]*hm[:,s]/2)
        metric=(a@g.Y.T)[...,None,None]*nn
        fr=np.sum(n*M,axis=-1);ft=M-fr[...,None]*n
        # div_S b for a tangent vector, computed in Cartesian components.
        def sphere_div(v):
            m=g.modes(v)
            return sum(np.einsum('ah,rh->ra',g.dY[j],m[...,j],optimize=True) for j in range(3))
        fm=g.modes(fr[...,None])[...,0];cm=g.modes(sphere_div(ft)[...,None])[...,0]
        V3=self.Volterra(3)
        aa=.5*(self.Volterra(4)@(fm[:,1:4]+cm[:,1:4])+self.Volterra(2)@(fm[:,1:4]-cm[:,1:4]))
        d1=np.stack([aa@dy[:,1:4].T for dy in g.dY],axis=-1)
        b=(V3@(ft+d1/(2*g.r[:,None,None])).reshape(len(g.r),-1)).reshape(ft.shape)
        am=V3@(fm-g.modes(sphere_div(b)[...,None])[...,0]/g.r[:,None])
        am[:,1:4]=aa
        chi=np.zeros_like(am);chi[:,4:]=am[:,4:]/(2-self.lam[4:]*(self.lam[4:]+1))
        angular=(chi@self.hs).reshape(*fr.shape,3,3)
        mom=(am@g.Y.T)[...,None,None]*(nn-.5*tang)+b[...,None]*n[...,None,:]+n[...,None]*b[...,None,:]+angular
        return metric,mom
def vacuum_extension(spec,amp,iterations=4):
    g=Grid(spec,dict(radius=1));op=Protected(g)
    orig,K0,u,v=slice_values(g.x[...,2],amp)
    cut=cutoff(g.r)[:,None,None,None]
    h0=I+cut*(orig-I)
    k0=cut*(K0-(np.trace(K0,axis1=-2,axis2=-1)/np.trace(orig,axis1=-2,axis2=-1))[...,None,None]*orig)
    h=h0.copy();kap=k0.copy();hist=[]
    mask=g.r>1.15
    for i in range(iterations+1):
        inv=np.linalg.inv(h)
        K=kap-(np.sum(inv*kap,axis=(-2,-1))/3)[...,None,None]*h
        H,M,ric,E=geometry(g,h,K)
        hist.append([sup(H[mask]),sup(M[mask])])
        if i==iterations:break
        # Separate the exact flat linearization of the correction, leaving
        # the discrete residual floor out of the next Picard update.
        dh=h-h0
        d=divergence(g,dh)-grad(g,np.trace(dh,axis1=-2,axis2=-1))
        DR=np.trace(grad(g,d),axis1=-2,axis2=-1)
        V=divergence(g,kap-k0)
        da,dk=op.solve(DR-H,V-M)
        h=h0+da;kap=k0+dk
    return g,h,K,hist,orig,K0
