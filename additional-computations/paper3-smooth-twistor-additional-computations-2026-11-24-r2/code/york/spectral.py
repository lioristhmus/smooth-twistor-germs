"""Antiperiodic box quadrature: k_j=2π(m_j+1/2)/L. All perturbations are real at nodes."""
import numpy as np
class Grid:
    def __init__(self,n,L):
        self.n=n;self.L=L
        self.x=np.stack(np.meshgrid(*([(np.arange(n)-n//2)*L/n]*3),indexing='ij'),axis=-1)
        v=2*np.pi*(np.fft.fftfreq(n)*n+.5)/L
        self.k=np.stack(np.meshgrid(v,v,v,indexing='ij'),axis=-1);self.r=np.linalg.norm(self.k,axis=-1)
        self.nvec=self.k/self.r[...,None]
        self.phase=np.exp(1j*np.pi*self.x.sum(axis=-1)/L)
    def extra(self,val,x):return val[(...,)+(None,)*(x.ndim-3)]
    def fft(self,x):
        a=x*self.extra(self.phase.conj(),x)
        return np.fft.fftn(np.fft.ifftshift(a,axes=(0,1,2)),axes=(0,1,2))/self.n**3
    def inv(self,v):
        a=np.fft.fftshift(np.fft.ifftn(v,axes=(0,1,2)),axes=(0,1,2))
        return a*self.extra(self.phase,v)*self.n**3
    def grad(self,x,i):
        v=self.fft(x);return self.inv(v*self.extra(1j*self.k[...,i],v)).real
    def frames(self):
        n=self.nvec
        ref=np.eye(3)[np.argmin(abs(n),axis=-1)]
        e=np.cross(ref,n);e/=np.linalg.norm(e,axis=-1)[...,None]
        z=e+1j*np.cross(n,e)
        return z
def outer(v):return v[...,None]*v[...,None,:]
def maxabs(x):return float(np.max(abs(x)))
def bump_seed(g,E):
    R=1.75;c=5.;gain=.008
    t=np.sum(g.x*g.x,axis=-1)/R**2;b=np.zeros_like(t);m=t<1;b[m]=np.exp(c-c/(1-t[m]))
    k=g.k
    D=outer(k)-g.r[...,None,None]**2*np.eye(3)
    four=(D@E@D-.5*D*np.einsum('...ij,ij->...',D,E)[...,None,None])*g.fft(b)[...,None,None]
    return gain*four,gain*6*8*(c*c/2-c)/R**4*E
