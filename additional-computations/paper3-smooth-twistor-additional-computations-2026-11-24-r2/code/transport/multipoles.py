"""Real spherical harmonics and open-boundary Poisson integration on Gauss panels."""
import numpy as np
from numpy.polynomial.legendre import leggauss,legvander,legder,legval
from scipy.special import sph_harm_y

class Grid:
    def __init__(self,profile,cfg):
        panels,degree,self.ell,outer=profile
        self.outer=outer
        a=cfg['radius']
        self.edges=list(np.linspace(0,a,panels+1))
        while self.edges[-1]<outer:
            self.edges.append(min(outer,self.edges[-1]+a/panels*max(1,self.edges[-1]/a)))
        self.edges=np.array(self.edges)
        self.v,self.w=leggauss(degree)
        self.basis_inv=np.linalg.inv(legvander(self.v,degree-1))
        self.half=np.diff(self.edges)/2
        self.mid=(self.edges[1:]+self.edges[:-1])/2
        self.r=(self.mid[:,None]+self.half[:,None]*self.v).ravel()
        self.dr=(self.half[:,None]*self.w).ravel()
        self.diff=np.column_stack([legval(self.v,legder(c)) for c in self.basis_inv.T])
        cos_theta,wtheta=leggauss(self.ell+3)
        phi=np.arange(2*self.ell+5)*(2*np.pi/(2*self.ell+5))
        theta,phi=np.meshgrid(np.arccos(cos_theta),phi,indexing='ij')
        self.dw=np.broadcast_to(wtheta[:,None]*2*np.pi/theta.shape[1],theta.shape).flatten()
        theta,phi=theta.ravel(),phi.ravel()
        ct,st,cp,sp=np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
        self.n=np.stack((st*cp,st*sp,ct),axis=-1)
        et=np.stack((ct*cp,ct*sp,-st),axis=-1)
        ep=np.stack((-sp,cp,0*sp),axis=-1)
        vals=[]; derivs=[]
        for l in range(self.ell+1):
            for m in range(-l,l+1):
                v,d=sph_harm_y(l,abs(m),theta,phi,diff_n=1)
                def real(z):
                    if m==0:return z.real
                    return np.sqrt(2)*(z.imag if m<0 else z.real)
                vals.append(real(v))
                derivs.append(et*real(d[:,0,None])+ep*(real(d[:,1])/st)[:,None])
        self.Y=np.array(vals).T.copy()
        self.proj=self.Y.T*self.dw
        self.dY=np.array(derivs).transpose(2,1,0).copy()
        self.x=self.r[:,None,None]*self.n
        self.kernels=self.integration(self.r)
    def modes(self,f):
        return np.einsum('ha,rac->rhc',self.proj,f,optimize=True)
    def sample(self,m):
        return np.einsum('ah,rhc->rac',self.Y,m,optimize=True)
    def gradient(self,m,rad):
        tang=np.stack([np.einsum('ah,rhc->rac',d,m,optimize=True) for d in self.dY],axis=-1)
        return tang/self.r[:,None,None,None]+self.sample(rad)[...,None]*self.n[None,:,None,:]
    def radial_diff(self,m):
        shape=m.shape
        v=m.reshape(len(self.half),len(self.v),-1)
        return ((self.diff@v)/self.half[:,None,None]).reshape(shape)
    def integration(self,targets):
        # Inner and outer integrals of the radial Green kernel. Requadrature on the
        # split panel keeps every power ratio <= 1, including the first radial cell.
        result=[]
        for l in range(self.ell+1):
            matrices=[]
            for R in targets:
                inner=self.r<R
                low=self.dr*(self.r**2/R)*(np.minimum(self.r/R,1)**l)*inner
                high=self.dr*self.r*(np.minimum(R/self.r,1)**l)*(~inner)
                j=np.searchsorted(self.edges,R,side='right')-1
                if 0<=j<len(self.mid):
                    sl=slice(j*len(self.v),(j+1)*len(self.v))
                    for mat,start,end,inside in [(low,self.edges[j],R,True),(high,R,self.edges[j+1],False)]:
                        h=(end-start)/2
                        nodes=(end+start)/2+h*self.v
                        interp=legvander((nodes-self.mid[j])/self.half[j],len(self.v)-1)@self.basis_inv
                        coeff=nodes**2/R*(nodes/R)**l if inside else nodes*(R/nodes)**l
                        mat[sl]=(h*self.w*coeff)@interp
                matrices.append(((low+high)/(2*l+1),(-(l+1)*low+l*high)/((2*l+1)*R)))
            result.append(np.array(matrices).transpose(1,0,2).copy())
        return result
    def potential(self,source,kernels=None):
        kernel=self.kernels if kernels is None else kernels
        out=[];deriv=[]
        for l,(a,b) in enumerate(kernel):
            modes=source[:,l*l:(l+1)**2,:]
            flat=modes.reshape(len(self.r),-1)
            shape=(len(a),2*l+1,source.shape[-1])
            out.append((a@flat).reshape(shape))
            deriv.append((b@flat).reshape(shape))
        return np.concatenate(out,axis=1),np.concatenate(deriv,axis=1)
