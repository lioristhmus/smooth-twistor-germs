"""A specified smooth continuum representative of the stored two-root data."""
from pathlib import Path
import numpy as np
from scipy.special import spherical_jn
from numpy.polynomial.legendre import leggauss
from multipoles import Grid
from read_cr import decode
from modal import transition
from pp import frame, wave_read, forward, sup

def angular_grid():
    # Only the angular part is needed. Product quadrature resolves harmonics through 14.
    return Grid([1,2,14,1],dict(radius=1))

def write_roots(F,k):
    g=angular_grid()
    tensor=g.sample(np.pad(F,((0,0),(0,(g.ell+1)**2-F.shape[1]),(0,0)))).reshape(len(k),len(g.n),2,3,3)
    z=frame(g.n)
    for i,rad in enumerate(k):
        positive=[];horizontal=[]
        for sign in (1,-1):
            Z=z if sign==1 else z.conj()
            A=tensor[i,:,0]/2+1j*sign/rad*tensor[i,:,1]
            positive.append(np.einsum('aij,ai,aj->a',A,Z,Z)/(4j*sign*rad))
            horizontal.append(np.einsum('aij,ai,aj->a',A,Z.conj(),Z.conj())*rad**3/(1j*sign))
        np.savez_compressed(f'smooth-{i:03}.npz',k=rad,positive=np.array(positive).T,horizontal=np.array(horizontal).T)

def read_coeff(base,k):
    g=angular_grid();z=frame(g.n)
    ell=np.repeat(np.arange(g.ell+1),2*np.arange(g.ell+1)+1)
    class Frame:
        def frames(self):return z
    fr=Frame();coeff=[];reality=0.;tt=0.;interp=0.
    full=Grid([1,2,18,1],dict(radius=1))
    P=np.eye(3)-full.n[:,:,None]*full.n[:,None,:]
    for i,rad in enumerate(k):
        fr.r=np.full(len(z),rad)
        with np.load(base/f'smooth-{i:03}.npz') as d:
            assert rad==d['k']
            q,p=decode(fr,d)
        tensor=np.stack([q,p],axis=1)
        tt=max(tt,sup(np.trace(tensor,axis1=-1,axis2=-2)),sup(np.einsum('abij,aj->abi',tensor,g.n)))
        m=g.modes(tensor.reshape(1,len(z),18))[0]
        reality=max(reality,sup(m.conj()-(-1)**ell[:,None]*m))
        interp=max(interp,sup(g.sample(m[None,...])[0]-tensor.reshape(len(z),18)))
        m=.5*(m+m.conj()*(-1)**ell[:,None])
        field=(full.Y[:,:len(m)]@m).reshape(len(P),2,3,3)
        pap=P[:,None]@field@P[:,None]
        physical=pap-.5*np.trace(pap,axis1=-2,axis2=-1)[...,None,None]*P[:,None]
        coeff.append(full.modes(physical.reshape(1,len(P),18))[0])
    return np.array(coeff),dict(root_reality_error=reality,shell_TT_error=tt,angular_interpolation_error=interp)

def integrate(g,k,coeff,order,window=None):
    # Choose a compact C-infinity cardinal interpolation on disjoint radial intervals.
    # Set the first lower node strictly above zero and the last upper node beyond cap.
    edges=np.r_[k[0]/2,k,k[-1]+k[0]]
    v,weights=leggauss(order);delta=np.diff(edges)
    radii=(edges[:-1,None]+delta[:,None]*(1+v[None,:])/2).ravel()
    weights=(delta[:,None]*weights[None,:]/2).ravel()
    if window:
        weights*=np.exp(-(radii/(.43*window))**2)*(1-transition((radii/window-.7)/.3)[0])
    factor=transition((v+1)/2)[0]
    c=np.pad(coeff,((1,1),(0,0),(0,0)))
    modes=np.empty((len(g.r),(g.ell+1)**2,18),complex)
    for l in range(g.ell+1):
        sl=slice(l*l,(l+1)**2)
        # Contract the Bessel quadrature onto the two endpoint coefficients first.
        J=spherical_jn(l,g.r[:,None]*radii)*(weights*radii**2)
        by=J.reshape(len(g.r),len(delta),order)
        mat=np.pad(by@(1-factor),((0,0),(0,1)))+np.pad(by@factor,((0,0),(1,0)))
        modes[:,sl]=((1j**l/(2*np.pi**2))*(mat@c[:,sl].reshape(len(c),-1))).reshape(len(g.r),2*l+1,18)
    # The smooth representative after re-projection has degree <=18. Evaluate its full angular expansion
    # on the configured physical grid; its differential operators use degree 10.
    from scipy.special import sph_harm_y
    theta=np.arccos(g.n[:,2]);az=np.arctan2(g.n[:,1],g.n[:,0])
    out=g.sample(modes).real
    for l in range(g.ell+1,int(np.sqrt(coeff.shape[1]))):
        sl=slice(l*l,(l+1)**2)
        by=(spherical_jn(l,g.r[:,None]*radii)*(weights*radii**2)).reshape(len(g.r),len(delta),order)
        mat=np.pad(by@(1-factor),((0,0),(0,1)))+np.pad(by@factor,((0,0),(1,0)))
        extra=((1j**l/(2*np.pi**2))*(mat@c[:,sl].reshape(len(c),-1))).reshape(len(g.r),2*l+1,18)
        Y=[]
        for m in range(-l,l+1):
            y=sph_harm_y(l,abs(m),theta,az)
            Y.append(y.real if m==0 else np.sqrt(2)*(y.imag if m<0 else y.real))
        out+=np.einsum('am,rmc->rac',np.array(Y).T,extra,optimize=True).real
    return [a.reshape(len(g.r),len(g.n),3,3) for a in np.split(out,2,axis=-1)]

def run(base,window=None):
    import json
    meta=json.loads((base/'results.json').read_text())['32']
    k=np.array(meta['k']);g=Grid(meta['spec'],dict(radius=1))
    c,res=read_coeff(base,k)
    d=g.sample(np.stack([np.load(f)['X'] for f in sorted(base.glob('band32-mark-*.npz'))]))
    prev=None;diffs=[]
    for order in [4,8,16]:
        q,p=integrate(g,k,c,order,window)
        if prev is not None:diffs.append([order,sup(q-prev[0]),sup(p-prev[1])])
        prev=q,p
    h,K,hist=forward(g,q,p)
    res.update(radial_quadrature=diffs,readout=wave_read(g,h,K,d),defects=hist,
               window=window,shells=len(k),support=[float(k[0]/2),float(k[-1]+k[0])])
    return res
