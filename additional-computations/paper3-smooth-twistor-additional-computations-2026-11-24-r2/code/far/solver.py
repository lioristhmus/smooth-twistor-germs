"""Independent elliptic solution in the q=0 chart, with scaled unknowns at small amplitude."""
import json
from pathlib import Path
import numpy as np
from seed import field
from multipoles import Grid

def sup(x):return float(np.max(np.abs(x)))

def york(grid,force):
    # G_ij=−(7δ_ij/r+x_i x_j/r³)/(32π). Write
    # V=−P f + (1/8)grad[x.P f−P(y.f)], P=(−Δ)^−1.
    packed=np.concatenate((force,np.sum(grid.x*force,axis=-1,keepdims=True)),axis=-1)
    pm,pr=grid.potential(grid.modes(packed))
    p=grid.sample(pm)[...,:3]
    grad=grid.gradient(pm,pr)
    v=-p+(p+np.einsum('rai,raij->raj',grid.x,grad[...,:3,:])-grad[...,3,:])/8
    m=grid.modes(v)
    dv=grid.gradient(m,grid.radial_diff(m))
    strain=(dv+dv.swapaxes(-1,-2)-
            (2/3)*np.trace(dv,axis1=-2,axis2=-1)[...,None,None]*np.eye(3))
    return m,strain

def sphere_read(grid,radii,kernels,source_m,amp2):
    modes,rad=grid.potential(source_m,kernels)
    z=grid.sample(modes)[...,0]
    dz=grid.sample(rad)[...,0]
    w=amp2*z
    # Evaluate u/ε² and its derivative without subtracting 1 from ψ⁴.
    scaled=z*(4+w*(6+w*(4+w)))
    weights=grid.dw/(4*np.pi)
    means=(scaled@weights)*radii
    flux=(-4*(1+w)**3*dz@weights)*radii**2
    return means,flux

def modal_residual(g,source,rad,m):
    ell=np.repeat(np.arange(g.ell+1),2*np.arange(g.ell+1)+1)
    lap=g.radial_diff(rad)+2*rad/g.r[:,None,None]-m*(
        ell*(ell+1))[None,:,None]/g.r[:,None,None]**2
    return g.sample(lap)+source

def rms(g,arr):
    if arr.ndim==2:arr=arr[...,None]
    return float(np.sqrt(np.einsum('rac,a,r->',arr**2,g.dw,g.dr*g.r**2)))

def save_record(g,key,arrays):
    # One independent small file per radial slice, preserving all retained modes.
    dest=Path(key+'-fields')
    # Runner exports flat filenames; use key as filename prefix instead of nesting.
    for i in range(len(g.r)):
        np.savez_compressed(f'{dest.name}-{i:03}.npz',**{k:v[i] for k,v in arrays.items()})
    np.savez_compressed(key+'-grid.npz',r=g.r,weights=g.dr,edges=g.edges,
                        ell=g.ell,unit_directions=g.n,angular_weights=g.dw)

def solve(g,cfg,eps,tol,multiplier=1.,zero=False,save=None):
    p=field(g.x,cfg)
    if zero:p*=0
    amp2=eps*eps
    z=np.zeros(p.shape[:2]);grad=np.zeros(p.shape[:2]+(3,));strain=np.zeros_like(p)
    hist=[]
    for it in range(cfg['max_iterations']):
        psi=1+amp2*z
        B=p+amp2*strain
        rho=multiplier*np.sum(B**2,axis=(-2,-1))/(8*psi**3)
        force=-2*np.einsum('raij,raj->rai',B,grad)/psi[...,None]
        rm=g.modes(rho[...,None])
        m,rad=g.potential(rm)
        znew=g.sample(m)[...,0]
        gradnew=g.gradient(m,rad)[...,0,:]
        vm,strain_new=york(g,force)
        change=max(sup(znew-z),sup(gradnew-grad),sup(strain_new-strain))
        hist.append(change)
        z,grad,strain=znew,gradnew,strain_new
        if change<=tol:break
    else:raise RuntimeError(('Picard iteration',eps,tol,hist))
    radii=np.array(cfg['radii'])
    if not hasattr(g,'shells'):
        g.shells=g.integration(radii)
        delta=1e-4
        g.fd_shells=[g.integration(radii*(1+s*delta)) for s in [-1,1]]
    mean,flux=sphere_read(g,radii,g.shells,rm,amp2)
    # A second flux obtained solely from neighbouring scalar-field samples.
    left,right=[sphere_read(g,radii*(1+s*1e-4),ker,rm,amp2)[0]
                for s,ker in zip([-1,1],g.fd_shells)]
    fd=-(right/(1+1e-4)-left/(1-1e-4))/(2e-4)
    source_mass=float(np.einsum('ra,a,r->',rho,g.dw,g.dr*g.r**2)/np.pi)
    estimate=float(2*mean[-1]-mean[-2])
    # Independent differential diagnostic for the York equation in Cartesian components.
    dvm=g.radial_diff(vm)
    vv=g.sample(vm)
    div=np.trace(g.gradient(vm,dvm),axis1=-2,axis2=-1)[...,None]
    dm=g.modes(div)
    grad_div=g.gradient(dm,g.radial_diff(dm))[...,0,:]
    york_defect=modal_residual(g,-force,dvm,vm)+grad_div/3
    scalar_defect=modal_residual(g,rho[...,None],rad,m)
    if save:
        save_record(g,save,dict(z=m[...,0],radial_z=rad[...,0],V=vm,
                                rho=rm[...,0],force=g.modes(force)))
    return dict(eps=eps,zero=zero,multiplier=multiplier,tolerance=tol,iterations=hist,
                mean_ratio=mean.tolist(),flux_ratio=flux.tolist(),finite_difference_flux_ratio=fd.tolist(),
                extrapolated_ratio=estimate,extrapolated_flux_ratio=float(2*flux[-1]-flux[-2]),
                coefficient=amp2*estimate,source_mass_ratio=source_mass,
                max_W=eps**3*sup(vv),max_u=amp2*sup(z*(4+amp2*z*(6+amp2*z*(4+amp2*z)))),
                max_LW=eps**3*sup(strain),scalar_L2_relative=rms(g,scalar_defect)/max(rms(g,rho),1e-200),
                york_L2_relative=rms(g,york_defect)/max(rms(g,force),1e-200),
                force_L2_scaled=rms(g,force),scalar_tail_outer_shell=float(
                    np.einsum('ra,a,r->',rho[g.r>.8*g.outer],g.dw,(g.dr*g.r**2)[g.r>.8*g.outer])/np.pi))
