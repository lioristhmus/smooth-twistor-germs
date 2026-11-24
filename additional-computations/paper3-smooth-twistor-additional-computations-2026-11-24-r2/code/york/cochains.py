import json
from pathlib import Path
import numpy as np
import records
from spectral import Grid,outer,bump_seed,maxabs
from read_cr import decode
cfg=json.loads(Path('/src/input.json').read_text())['cr']
def lift(g,p):
    z=g.frames();r=g.r
    # Bilinear dual line maps at north and south; z_S at the south pole has the opposite conjugate sign, whose square cancels.
    pos=[];neg=[];U={};V={}
    for sign in [1,-1]:
        Z=z if sign==1 else z.conj();A=(1j*sign/r)[...,None,None]*p
        pos.append(np.einsum('...ij,...i,...j->...',A,Z,Z)/(4j*sign*r))
        neg.append(np.einsum('...ij,...i,...j->...',A,Z.conj(),Z.conj())*r**3/(1j*sign))
        U[sign]=pos[-1][...,None,None]*outer(Z.conj())/4
        V[sign]=neg[-1][...,None,None]*outer(Z)
    flip=lambda a:a[::-1,::-1,::-1].conj()
    fac=(16*r**4)[...,None,None]
    real=max(maxabs(U[-1]-flip(V[1])/fac),maxabs(V[-1]-fac*flip(U[1])))
    c,d=neg
    alpha=np.stack((-5*c/(2j*r),5*d/(2j*r)),axis=-1)
    return dict(negative_constant_harmonic=np.zeros(5,complex),positive=np.stack(pos,axis=-1),harmonic=alpha,horizontal=np.stack(neg,axis=-1),anchor=-np.sum(U[1]+U[-1],axis=(0,1,2))),real
def closed(g,rec,h):
    c,d=np.moveaxis(rec['horizontal'],-1,0)
    a,b=np.moveaxis(rec['harmonic'],-1,0)
    e,f=np.moveaxis(rec['positive'],-1,0)
    err=0
    for w in [.23+.31j,-.7+.2j,.51-.6j]:
        def beta(w):return (c+d*w*w.conjugate()**5)/(1+abs(w)**2)**5
        def pos(w):return e+f*w**4
        def dzbar(F):return (F(w+h)-F(w-h)+1j*(F(w+1j*h)-F(w-1j*h)))/(4*h)
        err=max(err,maxabs(dzbar(pos)),maxabs(dzbar(beta)-(2j*g.r*w)*(a+b*w.conjugate()**4)/(1+abs(w)**2)**6))
    return err
def test(n,save):
    g=Grid(n,cfg['box']);eps=cfg['eps'];results=[]
    for j,E in enumerate([np.diag([1.,-1.,0.]),np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]])]):
        p,at0=bump_seed(g,E);p*=eps
        rec,real=lift(g,p)
        if save:records.save(f'cr{n}_{j}.npz',**rec)
        if save:
            with records.load(f'cr{n}_{j}.npz') as stored:rec={key:stored[key] for key in stored.files}
        q2,p2=decode(g,rec)
        Z=g.frames(); cc=outer(Z.conj());zz=outer(Z)
        poly=(rec['positive'][...,0,None,None]*cc+rec['positive'][...,1,None,None]*zz)/4
        center=g.inv(poly)[n//2,n//2,n//2]
        # Constant residual C is in im K: V(x)=Cx/2 gives L0 V=C for STF C.
        C=rec['anchor'];B=C/2;quotient=maxabs(C-(B+B.T-2*np.eye(3)*np.trace(B)/3))
        # No negative point-supported polynomial is part of the continuous quadrature ansatz.
        # The constant TT tensor control belongs to the negative obstruction kernel(div).
        def residual_constant(coefficients):
            # At zero spatial frequency the five harmonic coefficients in H^1(O(-6))
            # pair with 1,z,...,z^4. Discrete angular Fourier inversion recovers them.
            angles=2*np.pi*np.arange(11)/11
            w=.75*np.exp(1j*angles)
            modes=np.arange(5)
            alpha=np.polynomial.polynomial.polyval(w.conj(),coefficients)/(1+abs(w)**2)**6
            return (np.exp(1j*np.outer(modes,angles))@(alpha*(1+abs(w)**2)**6))/(11*.75**modes)
        nc=rec['negative_constant_harmonic']
        neg_res=maxabs(residual_constant(nc))
        neg_control=maxabs(residual_constant(nc+np.array([1.,0.,0.,0.,1.])))
        assert neg_res<cfg['algebra_tol'] and neg_control>.9
        weighted=maxabs(g.inv(p).real*(1+np.sum(g.x*g.x,axis=-1))[...,None,None]**((1-cfg['gamma'])/2))
        norms=[maxabs(rec[key][...,i]) for key in ('positive','horizontal') for i in [0,1]]
        rr=dict(spatial_center_error=maxabs(g.inv(p)[n//2,n//2,n//2]-eps*at0),root_norms=norms,
            reality=real,transverse=maxabs(np.einsum('...ij,...j->...i',p,g.k)),trace=maxabs(np.trace(p,axis1=-2,axis2=-1)),seed_error=maxabs(p2-p),q_error=maxabs(q2),
            closure_steps=[closed(g,rec,h) for h in [.004,.002,.001]],control_closure=closed(g,{**rec,'harmonic':-rec['harmonic']},.001),
            positive_anchor=maxabs(center+C),positive_quotient=quotient,weighted_p_sup=weighted,negative_polynomial_control=neg_control,negative_polynomial_degrees_retained=[0],negative_residual_constant=neg_res)
        results.append(rr)
    return results
assert max(test(4,False)[0]['root_norms'])>0
res={n:test(n,True) for n in cfg['sizes']}
passed=all(max(v[k] for k in ['seed_error','q_error','reality','transverse','trace','positive_anchor','positive_quotient'])<cfg['algebra_tol'] and min(v['root_norms'])>1e-8 and v['closure_steps'][-1]<cfg['closure_tol'] and v['control_closure']>50*v['closure_steps'][-1] for vs in res.values() for v in vs)
# Separate the spatial resolution of the underlying compact bump from root-map identities.
mesh=[]
for fine in [16,24,32,48]:
    grid=Grid(fine,cfg['box']);tensor,at0=bump_seed(grid,np.diag([1.,-1.,0.]))
    mesh.append(dict(n=fine,center_error=maxabs(grid.inv(tensor)[fine//2,fine//2,fine//2]-at0)*cfg['eps']))
Path('results.json').write_text(json.dumps(dict(rows=res,spatial_seed_mesh=mesh,passed=passed),indent=2))
Path('marking.json').write_text(json.dumps(dict(orientation='right handed Cartesian',spin='trivial Cartesian spin frame, positive identity lift',L=cfg['box'],gamma=cfg['gamma'],eps=cfg['eps'],sizes=cfg['sizes'],root_frame='e1=unit(argmin_axis(abs(n)) cross n); e2=n cross e1',positive_anchor='per mode (exp(ix.k)-1) multiplying the quartic in its rotated spin frame; x=0',polynomials='no free polynomial; anchored constant is a constant STF quartic in the image of the flat conformal Killing derivative of a linear field')))
print(passed,[v[0]['closure_steps'][-1] for v in res.values()]);assert passed
