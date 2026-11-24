import json,sys
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import legvander
from geometry import I,grad,divergence,geometry,slice_values,cutoff,sup,Grid
from modal import Panels,Selected

def linear_scalar(g,h):
    v=divergence(g,h)-grad(g,np.trace(h,axis1=-2,axis2=-1))
    return np.trace(grad(g,v),axis1=-2,axis2=-1)

def outer(a,b):return a[...,None]*b[...,None,:]

def base(g,amp):
    h,K,_,_=slice_values(g.x[...,2],amp)
    cut=cutoff(g.r)[:,None,None,None]
    h0=I+cut*(h-I)
    k0=cut*(K-(np.trace(K,axis1=-2,axis2=-1)/np.trace(h,axis1=-2,axis2=-1))[...,None,None]*h)
    return h0,k0,h,K

def metric(op,h0,phi,beta):
    n=op.g.n;P=op.P
    ang=P@h0@P
    edge=np.einsum('...ij,...j->...i',P@h0,n)
    lift=np.linalg.solve(ang+op.nn,edge[...,None])[...,0]
    old=np.sum(lift*edge,axis=-1)
    edge2=edge+np.einsum('...ij,...j->...i',ang,beta)
    new=np.sum((lift+beta)*edge2,axis=-1)
    factor=np.exp(2*phi)
    delta=(factor-1)[...,None,None]*ang
    de=factor[...,None]*edge2-edge
    return h0+delta+outer(de,n)+outer(n,de)+(factor*new-old)[...,None,None]*op.nn

def flat_h(op,phi,beta):
    return 2*phi[...,None,None]*op.P+outer(op.g.n,beta)+outer(beta,op.g.n)

def probe(level):
    deg=[5,7,9,10][level]
    g=Panels([10 if level==3 else 5,deg,8,8],12*(level+1));op=Selected(g,3)
    n=g.n;r=g.r[:,None]
    k=3**2+2;Y=g.Y[:,k];D=op.grad[k]/np.sqrt(12)
    bump=np.where((r>2)&(r<3),np.exp(4-1/np.maximum((r-2)*(3-r),1e-12)),0)
    H=bump*Y;M=bump[...,None]*(Y[:,None]*n+.7*D-.3*np.cross(n,D))
    p,b,v=op.solve(H,M);h=flat_h(op,p,b);R=linear_scalar(g,h)
    V=divergence(g,v)
    nnh=np.einsum('...ij,...i,...j->...',h,n,n)
    # The old radial metric response for the same scalar source has a nonzero rr-component.
    from geometry import Protected
    ho,vo=Protected(g).solve(H,M)
    out=g.r>1.5
    scalar=sup(R-H);momentum=sup(V-M)
    return dict(degree=deg,collar=12*(level+1),scalar_error=scalar,momentum_error=momentum,
        relative_scalar=scalar/sup(H),relative_momentum=momentum/sup(M),
        wrong_scalar_sign=sup(R+H),wrong_momentum_sign=sup(V+M),
        selected_rr=sup(nnh),old_rr=sup(np.einsum('...ij,...i,...j->...',ho,n,n)),
        shift_outside_collar=sup(b[out]),inner=sup(h[g.r<1])+sup(v[g.r<1]))

def save_fields(g,h,K,spec):
    target=Grid(spec,{'radius':1})
    assert np.array_equal(target.n,g.n)
    # Target data are evaluated from the internal polynomial, panel by panel.
    for i,r in enumerate(target.r):
        p=np.searchsorted(g.edges,r)-1
        sl=slice(p*len(g.v),(p+1)*len(g.v))
        row=legvander([(r-g.mid[p])/g.half[p]],len(g.v)-1)[0]@g.basis_inv
        hi=np.einsum('r,raij->aij',row,h[sl]-I);ki=np.einsum('r,raij->aij',row,K[sl])
        np.savez_compressed(f'protected-{i:03}.npz',h=g.modes(hi.reshape(1,len(g.n),9))[0],
                            K=g.modes(ki.reshape(1,len(g.n),9))[0])
        if r<1:np.savez_compressed(f'inner-{i:03}.npz',h=hi+I,K=ki)

def extension(level,amp,save=False,native=False):
    spec=[[4,5,7,8],[6,7,9,8],[8,8,10,8],[8,8,10,16]][level]
    L=[3,5,7,7][level]
    g=Panels(spec,[12,22,32,32][level]);op=Selected(g,L)
    h0,k0,orig,origK=base(g,amp)
    h=h0.copy();v=np.zeros_like(k0);p=np.zeros(h.shape[:2]);b=np.zeros(h.shape[:3])
    history=[]
    for i in range(6):
        k=k0+v;inv=np.linalg.inv(h)
        K=k-(np.sum(inv*k,axis=(-2,-1))/3)[...,None,None]*h
        H,M,_,_=geometry(g,h,K)
        idx=g.r>1
        history.append([sup(H[idx]),sup(M[idx])])
        if i==5:break
        # Solve the coupled implicit equations on the product of the two fixed ranges.
        # Subtraction of the discrete flat part prevents repeated fitting of its truncation error.
        p,b,v=op.solve(linear_scalar(g,flat_h(op,p,b))-H,divergence(g,v)-M)
        h=metric(op,h0,p,b)
        if not np.all(np.isfinite(h)) or sup(h)>3:raise ArithmeticError(history)
    points=[]
    for R in [1.25,2,4,6,7.8,12,15]:
        if R>=spec[-1]:continue
        j=np.argmin(abs(g.r-R))
        points.append([float(g.r[j]),sup(h[j]-I),sup(K[j]),sup(H[j]),sup(M[j])])
    if save:save_fields(g,h,K,spec)
    if native:
        for j,r in enumerate(g.r):
            np.savez_compressed(f"native-{j:03}.npz",r=r,h=h[j],K=K[j])
    return dict(amp=amp,spec=spec,L=L,history=history,inner_h=sup((h-orig)[~idx]),
        inner_K=sup((K-origK)[~idx]),tail=points,min_eigenvalue=float(np.min(np.linalg.eigvalsh(h))),worst_H_radius=float(g.r[np.unravel_index(np.argmax(abs(H)*idx[:,None]),H.shape)[0]]),trace=sup(np.sum(inv*K,axis=(-2,-1)))),h,K

stage=sys.argv[1]
if stage.startswith('probe'):
    res=probe(int(stage[-1]))
elif stage=='polynomial':
    from polynomial import run
    res=run()
elif stage=='report':
    from report import run
    res=run(Path('/up'))
elif stage=='native3':
    res,_,_=extension(3,.012,True,True)
elif stage=='fieldcheck':
    from native import check
    res=check(Path('/up'))
elif stage in ('native32','native96'):
    from native import packet
    res=packet(Path('/up'),int(stage[6:]))
elif stage.startswith('filter'):
    from filters import run
    res=run(Path('/up'),int(stage[6:]))
elif stage=='wide':
    from pp import inverse
    res={'96':inverse(Path('/up'),[8,8,10,16],96,'band96',True)}
elif stage=='moments':
    from moments import run
    res=run()
elif stage.startswith('path'):
    from current import run
    res=run(Path('/up'),int(stage[-1]))
elif stage in ('smooth','smooth32'):
    from smooth_packet import run
    res=run(Path('/up'),32 if stage=='smooth32' else None)
elif stage in ('inverse','pp'):
    from pp import run
    res=run(stage)
else:
    level=int(stage[-1]);res,h,K=extension(level,.012,level>=2)
    if level==2:
        varied,h1,K1=extension(level,.01212)
        res.update(variation=varied,dh_per_amp=sup(h-h1)/.00012,dK_per_amp=sup(K-K1)/.00012)
Path('results.json').write_text(json.dumps(res,indent=2,allow_nan=False))
