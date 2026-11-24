"""Finite inverse/forward on the protected plane-wave slice."""
import json
from pathlib import Path
import numpy as np
from scipy.special import spherical_jn,lpmv,gammaln
from scipy.interpolate import BarycentricInterpolator,CubicSpline
from scipy.integrate import cumulative_trapezoid,solve_ivp
from geometry import Grid,I,grad,tf,divergence,geometry,slice_values,profile,sup
from read_cr import decode

def york(g,F):
    packed=np.concatenate([F,np.sum(g.x*F,axis=-1,keepdims=True)],axis=-1)
    m,rm=g.potential(g.modes(packed))
    p=g.sample(m)[...,:3];d=g.gradient(m,rm)
    return -p+(p+np.einsum('rai,raij->raj',g.x,d[...,:3,:])-d[...,3,:])/8
def LW(g,V):
    d=grad(g,V);return tf(d+d.swapaxes(-1,-2))
def lie(g,F,X,dX,identity=False):
    derivative=grad(g,F-I if identity else F)
    product=F@dX
    return np.einsum('...ijk,...k->...ij',derivative,X)+product+product.swapaxes(-1,-2)
def pull(g,h,K):
    displacement=g.x*0
    hist=[]
    for _ in range(4):
        f=divergence(g,tf(h-I));hist.append(sup(f[g.r<1]))
        X=-york(g,f)
        # Based normalization by the zero-frequency spherical average at the origin.
        deg=len(g.v)
        center=BarycentricInterpolator(g.r[:deg],np.einsum('rac,a->rc',X[:deg],g.dw/(4*np.pi)),rng=3)(0)
        X-=center
        dx=grad(g,X)
        increments=[]
        for T,one in [(h,True),(K,False)]:
            first=lie(g,T,X,dx,one)
            increments.append(first+.5*lie(g,first,X,dx))
        movement=X+np.einsum('...ij,...j->...i',grad(g,displacement),X)
        displacement+=movement+.5*np.einsum('...ij,...j->...i',grad(g,movement),X)
        h+=increments[0];K+=increments[1]
    hist.append(sup(divergence(g,tf(h-I))[g.r<1]))
    return h,K,displacement,hist
def hann(r,L):
    s=np.clip((r-.6*L)/(.4*L),0,1)
    return 1-s**3*(10+s*(-15+6*s))
def frame(n):
    i=np.eye(3)[np.argmin(abs(n),axis=-1)]
    e=np.cross(i,n);e/=np.linalg.norm(e,axis=-1)[:,None];return e+1j*np.cross(n,e)
def roots(g,q,p,cap,tag,split=False):
    # Continuous Fourier–Bessel quadrature on the tapered extension; no periodic identification.
    width=g.outer
    samples=int(np.ceil(cap*width*2/np.pi))+1
    k=np.linspace(0,cap,samples+1)[1:];dk=k[0]
    wk=np.full_like(k,dk);wk[-1]*=.5
    taper=hann(g.r,width)[:,None,None];z=frame(g.n)
    shape=q.shape
    pair=np.concatenate([q.reshape(*shape[:2],9),p.reshape(*shape[:2],9)],axis=-1)
    coeff=g.modes(pair*taper)
    F=np.zeros((len(k),(g.ell+1)**2,18),complex)
    nodes=g.r;weights=g.dr
    if split:
        # Preserve the original field interpolant. Resolve the Bessel kernel
        # on outer panels by actual subpanel quadrature at fixed spectral input.
        from numpy.polynomial.legendre import legvander
        nodes=[];weights=[];finer=[]
        deg=len(g.v)
        for j,half in enumerate(g.half):
            count=max(1,int(np.ceil(cap*half/2)))
            for left in range(count):
                v=(g.v+1+2*left)/count-1
                nodes.extend(g.mid[j]+half*v);weights.extend(half/count*g.w)
                interp=legvander(v,deg-1)@g.basis_inv
                segment=coeff[j*deg:(j+1)*deg]
                finer.append((interp@segment.reshape(deg,-1)).reshape(segment.shape))
        nodes=np.array(nodes);weights=np.array(weights);coeff=np.concatenate(finer)
    def kernel(l):return spherical_jn(l,k[:,None]*nodes)
    for l in range(g.ell+1):
        sl=slice(l*l,(l+1)**2)
        m=coeff[:,sl,:].reshape(len(nodes),-1)
        F[:,sl,:]=(4*np.pi*(-1j)**l*((kernel(l)*(weights*nodes**2))@m)).reshape(len(k),2*l+1,18)
    vals=g.sample(F).reshape(len(k),len(z),2,3,3)
    def pack(values,Zbase):
        rec={}
        for sign in [1,-1]:
            Z=Zbase if sign==1 else Zbase.conj()
            A=values[:,:,0]/2+(1j*sign/k[:,None,None,None])*values[:,:,1]
            rec[sign]=(np.einsum('kaij,ai,aj->ka',A,Z,Z)/(4j*sign*k[:,None]),
                np.einsum('kaij,ai,aj->ka',A,Z.conj(),Z.conj())*(k[:,None]**3/(1j*sign)))
        return rec
    if tag=="band32":
        from smooth_packet import write_roots
        write_roots(F,k)
    records=pack(vals,z)
    if tag=='paired':
        ell=np.repeat(np.arange(g.ell+1),2*np.arange(g.ell+1)+1)
        antipode_values=g.sample(F*((-1)**ell)[None,:,None]).reshape(vals.shape)
        opp=pack(antipode_values,frame(-g.n))
        for i,r in enumerate(k):
            np.savez_compressed(f'opp-{i:03}.npz',positive=np.stack([opp[1][0][i],opp[-1][0][i]],axis=-1),
                               horizontal=np.stack([opp[1][1][i],opp[-1][1][i]],axis=-1))
    # Per radial shell packed into four arrays; binary output cap applies to each record.
    for i,r in enumerate(k):
        np.savez_compressed(f'{tag}-{i:03}.npz',k=r,
              positive=np.stack([records[1][0][i],records[-1][0][i]],axis=-1),
              horizontal=np.stack([records[1][1][i],records[-1][1][i]],axis=-1))
    return k,wk

def invert(g,path,k,wk,tag):
    z=frame(g.n);out=[]
    for i,r in enumerate(k):
        class Frame:
            def frames(self):return z
        f=Frame();f.r=np.full(len(z),r)
        with np.load(path/f'{tag}-{i:03}.npz') as d:
            assert d['k']==r
            a,b=decode(f,d)
        out.append(np.concatenate([a.reshape(-1,9),b.reshape(-1,9)],axis=1))
    coeff=g.modes(np.array(out))
    ans=np.empty((len(g.r),coeff.shape[1],18),complex)
    for l in range(g.ell+1):
        sl=slice(l*l,(l+1)**2)
        J=spherical_jn(l,g.r[:,None]*k)
        ans[:,sl,:]=((1j**l/(2*np.pi**2))*((J*(wk*k*k))@coeff[:,sl].reshape(len(k),-1))).reshape(len(g.r),2*l+1,18)
    real=g.sample(ans).real
    return [v.reshape(*real.shape[:2],3,3) for v in np.split(real,2,axis=-1)]

def extension(g,base):
    fs=sorted(base.glob('protected-*.npz'));assert len(fs)==len(g.r)
    fields=[[],[]]
    for f in fs:
        with np.load(f) as d:
            for seq,key in zip(fields,['h','K']):seq.append(d[key])
    h,k=[g.sample(np.stack(s)).reshape(*g.x.shape[:2],3,3) for s in fields]
    h+=I
    for f in sorted(base.glob('inner-*.npz')):
        j=int(f.stem.split('-')[-1])
        with np.load(f) as d:h[j]=d['h'];k[j]=d['K']
    return h,k
def inverse(base,spec,cap,prefix,split=False):
    g=Grid(spec,dict(radius=1))
    h,K=extension(g,base)
    h,K,dis,hist=pull(g,h,K)
    q=tf(h-I);kap=K-(np.trace(K,axis1=-2,axis2=-1)/np.trace(h,axis1=-2,axis2=-1))[...,None,None]*h
    W=york(g,divergence(g,kap));p=kap-LW(g,W)
    k,wk=roots(g,q,p,cap,prefix,split)
    # Coordinate identification is the saved marking; no old h,K is passed to forward solve.
    dm=g.modes(dis)
    for i in range(len(g.r)):np.savez_compressed(f'{prefix}-mark-{i:03}.npz',X=dm[i])
    return dict(k=k.tolist(),wk=wk.tolist(),gauge=hist,norm_q=sup(q[g.r<1]),norm_p=sup(p[g.r<1]),spec=spec,cap=cap)
def forward(g,q,p):
    # Fixed flat inverse of Appendix A.6 applied to the nonlinear remainder.
    # Projected smooth TT inputs have quadrature errors; keep their actual residual.
    u=np.zeros(q.shape[:2]);v=0*p;hist=[]
    for _ in range(5):
        h=I+q+u[...,None,None]*I
        inv=np.linalg.inv(h);kap=p+v
        K=kap-(np.sum(inv*kap,axis=(-2,-1))/3)[...,None,None]*h
        H,M,ric,electric=geometry(g,h,K)
        hist.append([sup(H[g.r<1]),sup(M[g.r<1])])
        if _==4:break
        lap=np.trace(grad(g,grad(g,u)),axis1=-2,axis2=-1)
        rhs=H+2*lap
        m,_=g.potential(g.modes(rhs[...,None]))
        u=-.5*g.sample(m)[...,0]
        v=LW(g,york(g,divergence(g,v)-M))
    return h,K,hist
def eval_points(g,m,points):
    r=np.linalg.norm(points,axis=1);directions=points/np.maximum(r,1e-30)[:,None]
    ph=np.arctan2(directions[:,1],directions[:,0]);cz=directions[:,2]
    basis=[]
    for l in range(g.ell+1):
        for j in range(-l,l+1):
            k=abs(j);scale=np.sqrt((2*l+1)/(4*np.pi)*np.exp(gammaln(l-k+1)-gammaln(l+k+1)))
            basis.append(scale*lpmv(k,l,np.clip(cz,-1,1))*(1 if j==0 else np.sqrt(2)*(np.sin(k*ph) if j<0 else np.cos(k*ph))))
    Y=np.array(basis).T
    result=np.zeros((len(r),m.shape[-1]))
    for j in np.unique(np.searchsorted(g.edges,r,side='right')-1):
        idx=(r>=g.edges[j])&(r<g.edges[j+1]);local=slice(j*len(g.v),(j+1)*len(g.v))
        sample=BarycentricInterpolator(g.r[local],m[local],rng=5)(r[idx])
        result[idx]=np.einsum('ah,ahc->ac',Y[idx],sample)
    return result
def wave_read(g,h,K,dis):
    # Invert the saved based spatial map on the marked Rosen centreline.
    z=np.linspace(-.86,.86,129)
    points=np.column_stack([0*z,0*z,z]);points[:,0]=.02
    dm=g.modes(dis);y=points.copy()
    for j in range(5):y=points-eval_points(g,dm,y)
    J=I+eval_points(g,g.modes(grad(g,dis).reshape(*dis.shape[:2],9)),y).reshape(-1,3,3)
    A=np.linalg.inv(J)
    def at(F):return A.swapaxes(1,2)@eval_points(g,g.modes(F.reshape(*dis.shape[:2],9)),y).reshape(-1,3,3)@A
    hs=at(h);ks=at(K);old,kold,uold,vold=slice_values(z,.012)
    a=np.sqrt(hs[:,0,0]);b=np.sqrt(hs[:,1,1]);du=np.sqrt(hs[:,2,2])/(a*b)
    u=cumulative_trapezoid(du,z,initial=0);u-=u[len(z)//2]
    au=ks[:,0,0]*b;bu=ks[:,1,1]*a
    Afn=CubicSpline(u,a);Bfn=CubicSpline(u,b)
    tidal=CubicSpline(u,au)(u,1)/a
    expected=profile(uold,.012)
    mask=(z>.35)&(z<.78);flat=(z<-.35)&(z>-.78)
    pi=np.array([.24,-.31]);u0=-.7;end=.73
    def geodesic(s,y):
        ar=Afn(s);br=Bfn(s)
        return [pi[0]/ar**2,pi[1]/br**2,.5*sum(pi**2/np.array([ar,br])**2)]
    sol=solve_ivp(geodesic,(u0,end),[.12,-.15,0],rtol=1e-11,atol=1e-12)
    # Compare in Brinkmann x=aX,y=bY,V=v+(a' a X²+b' b Y²)/2.
    X,Y,vr=sol.y[:,-1]
    physical=[Afn(end)*X,Bfn(end)*Y,vr+.5*(CubicSpline(u,au)(end)*Afn(end)*X**2+CubicSpline(u,bu)(end)*Bfn(end)*Y**2)]
    def brink(s,w):
        x,y,dx,dy,v=w;H=profile(s,.012)
        return [dx,dy,H*x,-H*y,.5*(dx*dx+dy*dy+H*(x*x-y*y))]
    check=solve_ivp(brink,(u0,end),[.12,-.15,*pi,0],rtol=1e-12,atol=1e-13)
    oracle=check.y[[0,1,4],-1]
    # Restore physical scale in the data, and change the measured affine
    # and transverse lengths by the same factor. Fit curvature anew in U.
    scale=2.
    hphys=scale**2*hs;kphys=scale*ks
    ap=np.sqrt(hphys[:,0,0])/scale;bp=np.sqrt(hphys[:,1,1])/scale
    U=cumulative_trapezoid(np.sqrt(hphys[:,2,2])/(ap*bp),z,initial=0)
    U-=U[len(z)//2]
    physical_dA=kphys[:,0,0]*bp/scale**2
    physical_tide=CubicSpline(U,physical_dA)(U,1)/ap
    target_tide=profile(U/scale,.012)/scale**2
    bad=CubicSpline(U,ks[:,0,0]*bp/scale**2)(U,1)/ap
    scale_info=dict(scale=scale,physical_curvature_error=sup(physical_tide[mask]-target_tide[mask]),
        missed_K_scale_error=sup(bad[mask]-target_tide[mask]),proper_length_error=sup(U-scale*uold),
        pullback_min_det=float(np.min(np.linalg.det(J))))
    # Build an actual vacuum pp metric from recovered data in the polarized
    # sector. Only the supplied marking identifies the two Killing directions.
    # The other polarization equation and all comparison data are withheld.
    Hfit=CubicSpline(u,au).derivative()
    inferred=lambda U:float(Hfit(U)/Afn(U))
    Btest=CubicSpline(u,bu).derivative()(u)/b
    ustart=-.72;ustop=.76
    def rosen_rhs(t,y):
        a1,b1,v1,v2=y;value=inferred(t)
        return [v1,v2,value*a1,-value*b1]
    anchors=[Afn(ustart),Bfn(ustart),CubicSpline(u,au)(ustart),CubicSpline(u,bu)(ustart)]
    dev=solve_ivp(rosen_rhs,(ustart,ustop),anchors,rtol=1e-11,atol=1e-13,
                  dense_output=True,max_step=.015)
    keeps=(u>ustart)&(u<ustop)
    probe=np.array([.61,-.32,0.,0.,0.])
    def memory(t,y,H):
        x0,y0,dx,dy,v0=y;A=H(t)
        return [dx,dy,A*x0,-A*y0,.5*(dx*dx+dy*dy+A*(x0*x0-y0*y0))]
    def ray(H):
        return solve_ivp(lambda t,y:memory(t,y,H),(ustart,ustop),probe,
                         rtol=1e-11,atol=1e-13,max_step=.015).y[:,-1]
    measured=ray(inferred);target=ray(lambda U:profile(U,.012))
    vacuum=dict(rosen_field_error=sup(dev.sol(u[keeps])[:2]-np.array([a[keeps],b[keeps]])),
          withheld_polarization_error=sup((tidal+Btest)[mask]),geodesic_error=sup(measured-target),
          geodesic_signal=sup(target-probe),measured=measured.tolist(),target=target.tolist(),
          measured_curvature=sup(tidal[mask]),flat_leakage=sup(tidal[flat]),profile_u=u[::3].tolist(),
          profile_fit=tidal[::3].tolist(),profile_truth=expected[::3].tolist())
    return dict(metric=sup(hs-old),K=sup(ks-kold),chart=sup(eval_points(g,dm,y)+y-points),
        positive_tidal=sup(tidal[mask]-expected[mask]),signal=sup(expected[mask]),
        flat_tidal=sup(tidal[flat]),tidal_sign_control=sup(tidal[mask]+expected[mask]),
        geodesic_difference=sup(np.array(physical)-oracle),endpoint=physical,oracle=oracle.tolist(),
        lapse_coordinate=sup(u-uold),scale_test=scale_info,vacuum_development=vacuum)
def run(stage):
    base=Path('/up')
    if stage in ('local','inverse'):
        spec=[8,8,10,8] if len(list(base.glob('protected-*.npz')))<240 else [8,8,10,16]
        return {str(cap):inverse(base,spec,cap,'band'+str(cap),stage=='inverse') for cap in (8,16,32)}
    meta=json.loads((base/'results.json').read_text())
    rows=[]
    for band,rec in meta.items():
        g=Grid(rec['spec'],dict(radius=1));tag='band'+band
        q,p=invert(g,base,np.array(rec['k']),np.array(rec['wk']),tag)
        d=np.stack([np.load(f)['X'] for f in sorted(base.glob(tag+'-mark-*.npz'))])
        dis=g.sample(d)
        h,K,hist=forward(g,q,p)
        rows.append(dict(band=band,defects=hist,readout=wave_read(g,h,K,dis)))
    return rows
