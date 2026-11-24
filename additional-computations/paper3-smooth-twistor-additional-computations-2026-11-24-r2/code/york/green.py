"""Spherical Fourier quadrature of an SO(3)-equivariant word, versus spatial Green convolution."""
import json
from pathlib import Path
import numpy as np
import platform,scipy
print(platform.python_version(),np.__version__,scipy.__version__)
from scipy.special import spherical_jn
from scipy.integrate import simpson
from chain import symbols
cfg=json.loads(Path('/src/input.json').read_text())['green']
def radial(r):
    mask=r<1
    b=np.zeros_like(r);d=b.copy();dd=b.copy()
    x=r[mask];s=1-x*x;y=np.exp(3-3/s)
    b[mask]=y
    d[mask]=-6*y/s**2
    dd[mask]=y*(36*x*x/s**4-6/s**2-24*x*x/s**3)
    return b,d,dd
def probes(K,dk,h,amax=1):
    # u*=b(r)e1. Δ_L(b e1) is a radial scalar times e1 plus a radial dyadic n n1.
    r=np.linspace(0,amax,round(amax/h)+1);b,d,dd=radial(r)
    k=np.arange(round(K/dk)+1)*dk
    def ft(g,ell):
        return np.concatenate([4*np.pi*simpson(spherical_jn(ell,w[:,None]*r)*(g*r*r),x=r,axis=1) for w in np.array_split(k,1+len(k)//256)])
    iso=dd+2*d+d/3;an=(dd-d)/3
    mass=ft(b,0);A=ft(iso+an/3,0);B=ft(an,2)
    # For k along e1, the two chains on e1,e2 determine the covariant longitudinal/transverse eigenvalues.
    _,mat=symbols(np.stack((k[1:],np.zeros(len(k)-1),np.zeros(len(k)-1)),axis=-1))
    L=np.r_[-.75,mat[:,0,0]*k[1:]**2];T=np.r_[-1.,mat[:,1,1]*k[1:]**2]
    # k=0 radial endpoint is a continuous one-sided limit, not an FFT zero cell.
    def invert(flong,ftrans,R,mu):
        j0=spherical_jn(0,k*R);j2=spherical_jn(2,k*R)
        wt=(j0+j2)/3-mu**2*j2
        return simpson(T*ftrans*j0+(L*flong-T*ftrans)*wt,x=k)/(2*np.pi**2)
    rec=[abs(invert(A-2*B/3,A+B/3,R,mu)-radial(np.array([R]))[0][0]) for R in [.0,.41,.79,1.3] for mu in (0,1)]
    vals=[invert(mass,mass,R,mu) for R in [1.6,4.,8.,16.] for mu in (0,1)]
    return max(rec),np.array(vals),mass[0]
def direct(radii=(1.6,4.,8.,16.)):
    # Real-space volume integral of G_i1; axial symmetry about observation direction permits exact azimuth averaging.
    u,w=np.polynomial.legendre.leggauss(64);r=np.linspace(0,1.,513)
    b=radial(r)[0];q=r[:,None];ans=[]
    for R in radii:
        dist=np.sqrt(R*R+q*q-2*R*q*u)
        for mu in (0,1):
            x1sq=(R-q*u)**2 if mu else .5*q*q*(1-u*u)
            val=-((7/dist+x1sq/dist**3)@w)/16
            ans.append(simpson(r*r*b*val,x=r))
    return np.array(ans)
def periodic(n,L):
    x=(np.arange(n)-n//2)*(L/n);xyz=np.meshgrid(x,x,x,indexing='ij',sparse=True)
    f=radial(np.sqrt(sum(z*z for z in xyz)))[0];freq=np.fft.fftfreq(n,d=L/n)*2*np.pi
    k=np.meshgrid(freq,freq,freq,indexing='ij',sparse=True);s=sum(z*z for z in k);s[0,0,0]=1
    m=-(1-.25*k[0]**2/s)/s;m[0,0,0]=0
    v=np.fft.ifftn(np.fft.fftn(f)*m).real
    return dict(n=n,L=L,h=L/n,removed=f.mean(),parallel=v[n//2+round(1.5*n/L),n//2,n//2],transverse=v[n//2,n//2+round(1.5*n/L),n//2])
pilot=probes(12,.5,1/12)
assert np.isfinite(pilot[0]) and pilot[2]>0
oracle=direct();rows=[]
for K,dk,h in cfg['levels']:
    rec,vals,mass=probes(K,dk,h)
    rows.append(dict(K=K,dk=dk,h=h,recovery=rec,free_error=float(max(abs(vals-oracle))),values=vals.tolist(),far_scaled=(16*vals[-2:]).tolist()))
rec,vals,mass=probes(*cfg['levels'][-1]);asym=-mass*np.array([7.,8.])/(32*np.pi)
farerr=[float(max(abs(R*vals[2*i:2*i+2]-asym))) for i,R in enumerate([1.6,4.,8.,16.])]
sep={key:[probes(K,dk,h,a)[0] for K,dk,h,a in cases] for key,cases in dict(grid=[(96,.04,h,1) for h in [1/16,1/32,1/64]],frequency_cutoff=[(K,.04,1/128,1) for K in [24,48,96]],spatial_domain=[(96,.04,1/64,a) for a in [.75,1.,1.5]]).items()}
period=[periodic(round(L/h),L) for L,h in [(6,.125),(9,.125),(12,.125),(6,.25),(6,.0625)]]
result=dict(pilot_recovery=pilot[0],rows=rows,independent_spatial=oracle.tolist(),mass=mass,far_coeff=asym.tolist(),far_errors=farerr,separation=sep,periodic_control=period,free_at_periodic_probe=direct([1.5]).tolist())
result['passed']=bool(rec<cfg['rec_tol'] and rows[-1]['free_error']<cfg['free_tol'] and farerr[-1]<.02*abs(asym[0]) and farerr[-1]<farerr[1]/8 and max(v['removed'] for v in period)>0 and rows[-1]['recovery']<rows[0]['recovery'])
Path('results.json').write_text(json.dumps(result,indent=2));print(result['passed'],rows,farerr)
assert result['passed']
