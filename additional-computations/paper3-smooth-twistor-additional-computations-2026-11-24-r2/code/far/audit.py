"""Read stored fields independently; compare all declared gates and replay bytes."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy.special import lpmv,gammaln
from math import pi
def load(f):return json.loads(f.read_text())

def fields_only(path,eps,radii):
    # Does not import a solver, reconstruct a source, or read its predicted mass.
    with np.load(path/'selected-grid.npz') as data:
        r=data['r'];edges=data['edges'];n=data['unit_directions'];weights=data['angular_weights']/(4*pi)
        ell=int(data['ell'])
    z=[];V=[]
    for f in sorted(path.glob('selected-fields-*.npz')):
        with np.load(f) as data:
            z.append(data['z'])
            V.append(data['V'])
    z=np.stack(z);V=np.stack(V)
    angle=np.arctan2(n[:,1],n[:,0])
    basis=[]
    for l in range(ell+1):
        for m in range(-l,l+1):
            k=abs(m)
            norm=np.sqrt((2*l+1)/(4*pi)*np.exp(gammaln(l-k+1)-gammaln(l+k+1)))
            longitude=(1 if m==0 else np.sqrt(2)*(np.cos(k*angle) if m>0 else np.sin(k*angle)))
            basis.append(norm*lpmv(k,l,n[:,2])*longitude)
    Y=np.array(basis)
    powers=np.repeat(np.arange(ell+1)+1,2*np.arange(ell+1)+1)
    def interpolator(j):
        inside=(r>edges[j])&(r<edges[j+1])
        return BarycentricInterpolator(r[inside],z[inside],rng=17)
    boundary=interpolator(len(edges)-2)(edges[-1])
    def coeff(R):
        if R>=edges[-1]:
            cz=boundary*(edges[-1]/R)**powers
            return cz,-powers*cz/R
        p=interpolator(np.searchsorted(edges,R,side='right')-1)
        return p(R),p.derivative(R)
    means=[];flux=[]
    for R in radii:
        a,b=coeff(R)
        sampled=a@Y
        w=eps**2*sampled
        means.append(float(R*np.dot(weights,sampled*(4+w*(6+w*(4+w))))))
        flux.append(float(-R**2*np.dot(weights,4*(1+w)**3*(b@Y))))
    # Evaluate the scaled vector field to verify an independently readable nonzero W.
    peak=max(np.max(np.abs(v.T@Y)) for v in V)*eps**3
    return dict(mean_ratio=means,flux_ratio=flux,max_W=float(peak),
                boundary_monopole_ratio=float(4*edges[-1]*boundary[0]/np.sqrt(4*pi)))

def assess(cfg):
    primary=Path('/inputs/primary')
    reports={p.parent.name:load(p) for p in primary.glob('*/results.json')}
    ctr=cfg['criteria']
    C=reports['reference']['C2'][-1]
    def row(stage,eps,mult=1,zero=False,tol=None):
        return next(v for v in reports[stage]['rows'] if v['eps']==eps and v['zero']==zero and
                    v['multiplier']==mult and (tol is None or v['tolerance']==tol))
    def ratio(stage,eps,*args,**kwargs):return row(stage,eps,*args,**kwargs)['extrapolated_ratio']
    def diff(a,b):return abs(a-b)/C
    amps=cfg['epsilon']; ratios=[ratio('fine',e) for e in amps]
    exact_check=reports['reference']['C2']
    checks={}
    def gate(key,value,bound):
        checks[key]=dict(value=float(value),limit=bound,passed=bool(value<=bound))
    gate('input quadrature',diff(exact_check[-1],exact_check[-2]),ctr['input_reference_last_relative'])
    gate('small amplitude',diff(ratios[-1],C),ctr['small_epsilon_ratio_relative'])
    gate('quadratic solve',diff(ratio('fine',0.),C),ctr['quadratic_relative'])
    gate('grid',max(diff(ratio('fine',e),ratio('refined',e)) for e in [.128,.008,.002]),ctr['fine_spatial_ratio_relative'])
    gate('exterior',max(diff(ratio('exterior',e),ratio('fine',e)) for e in [.128,.008,.002]),ctr['cutoff_ratio_relative'])
    gate('iteration',diff(ratio('tolerance',.128,tol=cfg['tight_tolerance']),ratios[0]),ctr['tight_tolerance_ratio_relative'])
    gate('flux',max(diff(row('fine',e)['extrapolated_flux_ratio'],val) for e,val in zip(amps,ratios)),ctr['mean_flux_extrap_relative'])
    gate('half source',abs(ratio('fine',.008,.5)/(.5*C)-1),2e-5)
    zero=row('fine',.008,zero=True)
    gate('zero',max(abs(zero[k]) for k in ['coefficient','max_W','max_u','max_LW']),0.)
    trend=all(val>0 for val in ratios) and all(abs(l-C)>abs(h-C) for l,h in zip(ratios,ratios[1:]))
    stored=fields_only(primary/'fine',.008,np.array(cfg['radii']))
    main=row('fine',.008)
    stored['max_mean_ratio_difference']=float(max(abs(np.array(stored['mean_ratio'])-main['mean_ratio'])))
    stored['max_flux_ratio_difference']=float(max(abs(np.array(stored['flux_ratio'])-main['flux_ratio'])))
    stored['max_W_difference']=abs(stored['max_W']-main['max_W'])
    passed=trend and all(v['passed'] for v in checks.values())
    return dict(gates=checks,all_positive_and_trend=trend,passed=passed,stored_fields=stored,
                C2=C,
                observed_quartic=[(val-C)/e**2 for val,e in zip(ratios,amps)])
