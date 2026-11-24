"""Independent reconstruction from saved gradient-frame coefficients only, with separate seed readout from CR files for comparison."""
from pathlib import Path
import json
import numpy as np
import records
from spectral import Grid,maxabs
from read_cr import decode
base=Path('/in/transport');cr=Path('/in/cochains')
cfg=json.loads(Path('/src/input.json').read_text())['transport']
mark=json.loads((base/'marking.json').read_text());I=np.eye(3)
def test(g,file,target):
    with records.load(file) as z:b=z['b'];K=z['boost']/b[...,None,None];w=z['rotation_vector']
    A=b**-2;u=A-1
    da=np.stack([g.grad(u,i) for i in range(3)],axis=-1)
    d2=np.stack([g.grad(da[...,i],i) for i in range(3)],axis=-1).sum(axis=-1)
    # h=exp(2φ)δ. Differentiate log A by the chain rule on sampled jets.
    phi_grad=da/(2*A[...,None]);norm2=(phi_grad**2).sum(axis=-1)
    lap_phi=d2/(2*A)-2*norm2
    R=-(4*lap_phi+2*norm2)/A
    H=R-(K**2).sum(axis=(-1,-2))/A**2
    mom=np.zeros_like(da)
    for j in range(3):
        for i in range(3):
            val=g.grad(K[...,i,j],i)
            for k in range(3):
                Gkii=2*I[k,i]*phi_grad[...,i]-phi_grad[...,k]
                Gkij=I[k,i]*phi_grad[...,j]+I[k,j]*phi_grad[...,i]-I[i,j]*phi_grad[...,k]
                val-=Gkii*K[...,k,j]+Gkij*K[...,i,k]
            mom[...,j]+=val/A
    trace=np.trace(K,axis1=-2,axis2=-1)
    kap=K-(trace/3)[...,None,None]*I
    P=I-g.nvec[...,None]*g.nvec[...,None,:];v=P@g.fft(kap)@P
    p=g.inv(v-.5*P*np.trace(v,axis1=-2,axis2=-1)[...,None,None]).real
    h=A[...,None,None]*I
    q=h-np.trace(h,axis1=-2,axis2=-1)[...,None,None]*I/3
    gauge=sum(g.grad(q[...,i,:],i) for i in range(3))
    return dict(hamilton=maxabs(H),momentum=maxabs(mom),maximal=maxabs(trace/A),gauge=maxabs(gauge),q=maxabs(q),decoded=maxabs(p-target),spin_frame_connection=maxabs(w-phi_grad),flat_control=maxabs((K*K).sum(axis=(-2,-1))))
rows=json.loads((base/'results.json').read_text())
for n in mark['sizes']:
    g=Grid(n,mark['L']);p=np.zeros(g.r.shape+(3,3))
    for j in [0,1]:
        with records.load(cr/f'cr{n}_{j}.npz') as rec:p+=g.inv(decode(g,rec)[1]).real
    for row in [v for v in rows if v['n']==n]:
        row.update(test(g,base/row['file'],p*row['eps']/mark['eps']))
        if row['steps']==max(cfg['steps']):
            row['independent_reference']=test(g,base/f"n{n}_e{row['eps']}_reference.npz",p*row['eps']/mark['eps'])
for r in rows:
    def fields(fn):
        with records.load(base/fn) as z:return z['b']**-2,z['boost']/z['b'][...,None,None]
    A,K=fields(r['file'])
    ref,kref=fields(f"n{r['n']}_e{r['eps']}_reference.npz")
    r['metric_K_reference_error']=max(maxabs(ref-A),maxabs(kref-K))
    if r['path']=='broken':
        A0,K0=fields(r['file'].replace('broken','straight'))
        r['metric_K_path_difference']=max(maxabs(A0-A),maxabs(K0-K))
seq=[r for r in rows if r['n']==cfg['size'] and r['path']=='broken' and r['eps']==max(q['eps'] for q in rows)]
passed=all(r['decoded']<cfg['decode_tol'] and max(r['gauge'],r['maximal'],r['spin_frame_connection'])<1e-9 for r in rows)
passed=passed and all(max(r['hamilton'],r['momentum'])<cfg['max_constraint'] for r in rows if r['steps']==max(cfg['steps']))
passed=passed and seq[-1]['path_difference']<seq[0]['path_difference']/5
Path('results.json').write_text(json.dumps(dict(rows=rows,passed=bool(passed)),indent=2))
print(passed,[(r['path_difference'],r['hamilton'],r['momentum']) for r in seq]);assert passed
