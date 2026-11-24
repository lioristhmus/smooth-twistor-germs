import json, sys, platform
from pathlib import Path
import numpy as np
import scipy
cfg=json.loads(Path('/src/plan.json').read_text())
stage=sys.argv[1]
if stage=='audit':
    from audit import assess
    result=assess(cfg)
elif stage=='reference':
    from reference import integrate
    result=integrate(cfg)
    assert abs(result['C2'][-2]/result['C2'][-1]-1)<cfg['criteria']['input_reference_last_relative']
else:
    from solver import Grid,solve
    mesh=cfg['profiles'][stage]
    g=Grid(mesh,cfg)
    common=cfg['default_tolerance']
    if stage=='pilot':
        cases=[(.008,common,1,True),(.008,common,1,False)]
    elif stage=='fine':
        cases=[(e,common,1,False) for e in cfg['epsilon']]
        cases.extend([(.008,common,.5,False),(.008,common,1,True),(0.,common,1,False)])
    elif stage=='tolerance':
        cases=[(.128,t,1,False) for t in (cfg['loose_tolerance'],cfg['tight_tolerance'])]
    else:
        cases=[(e,common,1,False) for e in (.128,.008,.002)]
    rows=[]
    for eps,tol,factor,zero in cases:
        save='selected' if stage=='fine' and eps==.008 and factor==1 and not zero else None
        row=solve(g,cfg,eps,tol,factor,zero,save)
        print(eps,factor,zero,row['coefficient'],row['extrapolated_ratio'],len(row['iterations']),flush=True)
        rows.append(row)
    result=dict(profile=mesh,radial_nodes=len(g.r),angular_nodes=len(g.n),rows=rows)
result.update(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
Path('results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if stage=='audit' and not result['passed']:
    raise SystemExit('A preregistered check failed; see results.json.')
