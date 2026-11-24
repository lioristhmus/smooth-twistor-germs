"""A flat spectral taper for a supported high-frequency collar; hold CR readout and chart fixed."""
from pathlib import Path
import json
import numpy as np
from pp import Grid,invert,forward,wave_read
from modal import transition
def run(base,cap):
    tag='band'+str(cap)
    rec=json.loads((base/'results.json').read_text())[str(cap)]
    g=Grid(rec['spec'],dict(radius=1));k=np.array(rec['k']);wk=np.array(rec['wk'])
    d=g.sample(np.stack([np.load(f)['X'] for f in sorted(base.glob(tag+'-mark-*.npz'))]))
    rows=[]
    for upper in ([16,24,32] if cap==32 else [48,64,96]):
        # Heat regularization is also nearly zero before the sampled upper boundary.
        filt=np.exp(-(k/(upper*.43))**2)*(1-transition((k/upper-.7)/.3)[0])
        q,p=invert(g,base,k,wk*filt,tag)
        h,K,hist=forward(g,q,p)
        rows.append(dict(band=cap,upper=upper,readout=wave_read(g,h,K,d),defects=hist))
    return rows
