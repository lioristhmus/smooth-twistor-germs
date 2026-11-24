"""Serialize a bounded NumPy record in 64 KiB parts under the per-file cap."""
from io import BytesIO
from pathlib import Path
import json
import numpy as np
def save(file,**fields):
    p=Path(file) if str(file).endswith('.npz') else Path(str(file)+'.npz')
    buf=BytesIO()
    np.savez_compressed(buf,**fields)
    data=buf.getbuffer();size=64*1024
    assert len(data)<12*1024**2
    parts=[]
    for i in range(0,len(data),size):
        f=p.with_name(p.name+f'.{i//size:03d}')
        f.write_bytes(data[i:i+size]);parts.append(f.name)
    p.write_text(json.dumps(parts))
def load(file):
    p=Path(file);parts=json.loads(p.read_text())
    assert len(parts)<200 and all(Path(n).name==n for n in parts)
    return np.load(BytesIO(b''.join((p.parent/n).read_bytes() for n in parts)),allow_pickle=False)
