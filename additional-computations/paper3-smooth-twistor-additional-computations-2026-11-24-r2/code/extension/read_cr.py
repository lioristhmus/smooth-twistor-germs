"""Decode sampled section values at the north and south roots of each stored rotated frame."""
import numpy as np
def decode(g,rec):
    vals=rec['positive']; neg=rec['horizontal'];z=g.frames()
    conj=z.conj();zz=np.einsum('...i,...j->...ij',z,z);cc=zz.conj()
    r=g.r[...,None,None]
    plus=4j*r*(vals[...,0,None,None]*cc/4)+1j/(4*r**3)*(neg[...,0,None,None]*zz)
    minus=-4j*r*(vals[...,1,None,None]*zz/4)-1j/(4*r**3)*(neg[...,1,None,None]*cc)
    return plus+minus,(-.5j*r)*(plus-minus)
