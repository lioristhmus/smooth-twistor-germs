import numpy as np
from geometry import grad,tf
def york(g,F):
    packed=np.concatenate([F,np.sum(g.x*F,axis=-1,keepdims=True)],axis=-1)
    m,rm=g.potential(g.modes(packed))
    p=g.sample(m)[...,:3];d=g.gradient(m,rm)
    return -p+(p+np.einsum('rai,raij->raj',g.x,d[...,:3,:])-d[...,3,:])/8
def LW(g,V):
    d=grad(g,V);return tf(d+d.swapaxes(-1,-2))
