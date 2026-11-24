"""Input-only Parseval calculation with radial quadrature; no solver fields or seed routine."""
import numpy as np
import sympy as sy
from scipy.special import roots_legendre

def integrate(cfg):
    r=sy.symbols('r', positive=True)
    radius=sy.Rational(str(cfg['radius']))
    profile=sy.exp(cfg['bump_shape']-cfg['bump_shape']/(1-(r/radius)**2))
    def lap(x):
        return sy.diff(r*r*sy.diff(x,r),r)/(r*r)
    radial=sy.lambdify(r,lap(lap(profile)),modules='numpy',cse=True)
    plus=np.diag([1.,-1.,0.])
    cross=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]])
    norm=np.sum((plus+cross)**2)
    estimates=[]
    for n in (32,64,128,256):
        x,w=roots_legendre(n)
        nodes=(x+1)*float(radius)/2
        integral=np.dot(w,nodes**2*radial(nodes)**2)*float(radius)/2
        estimates.append(float(.5*cfg['gain']**2*(2/5)*norm*integral))
    return dict(nodes=[32,64,128,256],C2=estimates,tensor_norm_squared=norm)
