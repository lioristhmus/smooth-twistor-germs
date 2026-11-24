"""Independent continuum collar coefficients, using adaptive quadrature and an adjoint IVP."""
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import expit
from modal import Panels,Selected
def quad1(f,a,b):return quad(f,a,b,epsabs=2e-12,epsrel=2e-10,limit=100)[0]
def bump(r):
    return np.exp(4-1/((r-2)*(3-r))) if 2<r<3 else 0.
def weight(r,eps):
    if r<=1.5:return 1.,0.
    if r>=2:return r**(-eps),-eps
    t=2*r-3
    v=expit(-1/t+1/(1-t));dv=v*(1-v)*(t**-2+(1-t)**-2)
    return np.exp(-eps*v*np.log(r)),-eps*(v+2*r*dv*np.log(r))
def oracle(l,eps):
    s=np.sqrt(l*(l+1));v=s/np.sqrt(2);nu=np.sqrt(s*s+4)
    lo=1+.1/l;hi=1+1/l
    def cp(r):
        t=(l*(r-1)-.1)/.9
        if not 0<t<1:return 0.
        v=expit(1/(1-t)-1/t)
        return v*(1-v)*(1/(1-t)**2+1/t**2)*l/.9
    # Integrate the scalar shape against the exterior adjoint by parts,
    # avoiding the second derivative of the compact cutoff.
    d=-quad1(lambda r:r**(v+1)*cp(r),lo,hi)/((v+1)*quad1(lambda r:r**v*cp(r),lo,hi))
    def G(r):
        return quad1(lambda x:(x**(v+1)+d*(v+1)*x**v)*cp(x),lo,r)-d*r**(v+1)*cp(r)
    def first(r):
        w,eta=weight(r,eps)
        return w*(1+d*eta/r)*cp(r)+r**(2-v)*w*eta*G(r)/r**4
    denom=quad1(first,lo,hi)
    scalar=.5*quad1(lambda r:r**(2-v)*bump(r),2,3)/denom
    if l==1:return [scalar]
    electric=-quad1(lambda r:r**(2-l)*(l/s+.7)*bump(r),2,3)/quad1(lambda r:weight(r,eps)[0]*cp(r),lo,hi)
    # Terminal covector for the positive exponent, scaled by its first coefficient.
    ev,vec=np.linalg.eig(np.array([[-2.,-s*s],[-1.,2.-eps]]));L=vec[:,np.argmax(ev)];L/=L[0]
    def adjoint(r,y):
        eta=weight(r,eps)[1]
        return np.array([2*y[0]+s*s*y[1],y[0]-(2+eta)*y[1]])/r
    adj=solve_ivp(adjoint,(8,1),L,rtol=3e-12,atol=1e-13,dense_output=True,max_step=.055).sol
    magnetic=.3*quad1(lambda r:adj(r)[0]*bump(r),2,3)/quad1(lambda r:adj(r)[0]*weight(r,eps)[0]*r**nu*cp(r),lo,hi)
    return [scalar,electric,magnetic]
def run():
    rows=[]
    for level,order in enumerate([6,8,10]):
        g=Panels([10,order,7,8],24+level*12);r=g.r[:,None]
        f=np.array([bump(x) for x in g.r])[:,None];zero=0*f
        for eps in (0.,.001):
            op=Selected(g,4,eps)
            for l in [1,2,3,4]:
                _,c=op.system['s',l].solve(np.stack([zero,-r*f/4],axis=-1))
                cs=[float(c[0])]
                if l>=2:
                    cs+= [float(op.system[key,l].solve(F)[1][0]) for key,F in [
                        ('e',np.stack([f,.7*f],axis=-1)),('h',np.stack([-.3*f,zero],axis=-1))]]
                exact=np.array(oracle(l,eps))
                rows.append(dict(degree=l,weight=eps,order=order,collocation=cs,reference=exact.tolist(),
                     relative=float(np.max(abs(cs-exact)/abs(exact)))))
    return rows
