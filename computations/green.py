"""Calibrated Fourier path; the independent oracle uses the York differential operator."""
from sympy import I, Matrix, Rational as Q, diag, eye, simplify, sqrt, zeros
from itertools import product

def run():
    k = Matrix([1, 2, 2])
    r = sqrt(k.dot(k))
    d = k*k.T-r**2*eye(3)
    cross = Matrix.hstack(*(k.cross(eye(3)[:,j]) for j in range(3)))
    def curl(b):
        return I*(cross*b+(cross*b).T)/2
    def encode(e):
        return d*e*d - d*(d*e).trace()/2
    def respond(b, sign=1):
        # Root multiplier 2 i C, dual factor k_minus/k_plus, conic 4 i C.
        dual = (I/(4*r**3))/(4*I*r)
        return 4*I*curl(sign*dual*2*I*curl(b))
    def chain(e, sign=1):
        b = encode(e)
        for _ in range(3):
            out = respond(b, sign)
            if sign == 1:
                assert out == -b/(2*r**2)
            b = out
        return b
    # Orthogonal (unnormalized) STF basis. Division by its Gram diagonal
    # implements the orthonormal contractions without irrational entries.
    basis = [diag(1,-1,0),diag(1,1,-2)]
    for i,j in [(0,1),(0,2),(1,2)]:
        e=zeros(3); e[i,j]=e[j,i]=1; basis.append(e)
    def contractions(sign):
        n=0; m=zeros(3)
        for e in basis:
            a=chain(e,sign); g=(e*e).trace()
            n+=4*(e*a).trace()/g; m+=a*e/g
        return n, Q(3,4)*n*eye(3)+2*m
    x=Matrix([2,-1,0]); y=k.cross(x)/r
    helicities=[]
    for h in [-1,1]:
        v=x+h*I*y
        b=v*v.T
        assert (curl(b)-h*r*b).applyfunc(simplify)==zeros(3)
        assert (respond(b)+b/(2*r**2)).applyfunc(simplify)==zeros(3)
        helicities.append(str(h*r))
    n, york=contractions(1)
    direct=(-r**2*eye(3)-k*k.T/3).inv()
    assert n == 1/(-k.dot(k)) and york==direct
    wrong,_=contractions(-1)
    truncated=Q(3,4)*n*eye(3)
    assert wrong!=n and truncated*k==direct*k and truncated*x!=direct*x
    return dict(frequency=list(map(int,k)),curl_eigenvalues=helicities,
                dual_factor=str(1/(16*r**4)),one_response=str(-1/(2*r**2)),
                scalar=str(n),constraint_scalar=str(-n/2),york=str(york),
                wrong_dual_scalar=str(wrong),missing_contraction_transverse=str((truncated-direct)*x))
