"""Boost in real Fourier variables; scalar divergence-form oracle in a Milne chart."""
from sympy import I, Matrix, Rational as Q, diff, exp, log, simplify, sqrt, symbols

def run():
    x,y,z=symbols('x y z', real=True); k=Matrix([x,y,z]); H=sqrt(k.dot(k))
    ch,sh=Q(5,4),Q(3,4)
    boosted=Matrix([ch*x-sh*H,y,z]); jac=boosted.jacobian(k).det()
    ratios=[]
    for pt in [(1,2,2),(-1,2,2),(2,1,2)]:
        sub=dict(zip(k,pt))
        source=H.subs(sub); target=sqrt(boosted.dot(boosted).subs(sub))
        j=jac.subs(sub)
        assert j==target/source
        ratios.append(str(j))
        for degree,wanted in [(0,j),(1,1)]:
            ratio=simplify((source/target)**(degree-Q(3,2))/sqrt(j))
            assert ratio==wanted
    assert len(set(ratios))>1
    s,b=symbols('s b',positive=True)
    u=exp(-I*b*log(s)); v=sqrt(s)*u
    wave=diff(s*diff(u,s),s)/s + b**2*u/s**2
    assert simplify(wave)==0
    Qwave=simplify(-diff(u,s)/u)
    B=simplify(-diff(v,s)/v)
    charge=simplify(I*s*(u.conjugate()*diff(u,s)-diff(u,s).conjugate()*u))
    assert simplify(B-I*b/s-diff(b/s,s)/(2*(b/s)))==0
    assert Qwave==I*b/s and charge==2*b
    assert simplify(I*s*(u*diff(u.conjugate(),s)-diff(u,s)*u.conjugate()))==-charge
    return dict(boost_degree_zero=ratios,boost_degree_one=[1]*3,branch=str(Qwave),
                half_density_generator=str(B),missing_amplitude=str(simplify(B-I*b/s)),
                klein_gordon_charge=str(charge),opposite_charge=str(-charge))
