"""Mode elimination in the unit-sphere divergence coordinate t=dT."""
from sympy import Function, Symbol, Rational, diff, expand, simplify, sqrt

def run():
    r=Symbol('r',positive=True); l=Symbol('l',integer=True,positive=True)
    lam=l*(l+1)
    e,t,z,a,f=(Function(n)(r) for n in ('e','t','z','a','f'))
    # Exact angular coordinate: D Y is the basis vector, dT=t D Y.
    # Czimek (4.53),(4.56) in the physical frame.
    sigmaE=f/2+z
    electric=r*(f/2-sigmaE-a/(2*r)-e/r)
    assert simplify(electric-(-a/2-e-r*z))==0
    # Coexact: divide the former scalar curl equation by its nonzero
    # unit-sphere curl factor. sigma_NN=e/r and rho is the tangent divergence.
    sigma=e/r
    rho=diff(e,r)+3*e/r+t/r
    lhs=diff(sigma,r,2)+3*diff(sigma,r)/r-(lam+3)*sigma/r**2
    rhs=diff((rho+z)/r,r)
    normalized=expand((rhs-lhs)*r**2)
    expected=diff(t,r)-2*t/r+diff(e,r)+(lam-2)*e/r+r*diff(z,r)-z
    assert simplify(normalized-expected)==0
    # A compactly supported coexact perturbation with divergence zero.
    # e is any bump supported away from the cutoff; set dT accordingly.
    free_t=-r*diff(e,r)-3*e
    assert simplify(rho.subs(t,free_t))==0
    # At r=2 choose a bump equal to one near 2 (first two derivatives zero).
    bad=simplify(expected.subs({t:free_t,z:0}).doit().subs(e,1).doit().subs(r,2))
    assert bad.subs(l,2)!=0
    wrong_shift=expected+2*e/r
    assert simplify((wrong_shift-normalized).subs(e,1).subs(r,2))==1
    # Cutoff factor chi'(l(r-1)) in S_r-normalized harmonics; conversion to
    # a radial parallel unit frame divides by r. Test with nonzero moments.
    mid=[]; chi=Function('chi')
    for degree in [2,3]:
        u=1+Rational(11,20*degree); nu=sqrt(degree*(degree+1)+4)
        # Use a unit-mass bump b on (2,3) for both residual components.
        # The weighted moment integrands from (4.44),(4.49) reduce to b.
        bump=Function('b')(r)
        fE=-r**(degree-1)*bump; fH=-r**(1+nu)*bump
        assert simplify(-r**(1-degree)*fE-bump)==simplify(-r**(-1-nu)*fH-bump)==0
        for exponent in (degree-1,1+nu):
            normalized_mode=r**exponent*diff(chi(degree*(r-1)),r)
            physical=normalized_mode/r
            assert simplify(physical-r**(exponent-1)*degree*diff(chi(Symbol('v')),Symbol('v')).subs(Symbol('v'),degree*(r-1)))==0
            mid.append(str(simplify((normalized_mode/physical).subs(r,u))))
        assert Rational(1,10)<degree*(u-1)<1
    # l=0,1 have no STF tensor harmonic: squared norm factor of tracefree Hessian.
    norm=lam*(lam-2)/2
    assert norm.subs(l,0)==norm.subs(l,1)==0 and norm.subs(l,2)>0
    return dict(electric_residual='0',magnetic_residual='0',degrees='symbolic l>=2',
                divergence_free_wrong_range_l2=str(bad.subs(l,2)),
                wrong_magnetic_shift_residual='2*e/r (wrong minus correct)',
                cutoff_missing_frame_factor=mid,low_tensor_norm=[str(norm.subs(l,j)) for j in range(4)])
