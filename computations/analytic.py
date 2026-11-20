"""Original Segre/half-form action and independent scalar wave column at two orders."""
from sympy import I, Matrix, Rational as Q, diag, diff, simplify, sqrt, symbols

def run():
    t,u,v=symbols('t u v', real=True, positive=True)
    p=Matrix([-t*u,-t*v,t*(1-u*u-v*v)/2]); H=t*(1+u*u+v*v)/2
    c=t/2  # evaluated below from the wedge residue
    jac=simplify(p.jacobian([t,u,v]).det())
    assert simplify(jac/(2*H)-c)==0
    # Original construction: rank-one Hermitian matrix on the real Segre sector;
    # square-preserving half-form transport is continued from spin identity.
    spin=diag(2,Q(1,2)); spinor=Matrix([1,u+I*v])
    null=t*spinor*spinor.H
    image=spin*null*spin.H
    T=image[0,0]; Z=simplify(image[1,0]/T)
    U,V=Z.as_real_imag()
    tj=simplify(Matrix([T,U,V]).jacobian([t,u,v]).det())
    angular_j=simplify(Matrix([U,V]).jacobian([u,v]).det())
    # O(-1,-1) square-root action on the ruled sky, tensored with the
    # degree-one radial coefficient, in the root initialized at identity.
    old_to_new=simplify(1/((T/t)*sqrt(angular_j)))
    assert simplify((T/2)*tj-c)==0
    assert old_to_new==1 and spin.det()==1
    # H.3 normal evaluation has kernel (w*z,-z,-w,1).
    w0,z0=symbols('w0 z0')
    normal=Matrix([[1,w0,z0,w0*z0],[0,1,0,z0],[0,0,1,w0]])
    segre=Matrix([w0*z0,-z0,-w0,1])
    assert normal*segre==Matrix.zeros(3,1)
    # Reversing/permuting spin basis identifies this rank-one line with (1,Z).
    tx,xx,xy,xz=symbols('tx xx xy xz',real=True)
    solder=Matrix([[tx-xz,xx-I*xy],[xx+I*xy,tx+xz]])/sqrt(2)
    form=-(null*solder).trace()/sqrt(2)
    components=Matrix([diff(form,vv) for vv in [tx,xx,xy,xz]])
    assert (components-Matrix([-H,*p])).applyfunc(simplify)==Matrix.zeros(4,1)
    # Wave route: convert the same spin action to an active Lorentz matrix,
    # compute Fourier Jacobian in Cartesian variables, then normalize by KG.
    x,y,z=symbols('x y z',real=True); xi=Matrix([x,y,z]); h=sqrt(x*x+y*y+z*z)
    ch,sh=Q(17,8),Q(15,8)
    new=Matrix([x,y,ch*z+sh*h])
    at={t:2,u:1,v:Q(1,2)}
    pt=dict(zip(xi,p.subs(at)))
    image_xyz=Matrix([-image[1,0].as_real_imag()[0],-image[1,0].as_real_imag()[1],(image[0,0]-image[1,1])/2])
    assert (new.subs(pt)-image_xyz.subs(at)).applyfunc(simplify)==Matrix.zeros(3,1)
    J=new.jacobian(xi).det().subs(pt)
    h1=sqrt(new.dot(new).subs(pt)); h0=h.subs(pt)
    wave_ratio=simplify(sqrt(h1/h0)/sqrt(J))
    assert wave_ratio==old_to_new
    wrong_root=simplify(-old_to_new/wave_ratio)
    assert wrong_root!=1
    # Physical rescaling by r: original cotangent volume and cone denominator
    # are recomputed at fixed k; the supplied original square is held fixed.
    scale=3; metric=diag(scale**2,-scale**2,-scale**2,-scale**2)
    analytic_square=simplify(1/sqrt(-metric.det())/metric.inv()[0,0])
    hs=diag(*([scale**2]*3))
    wave_scale=simplify(hs.det()**(-Q(1,4))*sqrt(h0/sqrt((xi.T*hs.inv()*xi)[0].subs(pt))))
    assert analytic_square==wave_scale**2 and wave_scale==Q(1,3)
    relative_square=1/analytic_square
    initialized=1/sqrt(relative_square)
    assert initialized==wave_scale
    # Original analytic module: express its initialized flat residue half-form
    # in Cartesian covectors (jac=t*H), then use the exact half-density pullback
    # for X1=f(y1). No Riccati/left-symbol coefficient is used on this route.
    assert simplify(jac-t*H)==0
    a=1/sqrt(2*h); curvature=Q(2,3)
    analytic1=I*curvature/2*(diff(a,x)-diff(x*a,x,2))
    # Independent wave side: S=C^(1/2) satisfies S^4 D=1/4 on a static
    # half-density carrier. Solve the subleading LEFT symbol, then reduce the
    # kernel column to its coefficient at the point. Only y1 varies.
    Y=symbols('Y',real=True); fp=1+curvature*Y
    D=x*x/fp**2+y*y+z*z
    Dsub=diff(fp**(-2),Y)*x/I
    b=(4*D)**(-Q(1,4))
    other=6*b**2*diff(b,x)*diff(b,Y)*D/I + b**4*Dsub + 4*b**3*diff(b,x)*diff(D,Y)/I
    left=-other/(4*b**3*D)
    wave1=(left+I*diff(b,Y,x)).subs(Y,0)
    rays=[pt,dict(zip(xi,new.subs(pt)))]
    checks=[]
    for ray in rays:
        lead=-I*a.subs(ray)
        lower=simplify(analytic1.subs(ray))
        assert simplify(b.subs(Y,0).subs(ray)-a.subs(ray))==0
        assert lower!=0 and simplify(wave1.subs(ray)-lower)==0
        assert simplify(lead.conjugate()+lead)==simplify(lower.conjugate()+lower)==0
        # Omitting analytic half-density Jacobian or the wave column reduction fails.
        wrong=-I*curvature*diff(x*a,x,2).subs(ray)/2
        assert simplify(wrong-lower)!=0 and simplify(left.subs(Y,0).subs(ray)-lower)!=0
        checks.append(dict(xi=list(map(str,ray.values())),degree1=str(lead),degree0=str(lower)))
    return dict(spin='diag(2,1/2)',ruled_map=[str(T),str(U),str(V)],
                cartesian_jacobian=str(J),analytic_frame=str(old_to_new),wave_frame=str(wave_ratio),
                frame_ratio=str(old_to_new/wave_ratio),wrong_initialized_root=str(wrong_root),
                metric_square_at_scale3=str(analytic_square),relative_unit_scale=str(wave_scale),
                formal_column=checks,even_parity_fails=True)
