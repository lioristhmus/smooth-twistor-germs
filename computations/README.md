# Computational supplement

This supplement accompanies [A two-chirality twistor correspondence for smooth Lorentzian Ricci-flat germs](../paper/smooth-twistor-germs-en-v1.1.md). These are the original version 1.0 checks. All recorded comparisons use exact arithmetic or symbolic differentiation in SymPy 1.14.0 with Python 3.12.3, from the SageMath 10.9 Linux/amd64 container. No discretization or tolerance enters the comparisons.

| Proof in the paper | Input and comparison | Code | Recorded result and execution |
|---|---|---|---|
| Propositions 4.9–4.10, Lemma 4.11, Theorem 4.12 | Nonaxial Fourier covector, two helicities, full scalar and vector inverse | [green.py](green.py) | [result](results/green.json) · [receipt](results/green-receipt.json) |
| Lemma B.10, used in Proposition B.11 | Exact and coexact tensor modes, radial frame, boundary profiles | [exterior.py](exterior.py) | [result](results/exterior.json) · [receipt](results/exterior-receipt.json) |
| Propositions 6.5, 6.7, 6.16, Theorem 6.6 | Three boosted rays and an exact time-dependent Gaussian mode | [wave.py](wave.py) | [result](results/wave.json) · [receipt](results/wave-receipt.json) |
| Propositions 9.2–9.6 | Physical spin arrow, initialized half-form, scale, two nonzero orders | [analytic.py](analytic.py) | [result](results/analytic.json) · [receipt](results/analytic-receipt.json) |

## Reproduction

Use an idle Docker engine with Linux/amd64 support and the SageMath image below available locally. The runner uses host Python 3 and executes each check in a separate, network-isolated container. From this directory, run the four commands in order, choosing a new output directory.

```sh
python3 run.py green    --output ./reproduced
python3 run.py exterior --output ./reproduced
python3 run.py wave     --output ./reproduced
python3 run.py analytic --output ./reproduced
```

Compare each result in `reproduced/` with the corresponding JSON file in `results/`, for example with `cmp reproduced/green.json results/green.json`. The recorded image is `sagemath/sagemath@sha256:e068670ae5863b54b2550e72437ec637b0283acb0dc712c8584c124dbf44e667`. Each job has one CPU, 768 MiB of memory with no extra swap, and a 150-second container deadline. The [runner](run.py) starts the [entry script](check.py) with the chosen case. Inputs are specified in the four case scripts.

The execution records give the observed image, resource limits, exit state, and digests for the original checks and saved results. They also identify the original receipt and execution wrapper by SHA-256. The reproduction wrapper uses the same calculation command and resource limits.

## 1. From the negative contact response to the York inverse

Use the Fourier convention of §3.3 with $k=(1,2,2)$, $r=3$. The two calibration constants $4i\varepsilon r$ and $i\varepsilon/(4r^3)$ give their ratio $1/1296$ on both branches. The root sum of Proposition 4.9 gives $2i\mathsf C B$ and the graph of Theorem 4.7 reads $4i\mathsf C T$. The code implements the curl as the symmetrized cross-product matrix, and the fourth-order differential encoder using $D=kk^t-r^2I$. For each of the five orthogonal trace-free basis tensors it computes the encoded value followed by three complete responses. Division by the squared basis norm implements orthonormal contraction. The intermediate identity is tested at each call. Separately, with $x=(2,-1,0)$ and $y=k\times x/r$, $(x\pm iy)^{\otimes2}$ have opposite curl eigenvalues; both acquire the response $-1/18$.

The oracle Fourier-transforms $\Delta$ and $\Delta_L=\Delta I+\nabla\operatorname{div}/3$ directly and inverts the resulting matrix. The source scalar is $-1/9$, its constraint block is $1/18$, and the vector inverse is
$$
\frac1{324}\begin{pmatrix}-35&2&2\\2&-32&4\\2&4&-32\end{pmatrix}.
$$
Reversing the dual sign changes the scalar inverse to $+1/9$. Dropping the partial-trace term $2\mathcal M$ passes the longitudinal test and gives the nonzero error $(1/18,-1/36,0)$ on $x$. These checks use the established contour calibration and test its propagation through the conic readout and contractions. The contour theorem, the weighted zero-frequency extension, and the nonlinear convergence and current inverse in §4.8 and Appendix A are separate arguments. [Calibration](../paper/smooth-twistor-germs-en-v1.1.md#eq:online-root-calibration-factors) · [Flat Green operator](../paper/smooth-twistor-germs-en-v1.1.md#eq:online-negative-response-full-green)

A reader can turn the mode into a smooth packet: multiply the two helicity projectors applied to a fixed seed by a smooth bump supported in a small ball about $(1,2,2)$ disjoint from zero. For example use the projections $(I\pm\mathsf C/|k|)\Pi/2$ on a tensor whose transverse trace-free part has both polarizations there. Adding the conjugate reflected packet gives a real tensor. This example describes a frequency-localized input; the stored run evaluates the stated covector.

## 2. The selected exterior momentum range

Use physical orthonormal components on the unit sphere transported along each ray. Write $t=dT$ for the unit-sphere divergence of the trace-free angular tensor. On $l\ge2$ this identifies each of its two polarizations with the corresponding one-form mode. In the electric sector use $DY$ as basis, so that the coefficient of $Da$ is $a$. Let $e$ be a mixed radial-angular component, $z$ the corresponding profile, and $f$ a tangential residual. The two source Hodge equations give $\sigma_E=f/2+z$ and $t=r(f/2-\sigma_E)-a/2-e$. The script substitutes the first transcribed relation into the second and compares the result with $-a/2-e-rz$. The zero residual checks consistency of the transcription and substitution. Both source relations enter as inputs.

In the coexact sector the source scalar curl is proportional to $e/r$. Its radial equation is independently implemented as
$$
(\partial_r^2+3r^{-1}\partial_r-(l(l+1)+3)r^{-2})(e/r)
  =\partial_r((\rho+z)/r),\qquad \rho=e'+3e/r+t/r .
$$
Subtracting the two sides and clearing the radial factor gives the magnetic B.10 equation, coefficient for coefficient, for symbolic $l\ge2$ and arbitrary radial jets. The $l=0,1$ trace-free tensor modes are absent, as shown by the trace-free Hessian norm factor $l(l+1)(l(l+1)-2)/2$. The source equations and the two profile powers are from Czimek §4.3, (4.38), (4.43)–(4.44), (4.47)–(4.50), (4.53), (4.56). [Czimek, arXiv v2](https://arxiv.org/abs/1609.08814v2) · [B.10 magnetic identity](../paper/smooth-twistor-germs-en-v1.1.md#eq:online-momentum-magnetic-range)

For boundary profiles take a unit-integral smooth bump $b$ in $(2,3)$, $f_N=0$, $f_E=-r^{l-1}b$, $f_H=-r^{1+\sqrt{l(l+1)+4}}b$ in the source normalization. Both moment integrands reduce exactly to $b$. The powers of $r$ in these test inputs are prescribed. The script also sets the physical component equal to the normalized harmonic divided by $r$, using the frame convention derived in Appendix B. It evaluates the ratio of the prescribed components, which is $r$ wherever the denominator is nonzero. At $l(r-1)=11/20$ in the transition band, omitting it gives erroneous factors $51/40$ and $71/60$ for $l=2,3$ wherever the cutoff derivative is nonzero (take a smooth monotone transition with positive slope there). Removing $+2$ from $\Delta_H+2$ leaves an error $2e/r$ in the residual with the code's sign convention.

For a stronger negative control set $a=0$, take a coexact mixed component $e$ compactly supported beyond the cutoff, equal to one near $r=2$, and choose $dT=-re'-3e$. This perturbation has zero radial and tangential divergence. Its magnetic range residual at $l=2,r=2$ equals **5**, even though its source and profile vanish. This is the extra information used by B.11. The low-degree norm values here evaluate the supplied trace-free Hessian factor. Appendix B proves the low-mode equations for the remaining scalar and mixed components, the full identity $M_\delta J=I$, distributional zero extension, uniform weighted estimates over all harmonics and all-order regularity of the same nonlinear branch.

## 3. Doppler factors and time-dependent half-densities

For two Minkowski hyperplanes use $(\cosh\theta,\sinh\theta)=(5/4,3/4)$ and the three incident covectors in the code. Differentiate the Cartesian spatial Fourier boost to obtain its Jacobian; independently compute the new frequency length. Their ratio law gives degree-zero amplitude mismatches $1,3/2,3/4$ and degree-one ratios $1,1,1$. One of the directions alone would have missed the defect. [Changing carriers](../paper/smooth-twistor-germs-en-v1.1.md#sec:online-covariant-wave-enrichment)

Next write flat spacetime as $dt^2-s^2dx_1^2-dx_2^2-dx_3^2$, $s=1+t>0$. The change to inertial coordinates is $(s\cosh x_1,s\sinh x_1,x_2,x_3)$. For $b>0$ use $e^{ibx_1}s^{-ib}$. Direct substitution into the divergence-form scalar wave equation gives zero. From the mode itself obtain $Q=ib/s$, $B=(ib-1/2)/s$, and charge density $2b$. The correction relative to the instantaneous generator is $-1/(2s)=\dot H/(2H)$. The opposite temporal phase also solves the wave equation but has charge $-2b$. These are exact mode and Jacobian checks. General FIO propagation, joint smoothing, all-direction transport and full Hom remain the conclusions of §§6–7. [Normal generator](../paper/smooth-twistor-germs-en-v1.1.md#eq:online-wave-corrected-generator) · [Charge](../paper/smooth-twistor-germs-en-v1.1.md#eq:online-wave-klein-gordon-form)

## 4. From the calibrated analytic frame to a nonzero lower wave coefficient

The example is the pointed flat Lorentz germ, on the time-zero disk and its image under a specified spin boost. Work near a nonzero past covector, with the positive radial coordinate and the continuous square root initialized at the identity. We use the analytic source's homogeneous ray, relative half-canonical square, exact half-density substitution and framed simple module. The smooth paper supplies the degree-one evaluation compared in this module. The analytic predecessor on its own does not select a smooth scalar wave state. The calculation tests the chosen interface on this initialized example. [Analytic v1.1, §3, D.7, G.2 and H.1–H.3](https://doi.org/10.5281/zenodo.24197319) · [Relative root](../paper/smooth-twistor-germs-en-v1.1.md#prop:relative-residue-calibration) · [Odd conjugation](../paper/smooth-twistor-germs-en-v1.1.md#prop:analytic-native-real-parity)

**Frames.** In the affine ruled chart of H.3 the first-jet matrix kills $(wz,-z,-w,1)$; this is a rank-one spinor tensor. On its Lorentz real locus use the dual spin frame $P=t(1,Z)(1,Z)^*$ with $Z=u+iv$ and $t>0$ (the spinor before its adjoint is a column). Pair it with the marked tangent soldering $X$ of §3 by $-\operatorname{tr}(PX)/\sqrt2$. Direct differentiation gives the past covector
$$
(-H,\xi)=\big(-t(1+u^2+v^2)/2,-tu,-tv,t(1-u^2-v^2)/2\big).
$$
Specify the lift by its action $\operatorname{diag}(2,1/2)$ on dual spinors, or the inverse adjoint on tangent spinors. It sends $(t,u,v)$ to $(4t,u/4,v/4)$. On the initialized analytic half-form the square root of the projective area Jacobian is $1/4$; the degree-one radial coefficient contributes $4$. The root and its separately initialized local-system frame use the lift continuous from the identity. A reversal of the initialized frame changes the comparison sign.

Independently the physical wave calculation applies the Cartesian Lorentz matrix $(\cosh\theta,\sinh\theta)=(17/8,15/8)$ along the third axis and differentiates its spatial frequency map. The code checks that the covector is exactly the one obtained from $P$. At $(t,u,v)=(2,1,1/2)$ the Cartesian Jacobian is $23/12$. Fourier half-density transport and the positive-frequency normalization $(2H)^{-1/2}$ therefore give the same normalized frame, with ratio **+1**. Squaring the frames alone would also accept the erroneous ratio **−1**. This test keeps a specified spin lift; it does not infer a spin lift from scalar amplitudes.

At constant physical scale $g\mapsto9g$, recompute the analytic cotangent volume and the normal derivative of the null quadric at fixed covector. The native residue square changes by $1/9$. Holding the supplied original root square fixed changes the relative quadratic form by $9$ and its initialized unit by $1/3$. Separately, the wave coefficient uses $(\det h)^{-1/4}(2H)^{-1/2}$ with the scaled spatial metric and changes by $1/3$. This is the simultaneous scaling convention of Proposition 9.4.

**Two orders of the column.** On each disk use analytic coordinates $X_1=y_1+cy_1^2/2$, $X_2=y_2$, $X_3=y_3$, with $c=2/3$, at the origin. Suppress only the common real Fourier constant $d_0=(2\pi)^{-3/2}$. In the initial affine chart the residue has density $(t/2)\,dt\,du\,dv$ and the Cartesian frequency Jacobian is $tH$, so its initialized degree-one physical coefficient is $a=(2H)^{-1/2}$. This fixes the principal input of the analytic calculation. The flat affine evaluation has no lower amplitudes before the chart change.

* Analytic module route. Use the exact base-substitution action on half-densities (v1.1 H.2, as used in Proposition 9.2). Its amplitude acquires $\sqrt{1+cy_1}$ and its phase an extra $c y_1^2\xi_1/2$. Taylor expansion and integration by parts at the basepoint give the physical term of degree $-3/2$,
$$
\frac{ic}{2}\big(\partial_{\xi_1}a-\partial_{\xi_1}^2(\xi_1a)\big).
$$
* Wave route. Compute in the new coordinates from the static half-density Laplacian. Put $D_2=\xi_1^2/(1+cy_1)^2+\xi_2^2+\xi_3^2$ and $D_1=\xi_1\partial_{y_1}(1+cy_1)^{-2}/i$. The static column operator $S=C^{1/2}$ obeys $S^4D_h=1/4$. With principal left symbol $b=(4D_2)^{-1/4}$, solve the next degree of this equation for its left symbol $b_{-3/2}$. Reducing the kernel column at the origin adds $i\partial_{y_1}\partial_{\xi_1}b$. This route uses neither the transformed phase nor the analytic coefficient formula.

The physical orders $-1/2,-3/2$ correspond to module degrees $1,0$. Rephasing multiplies their coefficients by $-i,1$, respectively. At $\xi=(-2,-1,-1/4)$ the two formal coefficients are $-i\sqrt2/3$ and $-32i\sqrt2/59049$. At the spin-transformed ray $\xi'=(-2,-1,59/16)$ both are again nonzero; the exact values are in the [result](results/analytic.json). The two routes agree at both orders and both rays. Conjugation of the formal coefficients at the **same** ray gives their negatives. Dropping the analytic Jacobian or omitting the wave kernel-column correction gives different answers at degree zero. These terms reflect a nonlinear chart on a flat metric and should not be interpreted as nonzero curvature. General analytic contact quantization, overlap cells, all-degree parity and comparison on other analytic germs use the proofs of §9 and their stated v1.1 inputs.
