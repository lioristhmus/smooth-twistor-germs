# York inversion, free-space transport and a protected plane-wave round trip

These computations accompany Paper 3, Lior Isthmus, *A two-chirality twistor correspondence for smooth Lorentzian Ricci-flat germs*, v1.0. Experiments 3.1 and 3.2 start with compact sources on flat space. Experiments 3.3 and 3.4 use the same smooth, nonanalytic vacuum plane wave. The runner first extends its marked slice (3.4), then reconstructs the local wave from signed CR data (3.3). [Paper v1.0](https://doi.org/10.5281/zenodo.24385872)

### 3.1 · Recovering the free data

Definition 3.11, Theorems 3.6 and 4.12, and Appendix A. Can the fourth-order TT encoder, three negative contact responses and final contraction recover both longitudinal and transverse components? Applying the York operator to a known compact field gives a source whose reconstructed field can be compared with the original. The largest error on the finest set of probes is 1.01e−6. A second test uses a localized source with nonzero mean. The spectral reconstruction and a separate real-space quadrature of the free-space Green tensor agree to 2.33e−12 at the sampled points. The table distinguishes the measured field multiplied by radius from the limiting coefficients of the Green tensor. Directions are relative to the source polarization. [Encoder and responses](code/york/chain.py) · [Two quadratures](code/york/green.py) · [Recovery and far-field data](reference/york/green/results.json)

| Direction | Measured at radius 16 | Analytic coefficient at infinity |
|---|---:|---:|
| Perpendicular | −0.03424478 | −0.03424340 |
| Parallel | −0.03913257 | −0.03913532 |

Separate refinements vary radial sampling, frequency cutoff and source-domain clipping. Removing the mean on a periodic box tests the effect of imposing a different exterior condition. For the cochain tests, the input consists of two symmetric trace-free polarizations of a compact smooth bump, made transverse-traceless (TT) by a fourth-order differential operator. The finite antiperiodic computations examine closure, reality, admissible roots and effective primitives. Exact rational rank tests in degrees 0–4 detect positive-chirality polynomial obstructions in degrees 3 and 4. The negative-chirality test tensors are divergence-free by their explicit form and have a nonzero polynomial slot. The canonical lift has zero polynomial remainder by construction. [Cochain results](reference/york/cochains/results.json) · [Exact rank calculations](code/extension/polynomial.py) · [Rank table](reference/extension/polynomial/results.json)

### 3.2 · Initialized transport and the nonlinear far field

Theorem 4.18 and Appendix A. Start with flat data and prescribe q = 0 and p = εp₀. The bump profile, radius and encoder gain are given. Fitting five polarization coefficients per seed to the saved chiral roots recovers p₀ within this radial family. The worst relative fit is 1.47e−15. An input-only Parseval quadrature takes the squared tensor norm of the **sum** of the polarizations. With the gain included in p₀ and ε applied once, the quadratic equation −2Δu₂ = |p₀|² predicts C₂ = (8π)⁻¹∫|p₀|² dx = 0.123378596859678. At ε = 0 the rescaled field equation computes this quadratic coefficient directly. Its result agrees with the input integral to a relative error of 7.41e−11. A separate zero-seed control returns a zero physical field correction. [Root fit](code/transport/continuum.py) · [Fit residuals](reference/transport/cr/results.json) · [Input quadrature](code/far/reference.py) · [Quadratic and zero-seed checks](reference/checks/far-readback/results.json)

At finite ε the initialized midpoint transport and a separate Picard iteration solve for the fields by different algorithms. They share the radial and angular grid, free-space Poisson inverse, exterior continuation and monopole readout. The fields determine their own exterior coefficient. At ε = 0.008 on the fine grid, C/ε² is 0.1233779685 on the straight path (four midpoint steps), 0.1233779618 on the broken path and 0.1233779551 for Picard. The straight-path discrepancy from Picard is 1.34e−8. On the medium grid it falls from 5.35e−8 to 1.34e−8 to 3.34e−9 at two, four and eight steps. Spherical means and gradient fluxes reread from the saved fields check the extraction of the same monopole. Grid, exterior-radius and tolerance variations are recorded separately. The zero difference in the tight-tolerance gate compares matching settings. Earlier antiperiodic endpoints are also included as controls. [Initialized transport](code/transport/transport.py) · [Picard solver](code/far/solver.py) · [Fine-grid endpoints](reference/transport/fine/results.json) · [Step refinement](reference/transport/medium/results.json) · [Field readback](reference/transport/readback/results.json)

### 3.3 · Reconstructing the nonanalytic vacuum wave

Definition 3.11, Theorem 4.18, §§3–5 and Appendix A. The profile on the marked maximal Rosen slice is F(u) = 0 for u ≤ 0 and 0.012 exp(−1/u) for u > 0. The extractor fixes the gauge on the extended data before the Fourier–Bessel transform. The reconstruction reads the saved signed roots and coordinate marking. The two Killing directions, spin frame and physical scale are inputs. We compare the recovered rising-side tidal profile with F, measure leakage on the flat side and evolve a Brinkmann null geodesic with the reconstructed and original profiles from the same initial conditions. The geodesic column reports the largest final component difference in this vacuum development. [Extraction and readout](code/extension/pp.py) · [Window results](reference/extension/native-filter/results.json) · [Tidal waveform](reference/extension/report/waveform.png)

| Upper frequency of the smooth window | Rising-side tidal error | Flat-side leakage | Geodesic error |
|---:|---:|---:|---:|
| 48 | 4.17e−5 | 2.10e−5 | 7.24e−6 |
| 64 | 2.23e−5 | 2.09e−5 | 3.98e−6 |
| 96 | 6.62e−6 | 6.26e−6 | 1.64e−6 |

The band-96 direct reconstruction uses four Picard updates. Its tidal error is 0.20% of the rising-side signal, 0.00332619. The scale check doubles lengths at readout by rescaling the recovered h and K. The resulting curvature error is 1.66e−6, or 4.15e−4 if the K rescaling is omitted. This checks the dimensions of the readout on the same reconstruction. [Windowed reconstruction](code/extension/filters.py) · [Scale measurements](reference/extension/native-filter/results.json)

The two-path experiment uses a separate **smooth band-32 representative**. Its straight-path tidal error is 3.53e−4, or 10.6% of the same signal. At four midpoint steps per segment the straight and q-then-p paths differ by 7.17e−11 in K. This latter number measures agreement of the two computed endpoints for that input. The path panel in the convergence figure uses band 32. [Two-path integration](code/extension/current.py) · [Endpoint and tidal errors](reference/extension/native-path4/results.json) · [Convergence plots](reference/extension/report/diagnostics.png)

Radial interpolation, angular TT projection and Hermitian symmetry select one smooth Fourier representative on a compact annulus. A TT bump supported on an unsampled frequency ball and its antipode can change the field while preserving every recorded shell. The paper gives no numerical lower bound for the smallness radius at this amplitude. The residuals and refinements quantify the finite models. A certified error to the continuous nonlinear solution would also require control of discretization and exterior truncation. [Representative selection](code/extension/smooth_packet.py) · [Acceptance criteria and results](reference/extension/report/results.json)

### 3.4 · Extending maximal data while preserving the interior

Theorem B.1 and Lemmas B.2 and B.10. The exterior correction holds the polar lapse fixed and uses the specified weighted scalar and momentum right inverses. Separate quadratures and an adjoint ODE supply reference collar coefficients for degrees 1–4 and weight exponents 0 and 0.001. At the finest order the largest relative discrepancy is 3.93e−8 or less. The collars for degrees 2–4 lie where the weight equals one. A separate differential residual test uses a single harmonic (ℓ, m) = (3, −1), with radial, electric and magnetic momentum components. Its finest relative scalar and momentum residuals are 1.14e−3 and 1.69e−4. The normal–normal component of the selected flat metric variation vanishes algebraically because the shift is tangential. The older radial correction illustrates the different range. [Range equations](exterior-range.md) · [Selected inverse](code/extension/modal.py) · [Reference quadratures](code/extension/moments.py) · [Coefficient comparisons](reference/extension/moments/results.json) · [Single-harmonic residuals](reference/extension/probe3/results.json)

On the extended wave slice the native-grid readback covers 472 radial nodes. In the protected interior, the largest changes in h and K are zero and 2.17e−19. Outside, the Hamiltonian and momentum maxima fall from 0.01283 and 0.004042 to 7.33e−4 and 7.52e−5. At this resolution the Hamiltonian residual levels off near radius 1.03, just outside the protected ball. Transferring the data to a smaller grid raises its maximum to 0.0458. These exterior residuals measure the extension directly. The local tidal readout in 3.3 samples the protected wave. [Native iteration history](reference/extension/native3/results.json) · [Native and resampled constraints](reference/extension/fieldcheck/results.json)

The radial tail is sampled outside the seed. A separate 1% amplitude change gives maximum field changes per unit amplitude of 0.415 in h and 0.337 in K on the tested grid. [Exterior variation](reference/extension/ext2/results.json)

## Reproduce from the archive

Use a POSIX host with Python 3.10 or later, Docker CLI and a local Linux Docker engine with cgroup v2, a shared bind-mount directory and at least 2 GiB of VM memory. The host runner uses only the Python standard library. Start Docker first. With network access, obtain the public runtime images by immutable digest:

```sh
docker pull --platform linux/amd64 sagemath/sagemath@sha256:e068670ae5863b54b2550e72437ec637b0283acb0dc712c8584c124dbf44e667
```

The SageMath 10.9 image supplies Linux x86-64 Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1 and Matplotlib 3.10.8. The tested host used Docker on macOS with Linux image emulation available. Enable emulation for any required non-native architecture. Different architectures, libraries and independently rebuilt images may produce different floating-point bytes; the exact-output test below applies to the pinned environments.

On a host with existing background services, identify them with `docker ps`. For each running container that you have checked is an unrelated background service, put its name in the space-separated `MATH_BACKGROUND_SERVICES` environment variable. For example, for two verified background services named svc1 and svc2, use `export MATH_BACKGROUND_SERVICES="svc1 svc2"`. The default is empty. An unlisted running workload stops the run. All cooperating mathematics jobs on the machine must use the same atomic directory lock, `~/.local/state/math-job.lock`, under one shared execution account. The runner records ownership, checks live VM and cgroup limits and releases the lock after confirming container termination. Each foreground computation runs offline with at most one CPU, no swap, 1,200 MiB memory or less, 48 PIDs and a ten-minute limit; temporary storage and logs are bounded.

Place the ZIP and its matching .zip.sha256 file together in a new, otherwise empty directory. From there, after obtaining the required images, run the following command chain. Extraction proceeds only if the external SHA-256 check succeeds.

```sh
shasum -a 256 -c paper3-smooth-twistor-additional-computations-2026-11-24-r2.zip.sha256 &&
unzip paper3-smooth-twistor-additional-computations-2026-11-24-r2.zip &&
python3 paper3-smooth-twistor-additional-computations-2026-11-24-r2/reproduce.py results
```

The runner first verifies the unpacked manifest. It creates a fresh sibling output directory and executes the ordered dependency list in [jobs.json](jobs.json). To run again, use a new output directory name. Inputs to each job are generated in earlier stages or supplied by the experiment code; the saved reference results are used for comparison. The sequence is York recovery and CR encoding, the independent free-space solver and field readback, CR-to-seed transport, then the selected exterior inverse and wave reconstruction. The native-grid stages and report contain the final extension results.

Expected terminal output is `PASS: 56 jobs, 8178 scientific files`. The generated `results/verification.json` lists the matched file counts; each stage has a UTC execution receipt, live limits, logs and its scientific outputs. The [reference directory](reference) contains the measured numerical tables and figures. The [expected digests](expected-sha256.json) also cover regenerated intermediate data. Timestamps and resource counters are recorded separately from that comparison.

Copyright 2026 Lior Isthmus. See [licenses and attribution](LICENSES.md). This additional computation package accompanies the cited paper v1.0. Its date is 2026-11-24 UTC.
