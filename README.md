# Lag-spectrum diagnostics for optical timing

A **45–60 minute camera, optics, vision and mathematics lesson**: can two lag-covariance eigenvalues reveal a temporal change hidden by a one-correlation model, and does the resulting alarm improve an optical-delay measurement? This is a small diagnostic extension of the [two-mode lab](https://github.com/ahmetalpkilic/two-mode-optical-timing-learning-lab), not a general covariance learner.

## Learn and run

Objectives: derive an affine-invariant lag spectrum, calibrate a family alarm using independent simulations, distinguish detection power from metrology benefit, and validate the link from a pupil to timing error. Prerequisites: covariance matrices, generalized eigenvalues, least squares and basic Fourier optics; Python 3.9–3.12 and a POSIX shell.

```sh
git clone https://github.com/ahmetalpkilic/lag-spectrum-optical-timing-learning-lab.git
cd lag-spectrum-optical-timing-learning-lab
sh run_demo.sh
```

One command creates `.venv`, installs pinned dependencies, runs tests and writes `outputs/results.json` and `outputs/experiment.png`. First installation requires internet; thereafter `.venv/bin/python lab.py` runs offline. Allow several minutes. Reference results are in `examples/`; last digits may vary across platforms. Spend 10 minutes on the physical model, 15 on the derivation, 15 on the experiment and 10–20 on an exercise.

| Area | Forward or measurement model | Connection |
|---|---|---|
| Optics | Fourier intensity of a 550 nm Gaussian pupil | Sets the point-spread width |
| Camera | 16 mm focal length, 4 µm pixels, pixel integration, read noise, exposure time tag | Converts the spot into sampled intensities and timestamps |
| Vision / metrology | Subpixel spot fit, two-clock calibration, exposure-delay inference | Turns image position into time |
| Mathematics | Generalized eigenvalues, affine invariance, rank thresholds, paired Monte Carlo | Tests whether temporal-shape alarms help the measurement |

## Physical conventions and assumptions

An untruncated Gaussian pupil has amplitude `exp(-(x²+y²)/a²)` with `a=1000 µm`. This radius is **not a hard aperture or f-number**. Monochromatic scalar paraxial propagation gives intensity standard deviation

```text
sigma_px = lambda*f/(2*pi*a*p) = 0.350140875 px,
mu_j(u) = B + F[Phi((j+.5-u)/sigma)-Phi((j-.5-u)/sigma)].
```

Here `lambda=.550 µm`, `f=16000 µm`, `p=4 µm`, flux `F=10000 ADU`, background `B=20 ADU`; `Phi` is the normal CDF. The data are a horizontal marginal of a separable 2-D PSF, not a literal sensor row. Signed pixel centers j=−16,…,16 increase rightward, with integration over `[j−.5,j+.5]`. Intrinsics are `K=diag(4000,4000,1)` pixels, principal point (0,0), no skew or distortion. Independent Gaussian read noise has SD 2 ADU per marginal sample. Width, flux, background and motion are known; no shot noise, saturation, quantization or finite exposure blur is modeled.

Known motion is `u(r)=.23+.8*r` px, with r in reference seconds. The clocks obey `r=alpha*t+beta`, true `alpha=1.002 reference s/master s`, `beta=.030 reference s`. At frame timestamp `T=2 master s`, true delay is `delta=.040 master s`. Fit image center u by least squares, then infer

```text
delta_hat = [(u_hat-.23)/.8-beta_hat]/alpha_hat - T.
d(delta) ≈ du/(.8*alpha) - d(beta)/alpha - (T+delta)*d(alpha)/alpha.
```

Forty independent synchronization events span latent master times [−1,1] s. Known alternating quality classes have two-clock marginal SDs (.030,.020) s and (.120,.040) s, with within-pair correlations .5 and .3. Each repeat/characterization block has a constant unknown mean and stationary temporal noise; blocks reset independently. Repeat lag has no assigned physical duration. See [MODEL.md](MODEL.md) for pupil, pixel integration, errors-in-variables clock fitting and whitening derivations.

## Derivation: a small temporal-shape diagnostic

For each class, let `e_k=L*x_k`, where the two independent unit-variance stationary modes obey `x_(k,j)=rho_j*x_(k-1,j)+sqrt(1-rho_j²)*z_(k,j)`. Stationary initialization uses `x_0~N(0,I)`. Thus

```text
C = Gamma_0 = L L^T,
Gamma_h = L diag(rho_1^h,rho_2^h) L^T,
Gamma_1 v = lambda C v  =>  lambda = rho_1 or rho_2.
```

The last equality follows by substituting `w=L^T v` and multiplying by `L^-1`. Equal rhos give the separable single-rho model. Unequal eigenvalues reveal two modes even if their average is unchanged. For any nonsingular A, transformed moments are `A Gamma_h A^T`: the generalized eigenvalues remain identical by congruence. The lab checks this on every old/pilot pair using `A=[[1.5,.4],[-.2,.8]]` plus a constant shift.

For a finite block Z, center its columns to get X, then compute

```text
C_hat = X^T X/n,
G_hat = (X[:-1]^T X[1:] + X[1:]^T X[:-1])/(2n),
ell = sorted eigvalsh(G_hat,C_hat).
```

Common normalization by n gives eigenvalues in [−1,1]: for any scalar projected sequence y, `|sum y_k*y_(k+1)| <= sum y_k²`. Centering and finite endpoints bias these estimates; they are not unbiased rho estimates. Symmetrization is appropriate for this reversible two-mode model, but discards antisymmetric lead-lag information in a general process. No identity ridge is added because it would break general affine invariance; singular inputs fail explicitly. The implementation uses SciPy's [symmetric generalized eigensolver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html).

For independent old (80 pairs/class) and pilot (40 pairs/class) blocks, compare two scores:

```text
D_spectrum = max over class and ordered eigenvalue |ell_pilot-ell_old|,
D_scalar   = max over class |rho_REML,pilot-rho_REML,old|.
```

The scalar estimator is the predecessor's bounded [−.95,.95] profile REML fit to a separable AR model, intentionally misspecified under two modes. `diagnostic.py` contains the annotated core code. Sorting avoids arbitrary eigenvector ordering, but cannot detect a rotation of mode directions with unchanged eigenvalues. This is a spectrum-change diagnostic, not a complete process equality test.

## Independent calibration and controlled experiment

Generate 999 independent matched-null old/pilot pairs. For each detector separately use the 950th ordered score as threshold, with **strict** `D>h`. For a fresh exchangeable matched-null pair, rank symmetry yields marginal false-alarm probability at most `(1000−950)/1000=5%`. Ties are conservative. This averages over training and a new old/pilot pair; it is not a conditional guarantee for a fixed deployed old estimate, a sequential monitoring guarantee, or a guarantee under model mismatch. Each detector has its own 5% bound; using their union does not retain 5%.

The test branch of `SeedSequence(20261017)` is independent of threshold training. It supplies 400 held-out trials per scenario; primitive normals are paired across scenarios and policies. No test results are used to choose thresholds. All tuning is offline synthetic work, excluded from acquisition costs.

| Scenario | Old even / odd class modes | Current even / odd class modes | Meaning |
|---|---|---|---|
| Matched null | (0,0) / (.4,.4) | (0,0) / (.4,.4) | Training distribution applies |
| Two-mode null | (−.3,.3) / (0,.8) | (−.3,.3) / (0,.8) | No change, but training model is wrong |
| Mode transition | (0,0) / (.4,.4) | (−.3,.3) / (0,.8) | Genuine shape change at constant marginal C |

Every feasible policy costs 640 paired timestamps: old160 + pilot80 + repeats400 when quiet; old160 + pilot80 + independent refresh80 + repeats320 on alarm. Repeats are 10 or 8 per event, respectively. Both detectors use exactly the same old/pilot observations. The pilot decides only whether to refresh. Whitening uses old or independent fresh scalar-rho estimates, **even for the spectrum detector**; a shape alarm does not repair the wrong covariance model. `never`/`always` are forced versions with the same pilot overhead; exact-C `oracle` is a diagnostic reference, not a feasible budget competitor. Resets and computation are free.

For each trial, covariance-characterization repeats, fresh clock-pair evaluation and noisy optical images are independent of alarm selection. Both policies share these random draws, enabling paired squared-error differences. The two conditional policies can select different repeat counts, but never read the unpurchased tail. `policy_covariance` expands the two pooled class covariances to the 40 event labels before the clock fit. RMSE uses the new optical delay; it is not evaluated on the old/pilot calibration data. Error bars are ±2 Monte Carlo standard errors conditional on the realized thresholds; they omit training uncertainty and are descriptive, not rigorous confidence bounds.

## Validation and reading the results

See `RESULTS.md` for the fixed-seed numerical readout. Check alarm rates first, then optical-delay RMSE, then paired MSE differences. A confidence-looking error bar crossing zero is not evidence of equal methods; it means this simulation does not establish a risk difference at that Monte Carlo resolution.

Tests cover independent pupil Fourier propagation, pixel integrals and derivatives, joint versus profiled clock fitting, analytic generalized eigenvalues, independent whitened-matrix calculation, finite-sample affine invariance, large-sample mode recovery, exact rank enumeration, singular-data rejection, budget/purchased-data isolation, forced policy branches, training/test independence and determinism. Fifteen predecessor tests are retained; eight diagnostic/integration tests are new.

## Exercises (pick one)

1. Replace both class mode pairs by equal rhos. Prove the population spectrum collapses; explain why sample eigenvalues still split.
2. Change the pilot length to 80. Adjust both thresholds and acquisition accounting before comparing power; keep the final test seed separate.
3. Rotate the latent mode directions while preserving eigenvalues. Demonstrate a process change invisible to this diagnostic.
4. Train thresholds under the stationary two-mode null. Re-run all scenarios with independent training and explain which false-alarm statement remains justified.
5. Increase read noise or reduce motion speed. Predict delay sensitivity from the derivative above, then measure whether timing-alarm policy differences still matter.

## Scope, attribution and limitations

All images and timestamps are synthetic; numerical recovery and repeatability do not establish instrument accuracy. Known optics, photometry, trajectory and class labels, Gaussian stationary modes, instantaneous exposures and free block resets are strong assumptions. Lag-one spectra can miss higher-lag changes and mode orientation; 40-sample pilots are noisy. Small fixed-dimensional diagnostics and biased scalar whitening are intentional. No universal detector, optimal policy, calibrated coverage or real-camera performance is claimed.

Original diagnostic/orchestration and MIT adaptations of `optics.py`, `clock_model.py`, `repeat_model.py`, `characterization.py`, `modes.py`, `threshold.py`, `MODEL.md` and 15 tests from the two-mode lab at **a7cac758f73051b234c56616d3d329cc889fcc37**. Prior author/license notices are retained under `licenses/predecessor/`; current upstream full license snapshots and exact activity/commit records are in `licenses/` and `sources.json`. Dependencies are NumPy (BSD), SciPy (BSD and bundled notices), Matplotlib (PSF-based license); prysm (MIT) is a Fourier-optics reference, not a runtime dependency. Upstream code beyond the attributed learning-lab helpers was not copied. API activity timestamps are recorded verbatim and are not independent evidence of a release date.
