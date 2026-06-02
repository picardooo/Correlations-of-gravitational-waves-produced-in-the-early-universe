# Correlations-of-gravitational-waves-produced-in-the-early-universe
Gravitational waves produced in the early universe and their possible detection would bring key insights into early universe dynamics such as inflation. This project uses MCMC methods to explore the integrand parameter space and the VEGAS Monte Carlo integration package to efficiently evaluate the spectrum.
# Gravitational Wave Power Spectrum

Computes the stochastic gravitational wave background (SGWB) energy density spectrum Ω(k) using a hybrid MCMC + adaptive Monte Carlo pipeline.

## Physics

The SGWB arises from second-order scalar perturbations sourced during radiation domination. Its energy density spectrum is:

```math
\Omega(k) \propto \int\!\!\int ds\,dt\;
u\,v\,
\mathrm{Pol}(u,v)^2\,
K(u,v)\,
P(u\,k \mid k_p)\,
P(v\,k \mid k_p)\,
(u\,v)^{-3}
```

where `K(u,v)` is the second-order kernel, `Pol(u,v)` is the polarisation factor, and `P(k | kp)` is the primordial power spectrum (PPS), modelled here as a log-normal peaked at `kp`.

The change of variables `(s,t) → (u,v)` maps a triangular integration domain onto a rectangle.

**Key distinction:** `kp` is a fixed physical scale (the peak of the PPS) and does not change between runs. `k` is the wavenumber at which Ω is evaluated. The full spectrum is obtained by computing one value of Ω(k) per job.

## Integration Method

Standard quadrature fails here because the integrand is highly oscillatory, exhibits a narrow ridge structure, and spans many orders of magnitude. A two-stage approach is therefore used.

### Stage 1 — MCMC Importance Map (`emcee`)

An ensemble sampler with 20 walkers explores

```math
\log\left| \mathrm{integrand}(s,t) \right|
```

to locate the high-weight regions of the domain.

After a burn-in phase, the production chain is checked for convergence via acceptance fraction and autocorrelation time `τ` before proceeding.

### Stage 2 — Adaptive Monte Carlo (`vegas`)

The MCMC samples are used to train a `vegas.AdaptiveMap`, concentrating evaluations in regions where the integrand is large.

Integration is then refined iteratively until the relative uncertainty

```math
\sigma/\mu
```

falls below 0.1%.

## Cluster Usage

Each evaluation of Ω(k) is independent. There is no communication between different values of `k`.

On a cluster, each node receives a single value of `k` as an argument and writes its result to a file. The full spectrum is assembled afterwards by collecting all outputs.

```bash
# Example SLURM array job (one k per task)
#SBATCH --array=0-199

python spectrum.py --k ${K_VALUES[$SLURM_ARRAY_TASK_ID]}
```

## Usage (Single k)

Install dependencies:

```bash
pip install numpy emcee vegas
```

Run:

```bash
python spectrum.py
```

Parameters are defined near the top of `spectrum.py`:

| Parameter | Default | Description |
|------------|---------|-------------|
| `K` | `1` | Wavenumber `k` at which Ω is evaluated |
| `KP` | `1` | Peak scale of the log-normal PPS |
| `SIGMA` | `0.5` | Log-normal width σ |
| `N_MAIN` | `10000` | MCMC production steps |
| `CONVERGENCE_TARGET` | `1e-3` | Vegas σ/μ stopping criterion |

## Sample Output

```text
=======================================================
Stage 1 — MCMC (emcee)
=======================================================
  ── Acceptance fraction ──────────────────────────────
     Mean : 0.341   (target 0.2 – 0.5)
     ✓  Acceptance fraction looks healthy.

  ── Autocorrelation time (τ) ─────────────────────────
     ✓  τ[s] = 18.3   N/τ = 546.4  (want > 50)
     ✓  τ[t] = 22.1   N/τ = 452.5  (want > 50)

  ✓  MCMC diagnostics passed. Proceeding to Vegas.

=======================================================
Stage 2 — Adaptive Monte Carlo (Vegas)
=======================================================
  Iter      σ/μ (%)           mean         sdev
  ──────────────────────────────────────────────────
     1      0.4821%   3.817432e-02   1.84e-04
     2      0.1823%   3.821107e-02   6.97e-05
     3      0.0891%   3.819654e-02   3.40e-05

  ✓  Converged at iteration 3.

=======================================================
Result
=======================================================
  Ω(k)   = 3.819654e-02
  ± sdev = 3.40e-05
  σ/μ    = 0.0891%
=======================================================
```

## Dependencies

```text
numpy
emcee
vegas
```
