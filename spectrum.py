"""
Gravitational Wave Power Spectrum
==================================
Computes the stochastic gravitational wave background (SGWB) power spectrum
using a two-stage hybrid integration pipeline:

  Stage 1 — MCMC (emcee):  Maps high-weight regions of the integrand to build
                            an importance-sampling proposal distribution.
  Stage 2 — Vegas:          Trains an AdaptiveMap on the MCMC samples, then
                            integrates until the relative uncertainty σ/μ < 0.1%.

Physical background
-------------------
The kernel K(u,v) encodes second-order scalar perturbation effects. The
primordial power spectrum (PPS) is modelled as log-normal. The change of
variables (s,t) → (u,v) maps the triangular integration domain to a rectangle.
"""

import sys
import numpy as np
import emcee
import vegas
from typing import Union

# ============================================================
# Parameters
# ============================================================

K     = 1        # Wavenumber k
KP    = 1        # Peak scale of Log-normal power spectrum
SIGMA = 0.5      # Log-normal width of the primordial power spectrum
NDIM  = 2        # Dimensionality of the MCMC sampler
NWALKERS = 20    # Number of emcee ensemble walkers (~2*NDIM)

# Emcee run lengths
N_BURNIN = 100   # Burn-in steps (discarded)
N_MAIN   = 10000 # Production steps

# Emcee diagnostic thresholds (used to warn before proceeding to Vegas)
MIN_ACCEPTANCE = 0.2   # Walkers below this fraction are poorly mixing
MAX_ACCEPTANCE = 0.5   # Walkers above this may be taking too-small steps
MAX_TAU_RATIO  = 50    # N_MAIN should be >> autocorrelation time tau

# Vegas convergence criterion
CONVERGENCE_TARGET = 1e-3   # σ/μ threshold
MAX_ITERATIONS     = 15     # Safety cap on Vegas refinement loops


# ============================================================
# Primordial power spectrum  (log-normal model)
# ============================================================

def _lognormal(x: Union[float, np.ndarray], sig: float) -> Union[float, np.ndarray]:
    """Evaluate a zero-mean log-normal distribution at x with width sig."""
    return (1.0 / (sig * np.sqrt(2 * np.pi))) * np.exp(-(x**2) / (2 * sig**2))


def pps(k: Union[float, np.ndarray], kp: float, sig: float) -> Union[float, np.ndarray]:
    """
    Primordial power spectrum P(k | kp, σ).

    Parameters
    ----------
    k  : Wavenumber (or array of wavenumbers).
    kp : Reference wavenumber k'.
    sig: Log-normal width σ.

    Returns
    -------
    P evaluated at k, peaked around k = kp.
    """
    return _lognormal(np.log(k / kp), sig)


# ============================================================
# Change of variables  (s, t) → (u, v)
# ============================================================

def u(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """u = (t + s + 1) / 2"""
    return 0.5 * (t + s + 1.0)


def v(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """v = (t - s + 1) / 2"""
    return 0.5 * (t - s + 1.0)


# ============================================================
# Kernel components
# ============================================================

def kernel_ia(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Kernel sub-component I_A(u,v)."""
    return 3.0 * (u**2 + v**2 - 3.0) / (4.0 * u**3 * v**3)


def kernel_ib(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Kernel sub-component I_B(u,v) — contains a logarithmic term."""
    num = 3.0 - (u + v)**2
    den = 3.0 - (u - v)**2
    return -4.0 * u * v + (u**2 + v**2 - 3.0) * np.log(np.abs(num / den))


def kernel_ic(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Kernel sub-component I_C(u,v) — non-zero only above the resonance threshold."""
    return np.where(u + v > np.sqrt(3.0), u**2 + v**2 - 3.0, 0.0)


def polarisation(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Polarisation factor Pol(u,v)."""
    return (4.0 * v**2 - (1.0 + v**2 - u**2)**2) / 4.0


def kernel(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Full SGWB kernel K(u,v).

    K = (1/2) * I_A² * (I_B² + π² I_C²)
    """
    ia = kernel_ia(u, v)
    ib = kernel_ib(u, v)
    ic = kernel_ic(u, v)
    return 0.5 * ia**2 * (ib**2 + np.pi**2 * ic**2)


# ============================================================
# Full integrand in (s, t) coordinates
# ============================================================

def integrand(s: np.ndarray, t: np.ndarray,
              k: float = K, kp: float = KP, sig: float = SIGMA) -> np.ndarray:
    """
    Full SGWB integrand in the (s, t) domain.

    Combines the Jacobian from the (s,t)→(u,v) change of variables,
    the polarisation factor, the kernel, and both PPS factors.
    """
    u1 = u(s, t)
    v1 = v(s, t)
    return (
        4.0 * u1 * v1
        * polarisation(u1, v1)**2
        * kernel(u1, v1)
        * pps(u1 * k, kp, sig)
        * pps(v1 * k, kp, sig)
        * (u1 * v1)**(-3)
    )


# ============================================================
# Stage 1 — MCMC with emcee
# ============================================================

def log_prob(x: np.ndarray, k: float, kp: float, sig: float) -> float:
    """
    Log-probability for the emcee sampler.

    The sampler explores the integrand magnitude (not its signed value),
    which is appropriate here because the integrand can change sign —
    we need it to map the high-weight regions regardless of sign,
    so that Vegas can later integrate the full signed function.

    Parameters
    ----------
    x   : [s, t] position of the walker.
    k, kp, sig : Physical parameters passed through from the sampler.

    Returns
    -------
    log|integrand(s, t)| if within the domain, else -inf.
    """
    s1, t1 = x

    # Hard domain boundaries
    if not (-1.0 < s1 < 1.0):
        return -np.inf
    if not (0.0 < t1 < 10.0):
        return -np.inf

    val = integrand(s1, t1, k, kp, sig)
    mag = np.abs(val)

    if mag < 1e-200:
        return -np.inf

    return np.log(mag)


def run_mcmc(k: float, kp: float, sig: float) -> np.ndarray:
    """
    Run the emcee ensemble sampler and return the flat production chain.

    Prints diagnostic statistics so the user can verify convergence before
    proceeding to Vegas.

    Returns
    -------
    flat_samples : ndarray of shape (N_MAIN * NWALKERS, 2)
    """
    print("=" * 55)
    print("Stage 1 — MCMC (emcee)")
    print("=" * 55)
    print(f"  Walkers : {NWALKERS}")
    print(f"  Burn-in : {N_BURNIN} steps")
    print(f"  Main run: {N_MAIN} steps\n")

    # Initialise walkers uniformly across the domain
    p0 = np.column_stack([
        np.random.uniform(-1.0, 1.0, size=NWALKERS),
        np.random.uniform(0.0, 10.0, size=NWALKERS),
    ])

    sampler = emcee.EnsembleSampler(
        NWALKERS, NDIM, log_prob, args=[k, kp, sig]
    )

    # Burn-in
    print("  Running burn-in...")
    state = sampler.run_mcmc(p0, N_BURNIN, progress=False)
    sampler.reset()

    # Production run
    print("  Running production chain...")
    sampler.run_mcmc(state, N_MAIN, progress=False)

    # ----------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------
    acceptance = sampler.acceptance_fraction          # shape (NWALKERS,)
    mean_acc   = np.mean(acceptance)
    min_acc    = np.min(acceptance)
    max_acc    = np.max(acceptance)

    print("\n  ── Acceptance fraction ──────────────────────────")
    print(f"     Mean : {mean_acc:.3f}   (target 0.2 – 0.5)")
    print(f"     Min  : {min_acc:.3f}")
    print(f"     Max  : {max_acc:.3f}")

    if mean_acc < MIN_ACCEPTANCE:
        print("  ⚠  Mean acceptance is LOW — walkers may be stuck.")
        print("     Consider a smaller step size or more burn-in steps.")
    elif mean_acc > MAX_ACCEPTANCE:
        print("  ⚠  Mean acceptance is HIGH — walkers may be taking tiny steps.")
    else:
        print("  ✓  Acceptance fraction looks healthy.")

    # Autocorrelation time
    try:
        tau = sampler.get_autocorr_time(quiet=True)   # shape (NDIM,)
        print("\n  ── Autocorrelation time (τ) ─────────────────────")
        for i, (name, t) in enumerate(zip(["s", "t"], tau)):
            ratio = N_MAIN / t
            flag  = "✓" if ratio > MAX_TAU_RATIO else "⚠"
            print(f"     {flag}  τ[{name}] = {t:6.1f}   "
                  f"N/τ = {ratio:5.1f}  (want > {MAX_TAU_RATIO})")
        if np.any(N_MAIN / tau < MAX_TAU_RATIO):
            print("     Chain may be too short relative to τ.")
            print("     Consider increasing N_MAIN.")
        else:
            print("     Chain length is sufficient relative to τ.")
    except emcee.autocorr.AutocorrError:
        print("\n  ⚠  Could not estimate autocorrelation time reliably.")
        print("     Chain is likely too short — consider more steps.")

    print()

    # Check whether to proceed
    _check_and_proceed(mean_acc)

    return sampler.get_chain(flat=True)


def _check_and_proceed(mean_acceptance: float) -> None:
    """
    Warn the user if diagnostics are poor and ask whether to continue.
    Exits cleanly if they decline.
    """
    poor = mean_acceptance < MIN_ACCEPTANCE or mean_acceptance > MAX_ACCEPTANCE
    if poor:
        try:
            ans = input("  Proceed to Vegas anyway? [y/N]: ").strip().lower()
        except EOFError:
            ans = "n"
        if ans != "y":
            print("  Exiting. Adjust MCMC parameters and re-run.")
            sys.exit(0)
    else:
        print("  ✓  MCMC diagnostics passed. Proceeding to Vegas.\n")


# ============================================================
# Stage 2 — Vegas adaptive Monte Carlo integration
# ============================================================

@vegas.batchintegrand
def _vegas_integrand(x: np.ndarray) -> np.ndarray:
    """
    Vegas batch integrand.

    x has shape (n_points, 2), with columns [s, t].
    """
    s1, t1 = x[:, 0], x[:, 1]
    return integrand(s1, t1, K, KP, SIGMA)


def run_vegas(flat_samples: np.ndarray) -> vegas.RAvg:
    """
    Train a Vegas AdaptiveMap on the MCMC samples and integrate to convergence.

    Parameters
    ----------
    flat_samples : Output of run_mcmc(), shape (N, 2).

    Returns
    -------
    result : Final Vegas result object (access .mean and .sdev).
    """
    print("=" * 55)
    print("Stage 2 — Adaptive Monte Carlo (Vegas)")
    print("=" * 55)

    # Use the last 2000 MCMC samples to train the importance map
    training_samples = flat_samples[-2000:]
    vegas_map = vegas.AdaptiveMap([[-1.0, 1.0], [0.0, 10.0]])
    vegas_map.adapt_to_samples(
        training_samples,
        _vegas_integrand(training_samples),
        nitn=5,
    )

    integ = vegas.Integrator(vegas_map, alpha=0.2)

    # Warm-up
    print("  Warming up...")
    integ(_vegas_integrand, neval=1e4, nitn=20)

    # Adaptive refinement loop
    print(f"\n  Refining until σ/μ < {CONVERGENCE_TARGET:.0e} "
          f"(max {MAX_ITERATIONS} iterations)\n")
    print(f"  {'Iter':>4}  {'σ/μ (%)':>10}  {'mean':>14}  {'sdev':>12}")
    print("  " + "-" * 46)

    result      = None
    sdv_ratio   = 1.0
    nitn_i      = 25

    for iteration in range(1, MAX_ITERATIONS + 1):
        result    = integ(_vegas_integrand, neval=1e6, nitn=nitn_i)
        sdv_ratio = abs(result.sdev / result.mean)

        print(f"  {iteration:>4}  {sdv_ratio*100:>9.4f}%  "
              f"{result.mean:>14.6e}  {result.sdev:>12.2e}")

        if sdv_ratio < CONVERGENCE_TARGET:
            print(f"\n  ✓  Converged at iteration {iteration}.")
            break
        nitn_i += 5
    else:
        print(f"\n  ⚠  Did not converge within {MAX_ITERATIONS} iterations.")

    return result


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Run the full MCMC → Vegas pipeline."""
    np.random.seed(42)   # Reproducibility

    flat_samples = run_mcmc(K, KP, SIGMA)
    result       = run_vegas(flat_samples)

    print("\n" + "=" * 55)
    print("Result")
    print("=" * 55)
    print(f"  Ω(k)  = {result.mean:.6e}")
    print(f"  ± sdev  {result.sdev:.2e}")
    print(f"  σ/μ   = {abs(result.sdev/result.mean)*100:.4f}%")
    print("=" * 55)


if __name__ == "__main__":
    main()
