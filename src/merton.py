import numpy as np
from typing import Optional

def simulate_merton(
    S0: float,
    mu: float,
    sigma: float,
    lambd: float,
    mu_j: float,
    sigma_j: float,
    n_steps: int,
    n_paths: int,
    dt: float,
    risk_neutral: bool = True,
    seed: Optional[int] = None,
    return_logret: bool = False
):
    """
    Simulate asset price paths under the Merton Jump–Diffusion model.

    Parameters
    ----------
    S0 : float
        Initial price.
    mu : float
        Annual drift (physical or risk-neutral depending on flag).
    sigma : float
        Annual volatility.
    lambd : float
        Jump intensity (jumps per year).
    mu_j : float
        Mean of jump size in log-space.
    sigma_j : float
        Std dev of jump size in log-space.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of simulated paths.
    dt : float
        Time step size in years.
    risk_neutral : bool, optional
        If True, adjusts drift to make discounted asset a martingale.
    seed : int, optional
        Random seed for reproducibility.
    return_logret : bool, optional
        If True, also returns simulated log-returns.

    Returns
    -------
    S : ndarray
        Simulated prices, shape (n_paths, n_steps+1).
    t : ndarray
        Time grid, shape (n_steps+1,).
    log_ret : ndarray, optional
        Log-returns, shape (n_paths, n_steps) if return_logret=True.
    """
    rng = np.random.default_rng(seed)
    kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    drift = (mu - lambd * kappa) if risk_neutral else mu

    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    log_ret = np.zeros((n_paths, n_steps))

    for i in range(n_steps):
        Z = rng.standard_normal(n_paths)

        # Poisson jumps count
        K = rng.poisson(lambd * dt, size=n_paths)

        # Jump size term
        jump_log = K * mu_j + rng.standard_normal(n_paths) * np.sqrt(K) * sigma_j

        # Log-return for this step
        log_ret[:, i] = (drift - 0.5 * sigma**2) * dt \
                        + sigma * np.sqrt(dt) * Z \
                        + jump_log

        # Price update
        S[:, i+1] = S[:, i] * np.exp(log_ret[:, i])

    t = np.linspace(0.0, n_steps * dt, n_steps + 1)

    if return_logret:
        return S, t, log_ret
    else:
        return S, t