import numpy as np
from typing import Callable, Optional
from scipy.optimize import minimize
from merton import simulate_merton

SimFunc = Callable[..., tuple[np.ndarray, np.ndarray]]

def obj_fun(
    params: np.ndarray,
    sim_fun: SimFunc,
    target_prices: np.ndarray,
    seed: Optional[int] = None,
    n_paths: int = 500,
    dt: float = 1/252,
    risk_neutral: bool = False
) -> float:
    """
    Objective function: aggregated SSE on log-returns.

    Parameters
    ----------
    params : np.ndarray
        Model parameters [mu, sigma, lambd, mu_j, sigma_j].
    sim_fun : Callable
        Simulation function returning (S, t).
    target_prices : np.ndarray
        Observed price series (1D).
    seed : int, optional
        RNG seed for reproducibility.
    n_paths : int
        Number of Monte Carlo paths.
    dt : float
        Time step size.
    risk_neutral : bool
        If True, use risk-neutral drift adjustment.

    Returns
    -------
    float
        Mean sum of squared errors (SSE) across paths.
    """
    mu, sigma, lambd, mu_j, sigma_j = params

    # Guard against invalid params
    if not np.isfinite(params).all() or sigma < 0 or lambd < 0 or sigma_j < 0 or dt <= 0:
        return 1e50

    n_steps = len(target_prices) - 1
    target_ret = np.diff(np.log(target_prices))

    S, t = sim_fun(
        S0=float(target_prices[0]),
        mu=mu, sigma=sigma,
        lambd=lambd, mu_j=mu_j, sigma_j=sigma_j,
        n_steps=n_steps, n_paths=n_paths, dt=dt,
        risk_neutral=risk_neutral, seed=seed, return_logret=False
    )

    sim_ret = np.diff(np.log(S), axis=1)
    residuals = sim_ret - target_ret[None, :]
    sse_per_path = np.sum(residuals**2, axis=1)
    loss = float(np.mean(sse_per_path))

    if not np.isfinite(loss):
        return 1e50
    return loss


def calibrate_with_powell(
    sim_fun: SimFunc,
    target_prices: np.ndarray,
    x0: np.ndarray,
    seed: Optional[int] = None
):
    """
    Calibrate model parameters via Powell's method.

    Parameters
    ----------
    sim_fun : Callable
        Simulation function returning (S, t).
    target_prices : np.ndarray
        Observed price series (1D).
    x0 : np.ndarray
        Initial guess for parameters [mu, sigma, lambd, mu_j, sigma_j].
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    OptimizeResult
        Result object from scipy.optimize.minimize.
    """
    bounds = [
        (-1.0, 1.0),     # mu
        (1e-8, 3.0),     # sigma
        (1e-10, 10.0),   # lambd
        (-2.0, 2.0),     # mu_j
        (1e-8, 2.0),     # sigma_j
    ]

    res = minimize(
        obj_fun,
        x0,
        args=(sim_fun, target_prices, seed),
        method="Powell",
        bounds=bounds,
        options={"maxiter": 200, "xtol": 1e-4, "ftol": 1e-4, "disp": True},
    )
    return res