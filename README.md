# Merton Jump–Diffusion Model Calibration with Derivative-Free Optimization

This project implements and calibrates the **Merton (1976) Jump–Diffusion Model** for asset prices using **Powell’s derivative-free optimization methods**.

## Overview

The Merton model extends the Black–Scholes framework by incorporating **random jumps** in the asset price process, modeled via a **compound Poisson process** with normally distributed jump sizes. This makes it better suited for capturing sudden market moves and fat-tailed return distributions.

In this project:
- We **simulate** Merton jump–diffusion paths.
- We **calibrate** model parameters to synthetic or real data using Powell’s method (from `scipy.optimize`).
- The implementation is **pure Python**, using NumPy, SciPy, and Seaborn/Matplotlib for visualization.

## Features
- Exact discretization of the jump–diffusion process (no Euler error).
- Support for **multiple simulated paths**.
- Calibration of parameters:
  - Drift (\( \mu \))
  - Volatility (\( \sigma \))
  - Jump intensity (\( \lambda \))
  - Jump mean (\( \mu_J \))
  - Jump volatility (\( \sigma_J \))
- Visual plots of simulated price paths and Poisson jump counts.

## Installation
git clone https://github.com/louisrutgeerts01/merton-dfo-calibration.git
cd merton-dfo-calibration

# Install dependencies
pip install -r requirements.txt
