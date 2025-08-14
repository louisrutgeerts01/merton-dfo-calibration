---
title: Merton DFO Calibration
layout: default
mathjax: true
---

# Merton Jump–Diffusion Calibration (Derivative-Free)

This repository contains a Python implementation of the **Merton (1976) jump–diffusion** model and a calibration routine based on **Powell’s derivative-free optimization**.

- **Repo:** <https://github.com/louisrutgeerts01/merton-dfo-calibration>  
- **README:** Quickstart, references, and API details are in the repository.  
- **Math on this page** is rendered with MathJax.

<!-- Load MathJax -->
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<!-- Make equations larger and add vertical spacing -->
<style>
  .mjx-chtml { font-size: 120% !important; }
  .MathJax_Display { margin: 1.2em 0 !important; }
</style>

---

## Mathematical Model

Let the asset price be

$$
S_t.
$$

The **Merton jump–diffusion** stochastic differential equation (SDE) is

$$
\frac{dS_t}{S_{t^-}} \;=\; (a - \lambda k)\,dt \;+\; \sigma\,dW_t \;+\; dq_t,
$$

where

$$
W_t \text{ is standard Brownian motion,} \qquad
dq_t \;=\; (Y-1)\,dN_t,\qquad
N_t \sim \text{Poisson process with intensity } \lambda>0,
$$

and the jump **multiplier** satisfies

$$
S_t \;=\; Y\,S_{t^-}\quad\text{at a jump time},\qquad
k \;:=\; \mathbb{E}[Y-1].
$$

---

### Merton’s specification (lognormal jumps)

Write

$$
Y \;=\; e^{J},\qquad J \sim \mathcal{N}(\mu_J,\sigma_J^2).
$$

Then the jump compensation term is

$$
k \;=\; \mathbb{E}[e^{J}-1] \;=\; \exp\!\Big(\mu_J+\tfrac12\sigma_J^2\Big)-1.
$$

---

## Log-Price Dynamics

Define the log price

$$
X_t \;:=\; \ln S_t.
$$

By Itô’s lemma with jumps,

$$
dX_t
\;=\;
\Big(a - \lambda k - \tfrac12\sigma^2\Big)\,dt
\;+\; \sigma\,dW_t
\;+\; \ln Y \, dN_t.
$$

---

## Exact Discretization over \( \Delta t \)

Integrate on the interval \( [t,\,t+\Delta t] \). The Brownian increment is

$$
\int_t^{t+\Delta t} \sigma\,dW_s \;=\; \sigma\sqrt{\Delta t}\,Z, 
\qquad Z \sim \mathcal{N}(0,1).
$$

The jump count is

$$
K \;:=\; N_{t+\Delta t}-N_t \;\sim\; \mathrm{Poisson}(\lambda\,\Delta t).
$$

Conditionally on \(K\), the sum of log-jumps is Normal:

$$
\sum_{j=1}^{K} \ln Y_j \;\Big|\;K \;\sim\; \mathcal{N}\!\big(K\mu_J,\;K\sigma_J^2\big).
$$

Hence the **exact** log-return is

$$
\ln\!\frac{S_{t+\Delta t}}{S_t}
\;=\;
\Big(a - \lambda k - \tfrac12\sigma^2\Big)\Delta t
\;+\; \sigma\sqrt{\Delta t}\,Z
\;+\; \sum_{j=1}^{K} \ln Y_j.
$$

Equivalently, the **price update** is

$$
S_{t+\Delta t}
\;=\;
S_t\,
\exp\!\Big(
\big(a - \lambda k - \tfrac12\sigma^2\big)\Delta t
\;+\; \sigma\sqrt{\Delta t}\,Z
\;+\; \sum_{j=1}^{K} \ln Y_j
\Big).
$$

> **Risk-neutral measure.** Set \(a=r\) (risk-free rate) and retain the term \( -\lambda k \) so that the discounted price is a martingale.

---

## Quickstart

```bash
git clone https://github.com/louisrutgeerts01/merton-dfo-calibration.git
cd merton-dfo-calibration

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
