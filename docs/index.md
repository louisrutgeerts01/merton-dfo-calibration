---
title: Merton DFO Calibration
layout: default
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

---

## Mathematical Model

Let \(S_t\) denote the asset price. The Merton jump–diffusion SDE is
\[
\frac{dS_t}{S_{t^-}} = (a - \lambda k)\,dt + \sigma\,dW_t + dq_t,
\]
where \(W_t\) is standard Brownian motion, \(q_t = (Y-1)N_t\) is a compound Poisson process with
intensity \(\lambda>0\), and \(Y\) is the jump multiplier (\(S_t = Y\,S_{t^-}\) at a jump). The mean relative
jump size is \(k := \mathbb{E}[Y-1]\).

---

## Log-Price Dynamics

Define \(X_t := \ln S_t\). By Itô’s lemma for jump–diffusions,
\[
dX_t = \Big(a - \lambda k - \tfrac12 \sigma^2\Big)\,dt \;+\; \sigma\,dW_t \;+\; \ln(Y)\,dN_t.
\]

In Merton’s specification, the log-jump \(J := \ln Y\) is normal:
\[
J \sim \mathcal{N}(\mu_J,\sigma_J^2), \qquad
k = \mathbb{E}\big[e^{J}-1\big] = \exp\!\Big(\mu_J + \tfrac12 \sigma_J^2\Big) - 1.
\]

---

## Exact Discretization

Integrate over \([t, t+\Delta t]\). Let \(Z\sim\mathcal{N}(0,1)\), \(K := N_{t+\Delta t}-N_t \sim \mathrm{Poisson}(\lambda\Delta t)\),
and (conditionally) \(\sum_{j=1}^K J_j \sim \mathcal{N}(K\mu_J, K\sigma_J^2)\). Then the **exact log-return** is
\[
\ln\frac{S_{t+\Delta t}}{S_t}
= \Big(a - \lambda k - \tfrac12 \sigma^2\Big)\Delta t \;+\; \sigma\sqrt{\Delta t}\,Z \;+\; \sum_{j=1}^{K} J_j.
\]

Equivalently, the **price update** is
\[
S_{t+\Delta t} \;=\; S_t \exp\!\left(
\Big(a - \lambda k - \tfrac12 \sigma^2\Big)\Delta t + \sigma\sqrt{\Delta t}\,Z + \sum_{j=1}^{K} J_j
\right).
\]

> Risk-neutral measure: set \(a=r\) and retain the \(-\lambda k\) compensation so discounted prices are martingales.

---

## Quickstart

```bash
git clone https://github.com/louisrutgeerts01/merton-dfo-calibration.git
cd merton-dfo-calibration

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt