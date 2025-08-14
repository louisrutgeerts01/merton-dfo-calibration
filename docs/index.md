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

<style>
  .mjx-chtml { font-size: 120% !important; }
  .MathJax_Display { margin: 1.2em 0 !important; }
</style>

---

## Mathematical Model

Let  $ S_t $ be the asset price. The **Merton jump–diffusion** stochastic differential equation (SDE) is  

$$
\frac{dS_t}{S_{t^-}} = (a - \lambda k)\,dt + \sigma\,dW_t + dq_t
$$  

where  

$$
dq_t = (Y-1)\,dN_t, \quad N_t \text{ is Poisson with intensity } \lambda > 0
$$  

and the jump multiplier satisfies  

$$
S_t = Y\,S_{t^-}, \quad k = \mathbb{E}[Y-1].
$$

---

### Merton’s specification (lognormal jumps)

$$
Y = e^{J}, \quad J \sim \mathcal{N}(\mu_J, \sigma_J^2),
$$  

so that  

$$
k = \mathbb{E}[e^J - 1] = \exp\left(\mu_J + \frac12\sigma_J^2\right) - 1.
$$

---

## Log-Price Dynamics

Let  

$$
X_t = \ln S_t
$$  

then by Itô’s lemma with jumps  

$$
dX_t = \left( a - \lambda k - \frac12\sigma^2 \right) dt + \sigma\,dW_t + \ln Y \, dN_t.
$$

---

## Exact Discretization over \( \Delta t \)

Over an interval \( \Delta t \),  

$$
\int_t^{t+\Delta t} \sigma\,dW_s = \sigma\sqrt{\Delta t}\,Z, \quad Z \sim \mathcal{N}(0,1),
$$  

$$
K := N_{t+\Delta t} - N_t \sim \mathrm{Poisson}(\lambda\Delta t),
$$  

and conditionally on \(K\),  

$$
\sum_{j=1}^{K} \ln Y_j \sim \mathcal{N}(K\mu_J, K\sigma_J^2).
$$

Thus, the exact log-return is  

$$
\ln\frac{S_{t+\Delta t}}{S_t} = \left(a - \lambda k - \frac12\sigma^2\right) \Delta t + \sigma\sqrt{\Delta t}\,Z + \sum_{j=1}^{K} \ln Y_j.
$$

Equivalently, the price update is  

$$
S_{t+\Delta t} = S_t \exp\left[ \left(a - \lambda k - \frac12\sigma^2\right) \Delta t + \sigma\sqrt{\Delta t}\,Z + \sum_{j=1}^{K} \ln Y_j \right].
$$

> **Risk-neutral measure:** Set \(a = r\) and keep the \( -\lambda k \) term so discounted prices are martingales.