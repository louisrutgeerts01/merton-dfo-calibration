
# Merton Jump–Diffusion Calibration (Derivative-Free)

This repository contains a Python implementation of the **Merton (1976) jump–diffusion model** and a calibration routine based on **Powell’s derivative-free optimization method**.

- **Repo:** <https://github.com/louisrutgeerts01/merton-dfo-calibration>  
- **README:** Quickstart, references, and API details are in the repository.  
- **Math on this page** is rendered with MathJax.

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

---

## Mathematical Model

The **Merton jump–diffusion** extends the Black–Scholes model by adding **Poisson-driven jumps** to the asset price process.  
Let \( S_t \) denote the asset price at time \( t \). The SDE is:

\[
\frac{dS_t}{S_{t^-}} = (a - \lambda k)\,dt + \sigma\,dW_t + d q_t,
\]
where:

- \( W_t \) : standard Brownian motion (diffusive part)  
- \( q_t = (Y - 1) N_t \) : compound Poisson jump process  
- \( N_t \) : Poisson process with intensity \( \lambda > 0 \)  
- \( Y \) : i.i.d. **jump multiplier** ( \( S_t \gets Y \, S_{t^-} \) when a jump occurs )  
- \( k = \mathbb{E}[Y - 1] \) : mean relative jump size  
- \( a \) : instantaneous expected return (historical drift)  
- \( \sigma \) : volatility of the continuous part  
- \( \lambda \) : average number of jumps per unit time  

The drift term includes the jump compensation \( -\lambda k \) to ensure martingale property under the risk-neutral measure.

---

## Log-Price Dynamics

Let \( X_t = \ln S_t \). By applying Itô’s lemma for jump processes:

\[
dX_t = \left( a - \lambda k - \frac{1}{2}\sigma^2 \right) dt + \sigma\,dW_t + \ln Y \, dN_t.
\]

Here \( \ln Y \) is the **log-jump size**. In Merton's specification:

\[
\ln Y \equiv J \sim \mathcal{N}(\mu_J, \sigma_J^2),
\]
so that
\[
k = \mathbb{E}[e^J - 1] = \exp\left(\mu_J + \frac{1}{2} \sigma_J^2\right) - 1.
\]

---

## Exact Discretization

We integrate \( dX_t \) over an interval \( [t, t + \Delta t] \):

- Brownian increment:
\[
\int_t^{t+\Delta t} \sigma \, dW_s = \sigma \sqrt{\Delta t} \, Z, \quad Z \sim \mathcal{N}(0,1).
\]
- Poisson jump count:
\[
K := N_{t+\Delta t} - N_t \sim \mathrm{Poisson}(\lambda \Delta t).
\]
- Sum of log-jumps (conditionally normal):
\[
\sum_{j=1}^K J_j \;\big|\;K \;\sim\; \mathcal{N}(K \mu_J,\, K \sigma_J^2).
\]

Thus, the **exact log-return** is:
\[
\ln\frac{S_{t+\Delta t}}{S_t} =
\left(a - \lambda k - \frac{1}{2}\sigma^2\right) \Delta t
+ \sigma \sqrt{\Delta t} \, Z
+ \sum_{j=1}^K J_j.
\]

---

## Simulation Formula

Price update:
\[
S_{t+\Delta t} = S_t \exp\left[
\left(a - \lambda k - \frac{1}{2}\sigma^2\right) \Delta t
+ \sigma \sqrt{\Delta t} \, Z
+ \sum_{j=1}^K J_j
\right].
\]

**Risk-neutral case:**  
Set \( a = r \) (risk-free rate) and keep the \( -\lambda k \) term to ensure discounted prices are martingales.

---

## Quickstart

```bash
# clone and enter
git clone https://github.com/louisrutgeerts01/merton-dfo-calibration.git
cd merton-dfo-calibration

# optional: virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# install
pip install -r requirements.txt
