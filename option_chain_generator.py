# Generation of option chain data 


import numpy as np
import random
import pandas as pd


def make_synth_iv_chain(
    S0=100.0,
    r=0.02,
    q=0.00,
    maturities_days=(7, 14, 30, 60, 90, 180, 360),
    k_min=-0.30, k_max=0.30, n_k=31,
    sigma0=0.20, a=0.02, s=-0.35, c=0.40, eps=1e-3,
    noise_atm=0.003, noise_wings=0.010,
    seed=0):
    """
    Generates synthetic option data

    Args:
        S0: spot price
        r: rf rate
        q: dividend yield

    """
    rng = np.random.default_rng(seed)

    k_grid = np.linspace(k_min, k_max, n_k)
    rows = []

    for days in maturities_days:
        t = days / 365.0
        F = S0 * np.exp((r - q) * t)

        for k in k_grid:
            K = F * np.exp(k)

            iv = (
                sigma0
                + a * np.sqrt(t)
                + s * (k / np.sqrt(t + eps))
                + c * (k**2)
            )
            iv = float(np.clip(iv, 0.03, 2.00))

            w = abs(k) / max(abs(k_min), abs(k_max))
            noise_std = noise_atm * (1 - w) + noise_wings * w
            iv_mid = iv + rng.normal(0.0, noise_std)

            iv_spread = 0.002 + 0.006 * w
            iv_bid = max(0.01, iv_mid - iv_spread / 2)
            iv_ask = iv_mid + iv_spread / 2

            rows.append((t, days, float(K), "C", iv_mid, iv_bid, iv_ask, F))
            rows.append((t, days, float(K), "P", iv_mid, iv_bid, iv_ask, F))
            

    df = pd.DataFrame(
        rows,
        columns=["T", "T_days", "K", "OptionType", "iv_mid", "iv_bid", "iv_ask", "F"]
    )
    df["S0"] = S0
    df["r"] = r
    df["q"] = q
    return df


import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

def local_vol_equity(t, S, S0, sigma0=0.18, alpha=0.25, t0=0.20, beta=-0.35):
    # smooth, equity-like skew + short-term elevation
    x = np.log(max(S, 1e-12) / S0)
    sig = sigma0 * (1.0 + alpha * np.exp(-t / t0) + beta * np.tanh(x))
    return float(np.clip(sig, 0.03, 2.0))

def bs_call_forward(F, K, T, r, vol):
    if T <= 0:
        return max(0.0, np.exp(-r*T)*(F-K))
    vol = max(vol, 1e-12)
    df = np.exp(-r*T)
    srt = vol*np.sqrt(T)
    d1 = (np.log(F/K) + 0.5*vol*vol*T)/srt
    d2 = d1 - srt
    return float(df*(F*norm.cdf(d1) - K*norm.cdf(d2)))

def implied_vol_call(C, F, K, T, r):
    df = np.exp(-r*T)
    intrinsic = df*max(F-K, 0.0)
    upper = df*F
    C = float(np.clip(C, intrinsic, upper))

    def f(v): return bs_call_forward(F, K, T, r, v) - C
    return float(brentq(f, 1e-6, 3.0, maxiter=200))

def make_chain_from_local_vol(
    S0=100.0, r=0.02, q=0.00,
    maturities_days=(14, 30, 60, 90, 180, 360),
    log_moneyness_min=-0.20, log_moneyness_max=0.20, n_k=21,
    n_paths=30000, steps_per_year=120, seed=0,
    noise_iv_atm=0.0008, noise_iv_wings=0.0025
):
    rng = np.random.default_rng(seed)
    k_grid = np.linspace(log_moneyness_min, log_moneyness_max, n_k)
    rows = []

    for days in maturities_days:
        T = days/365.0
        F = S0*np.exp((r-q)*T)
        K_grid = F*np.exp(k_grid)

        n_steps = max(2, int(np.ceil(T*steps_per_year)))
        dt = T/n_steps

        # simulate S_T under local vol via log-Euler
        S = np.full(n_paths, S0, dtype=float)
        t = 0.0
        for _ in range(n_steps):
            t += dt
            # local vol per path (simple loop; ok for 30k paths)
            sig = np.array([local_vol_equity(t, s, S0) for s in S], dtype=float)
            Z = rng.standard_normal(n_paths)
            S *= np.exp((r-q-0.5*sig*sig)*dt + sig*np.sqrt(dt)*Z)

        df_disc = np.exp(-r*T)

        # price calls
        payoffs = np.maximum(S[:, None] - K_grid[None, :], 0.0)
        C = df_disc * payoffs.mean(axis=0)

        # implied vols from calls
        iv = np.array([implied_vol_call(c, F, k, T, r) for c, k in zip(C, K_grid)], dtype=float)

        # add tiny IV noise (optional), wings noisier
        w = np.abs(k_grid)/max(abs(log_moneyness_min), abs(log_moneyness_max))
        noise = (1-w)*noise_iv_atm + w*noise_iv_wings
        iv_mid = np.clip(iv + rng.normal(0.0, noise), 0.03, 2.0)

        # synthetic bid/ask spread in IV
        iv_spread = 0.002 + 0.006*w
        iv_bid = np.maximum(0.01, iv_mid - 0.5*iv_spread)
        iv_ask = iv_mid + 0.5*iv_spread

        for K, m, b, a in zip(K_grid, iv_mid, iv_bid, iv_ask):
            rows.append((T, days, float(K), "C", float(m), float(b), float(a), float(F), S0, r, q))
            rows.append((T, days, float(K), "P", float(m), float(b), float(a), float(F), S0, r, q))

    return pd.DataFrame(rows, columns=["T","T_days","K","OptionType","iv_mid","iv_bid","iv_ask","F","S0","r","q"])


def true_local_vol(t, lm):
    """
    Simple, smooth, equity-like local volatility
    t : maturity (years)
    lm: log-moneyness
    """
    sigma0 = 0.20
    term = 0.05 * np.exp(-t / 0.5)          # short-term elevation
    skew = -0.15 * lm * np.exp(-t / 1.0)    # downside skew that fades
    curvature = 0.10 * lm**2

    return sigma0 + term + skew + curvature
