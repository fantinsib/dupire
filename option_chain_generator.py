################################ Generation of option chain data 
'''
This function aims to generate synthetic option data based on the SVI model :
   
       w(k) = a + b * (rho*(k-m) + sqrt{(k-m)^2 + sigma**2)}
   
    with : 
        w(k) : the total variance 
        a : baseline level of total variance
        b: slope of the curve (b>0)
        rho: the skew 
        m: horizontal shift
        sigma: curvature, with sigma > 0

The objective is to model, for various maturities, the following properties :
- lower total variance for options with high log-moneyness vs negative log-moneyness
- as maturity increases, total variance should increase 

To do this, this function generates for each maturity T time-dependant 
parameters for an SVI curve plus some slight noise. We start by setting a
target ATM total variance, we then make realistic values for b, rho, m, sigma 
Finally we solve for a 
 '''

import numpy as np
import random
import pandas as pd


def make_synth_iv_chain_svi(
    S0=100.0, r=0.02,q=0.00,
    maturities_days=(30, 60, 90, 180, 360),
    k_min=-0.30, k_max=0.30, n_k=31,
    seed=0)-> pd.DataFrame:
    """
    Generate syntethic option chain data

    Args:
        S0: Spot price 
        r: risk-free rate
        q: dividiend yield
        maturities_days: tuple of maturities (in days)
        k_min: min log-moneyness
        k_max: max log-moneyness
        n_k: number of option per maturity
    Returns:
        pd.DataFrame of the data
    """
    rng = np.random.default_rng(seed)

    k_grid = np.linspace(k_min, k_max, n_k)
    rows = []

    for days in maturities_days:
        T = days / 365.0
        F = S0 * np.exp((r - q) * T)

        #We set the ATM forward level (k=0):
        #22% for long-term level, +6% for short term, 0.25 for decrease over time
        iv_atm = 0.22 + 0.06 * np.exp(-T / 0.25)  
        w_atm = iv_atm**2 * T ## target ATM forward total variance 

        
        bT   = 0.04 + 0.02 * np.exp(-T / 0.3) #smile amplitude
        rhoT = -0.55 + 0.20 * (T / (T + 0.5)) ## skew
        rhoT = np.clip(rhoT, -0.999, 0.999)
        sigT = 0.15 + 0.25 * np.sqrt(T) ##curvature
        mT   = -0.02 * np.exp(-2*T) ##horizontal shift

        aT = w_atm - bT * (rhoT*(-mT) + np.sqrt(mT*mT + sigT*sigT)) #ATM : k = 0
        aT = max(aT, 1e-6)

        for k in k_grid: #for each log moneyness value in the range k_min, k_max:
            w = aT + bT*(rhoT*(k-mT) + np.sqrt((k-mT)**2 + sigT**2)) # total variance
            iv = np.sqrt(w / T) #implied vol 

            iv += rng.normal(0.0, 0.002 + 0.004*abs(k))
            iv = max(iv, 1e-4)  ##noise + clipping

            K = F*np.exp(k) ## strike price from log-moneyness def (K = Fe^k))

            #simple constant IV spread around mid (for demo purposes) :
            rows.append((T, days, K, "C", iv, iv-0.003, iv+0.003, F, S0, r, q))
            rows.append((T, days, K, "P", iv, iv-0.003, iv+0.003, F, S0, r, q))

    return pd.DataFrame(
        rows,
        columns=["T","T_days","K","OptionType","iv_mid","iv_bid","iv_ask","F","S0","r","q"]
    )


make_synth_iv_chain_svi()