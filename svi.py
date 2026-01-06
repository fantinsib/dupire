##### SVI model 
# 
# k = log(K/Ft) -> log moneyness
# a -> verical shift
# b > 0 -> smile amplitude 
# p in (-1,1) -> skew 
# m -> horizontal translation
# sigma > 0 -> lissage
#

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import least_squares

@dataclass(frozen=True)
class SVIParams:
    """
    Args:
        log_moneyness: log moneyness (log(K/F))
        a: minimum total variance
        b: slope. must be stricly positive
        rho: skew. must be in (-1,1)
        m: center, location of smile minimum
        sigma: curvature; must be strictly positive
    """
    a: float
    b : float
    rho: float 
    m: float 
    sigma: float 

class SVI:
    
    def compute(self, log_moneyness: np.array, params: SVIParams):
        """
        Computes SVI 

        Args:
        params: SVIParams object

        Returns:
            w(k): total implied variance

        """
        p2 = np.sqrt((log_moneyness-params.m)**2 + params.sigma**2)
        p1 = params.rho*(log_moneyness-params.m)
        return params.a + params.b*(p1+p2)


    def fit(self, log_moneyness, w, weights=None, init=None, max_nfev=10000):
        """
        Returns the parameters of SVI with the best fit from the data

        Args: 


        """
        log_moneyness = np.asarray(log_moneyness, dtype=float).ravel()
        w = np.asarray(w, dtype=float).ravel()

        if log_moneyness.shape != w.shape:
            raise ValueError(f"log_moneyness and w must have same shape, got {log_moneyness.shape} vs {w.shape}")
        if log_moneyness.size < 5:
            raise ValueError("Need at least 5 points to fit 5 SVI parameters robustly.")
        if np.any(~np.isfinite(log_moneyness)) or np.any(~np.isfinite(w)):
            raise ValueError("log_moneyness and w must be finite.")
        if np.any(w < 0):
            raise ValueError("Total variance w must be non-negative.")

        if weights is not None:
            weights = np.asarray(weights, dtype=float).ravel()
            if weights.shape != log_moneyness.shape:
                raise ValueError("weights must have same shape as log_moneyness and w.")
            if np.any(weights < 0) or np.any(~np.isfinite(weights)):
                raise ValueError("weights must be finite and non-negative.")
            sqrt_wt = np.sqrt(weights)
        else:
            sqrt_wt = None


        user_init = init is not None

        if init is None:
            i_min = int(np.argmin(w))
            m0 = float(log_moneyness[i_min])
            log_moneyness_range = float(np.max(log_moneyness) - np.min(log_moneyness))
            sigma0 = max(1e-3, 0.25 * log_moneyness_range)  

            w_spread = float(np.percentile(w, 90) - np.percentile(w, 10))
            b0 = max(1e-6, w_spread / (log_moneyness_range + 1e-6))
            rho0 = -0.2
            a0 = max(0.0, float(w[i_min]) - b0 * sigma0)

            init = SVIParams(a=a0, b=b0, rho=rho0, m=m0, sigma=sigma0)

        if user_init:
            inits = [init]
        else:
            inits = []
            rho_guesses = [-0.7, -0.2, 0.2]
            sigma_guesses = [init.sigma * s for s in (0.5, 1.0, 2.0)]
            b_guesses = [init.b]
            for rho0 in rho_guesses:
                for sigma0 in sigma_guesses:
                    for b0 in b_guesses:
                        a0 = max(0.0, float(np.min(w)) - b0 * sigma0)
                        inits.append(SVIParams(a=a0, b=b0, rho=rho0, m=init.m, sigma=max(1e-8, sigma0)))

        a_low = -1e-6
        a_high = np.inf
        b_low, b_high = 1e-12, np.inf
        rho_low, rho_high = -0.999, 0.999
        m_low, m_high = float(np.min(log_moneyness) - 2.0), float(np.max(log_moneyness) + 2.0)
        sig_low, sig_high = 1e-8, np.inf

        lb = np.array([a_low, b_low, rho_low, m_low, sig_low], dtype=float)
        ub = np.array([a_high, b_high, rho_high, m_high, sig_high], dtype=float)

        # ------ residuals ----------
        def residuals(x: np.ndarray) -> np.ndarray:
            a, b, rho, m, sigma = x
            params = SVIParams(a,b, rho, m, sigma)
            w_model = self.compute(log_moneyness, params)
            r = (w_model - w)
            if sqrt_wt is not None:
                r = r * sqrt_wt
            return r

        best_res = None
        for cand in inits:
            x0 = np.array([cand.a, cand.b, cand.rho, cand.m, cand.sigma], dtype=float)
            res = least_squares(
                residuals,
                x0=x0,
                bounds=(lb, ub),
                method="trf",
                loss="linear",
                f_scale=1.0,
                max_nfev=max_nfev,
                    x_scale="jac",
            )
            if best_res is None or res.cost < best_res.cost:
                best_res = res
            if res.success:
                best_res = res
                break

        if best_res is None or not best_res.success:
            msg = best_res.message if best_res is not None else "no result"
            raise RuntimeError(f"SVI fit failed after {len(inits)} inits: {msg}")

        a, b, rho, m, sigma = best_res.x
        return SVIParams(float(a), float(b), float(rho), float(m), float(sigma))
