r"""
 ______     _____   _____           _     
/ ___\ \   / /_ _| |_   _|__   ___ | |___ 
\___ \\ \ / / | |    | |/ _ \ / _ \| / __|
 ___) |\ V /  | |    | | (_) | (_) | \__ \
|____/  \_/  |___|   |_|\___/ \___/|_|___/
                                          

SVI model Notes :

log_moneyness: log moneyness (log(K/F))
a: minimum total variance
b: slope. must be stricly positive
rho: skew. must be in (-1,1)
m: center, location of smile minimum
sigma: curvature; must be strictly positive

w(k) = a + b(rho(k-m) + sqrt[(k-m)^2 + sigma^2]

"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.optimize import least_squares
from .options import OptionBook
import warnings

from .errors import SVIParamError, SVIFitWarning, SVIFitError

def svi_w_scalar(lm: float, a:float, b:float, rho:float, m:float, sigma:float) -> float:
    """
    Compute total implied variance from the 
    SVI parameters as scalar

    Parameters
    ----------
    lm: float
        Log-moneyness = log(K/F)
    a: float
        Minimum total variance
    b: float
        Slope. must be stricly positive
    rho: float 
        Skew. must be in (-1,1)
    m: float
        Center, location of smile minimum
    sigma: float
        Curvature; must be strictly positive
    
    Returns
    -------
    float
        Total Implied Variance
    """
    p2 = np.sqrt((lm-m)**2 + sigma**2)
    p1 = rho*(lm-m)
    return a + b*(p1+p2)

def svi_total_variance(lm: np.ndarray, a:float, b:float, rho:float, m:float, sigma:float):
    """
    Compute total implied variance for a 
    vector of log moneyness

    Parameters
    ----------
        lm: np.ndarray
            Log-moneyness = log(K/F)
        a: float
            Minimum total variance
        b: float
            Slope. must be stricly positive
        rho: float 
            Skew. must be in (-1,1)
        m: float
            Center, location of smile minimum
        sigma: float
            Curvature; must be strictly positive
    """
    p2 = np.sqrt((lm-m)**2 + sigma**2)
    p1 = rho*(lm-m)
    return a + b*(p1+p2)

@dataclass
class SVIParam:
    """
    Simple container for SVI curve parameters
    """
    a: float
    b : float
    rho: float 
    m: float 
    sigma: float 

    def __post_init__(self):
        if self.b <= 0: 
            raise SVIParamError("SVIParam: b must be strictly positive")
        if (self.rho <= -1 ) or (self.rho >= 1):
            raise SVIParamError("SVIParam: rho must be in (-1,1)")
        if (self.sigma <= 0):
            raise SVIParamError("SVIParam: sigma must be superior to zero")

    def __iter__(self):
        yield self.a
        yield self.b
        yield self.rho
        yield self.m
        yield self.sigma

    def to_slice(self, T:float):
        """
        Returns a SVISlice from the attributes of the SVIParam
        and a maturity T

        Parameters
        ----------
        T: float
            The maturity in years of the slice

        Returns
        -------
        SVISlice
            A SVISlice with the specified maturity T
        """
        return SVISlice(self.a, self.b, self.rho, self.m, self.sigma, T)

@dataclass(frozen=True)
class SVISlice:
    """
    Set of SVI parameters that correspond to a certain maturity

    Parameters
    ----------
    log_moneyness: float
        log moneyness (log(K/F))
    a: float
        minimum total variance
    b: float
        slope. must be stricly positive
    rho: float 
        skew. must be in (-1,1)
    m: float
        center, location of smile minimum
    sigma: float
        curvature; must be strictly positive
    T: float
        Time to maturity in years
    """

    a: float
    b : float
    rho: float 
    m: float 
    sigma: float 
    T: float

    def __post_init__(self):
        if self.b <= 0: 
            raise SVIParamError("SVIParam: b must be strictly positive")
        if (self.rho <= -1 ) or (self.rho >= 1):
            raise SVIParamError("SVIParam: rho must be in (-1,1)")
        if (self.sigma <= 0):
            raise SVIParamError("SVIParam: sigma must be superior to zero")
        if (self.T <= 0):
            raise SVIParamError("SVIParam: T must be superior to zero")

    @property
    def params(self):
        return SVIParam(self.a, self.b, self.rho, self.m, self.sigma)
    

def fit_svi(initial: SVIParam, k:np.ndarray, w_target:np.ndarray, weight_i: np.ndarray = None, norm_residuals=False, lam:float = None, loss:str = None, f_scale: float = 1):
    """
    Returns the optimal SVI curve parameters for fixed 
    log moneyness and target total variance.

    Parameters
    ----------
    initial : SVIParam
        The initial values for the search
    k : np.ndarray
        A NumPy array of log-moneyness values
    w_target: np.ndarray
        The target total variance
    weight_i: np.ndarray
        The weight of the ith value of w_target
    norm_residuals: boolean
        Wether to apply normalisation to the residuals function
    lam : float
        Loss factor to be added to the normalisation. Default value
        when norm_residuals set to True is np.median(w_target)
    
    Returns
    -------
    SVIParam
        The optimal SVI curve parameters found

    Notes
    -----
    The minimisation is made with the `spicpy.least_squares` method. Default value 
    for lam if not specified when norm_residuals == True is the median of w_target
    """

    a0, b0, rho0, m0, sigma0 = initial
    x0 = np.array([a0, b0, rho0, m0, sigma0])
    if (norm_residuals == False) and lam is not None:
        warnings.warn("A value was set for lam but norm_residuals == False ; lam will be ignored", SVIFitWarning)

    if lam is None: lam = np.median(w_target)


    def residuals(params: np.ndarray, k: np.ndarray, w_target: np.ndarray):
        a, b, rho, m, sigma = params
        w_svi = svi_total_variance(k, a, b, rho, m, sigma)
        if norm_residuals:
            scale = np.maximum(w_target, 1e-6)
            r = (w_svi - w_target)/(scale+lam)

        else:
            r = w_svi-w_target 
        if weight_i is None:
            return r
        else:
            if weight_i.shape != w_target.shape:
                raise SVIFitError("weights and target shape differ")
            return np.sqrt(weight_i)*r

    if loss is None: loss = 'linear'
    res = least_squares(
        fun=residuals,
        x0=x0,            
        loss  = loss,
        f_scale = f_scale,
        args=(k, w_target),
        bounds=(
            [0.0, 1e-8, -0.999, -np.inf, 1e-8],
            [np.inf, np.inf,  0.999,  np.inf, np.inf],

        )
    )
    return SVIParam(*res.x)


def svi_error(book: OptionBook, fit: SVIParam, metric:str = "mse"):
    """
    Returns the error between market points and the SVI curve.

    Parameters
    ----------
    book: OptionBook
        A set of market quotes
    fit: SVIParam
        The parameters of the SVI fit
    metric: str
        The metric to use, among "mse"

    Returns
    -------
    float: the error metric
    """

    if metric == 'mse':
        lm = book.LM
        svi_tvar = svi_total_variance(lm, *fit)
        market_tvar = book.W
        return np.mean((svi_tvar-market_tvar)**2)