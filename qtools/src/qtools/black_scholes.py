r"""
 ____  _            _        ____       _           _           
| __ )| | __ _  ___| | __   / ___|  ___| |__   ___ | | ___  ___ 
|  _ \| |/ _` |/ __| |/ /___\___ \ / __| '_ \ / _ \| |/ _ \/ __|
| |_) | | (_| | (__|   <_____|__) | (__| | | | (_) | |  __/\__ \
|____/|_|\__,_|\___|_|\_\   |____/ \___|_| |_|\___/|_|\___||___/

"""

import numpy as np
from enum import StrEnum
from datetime import date, datetime
import math
from scipy.optimize import brentq
from typing import Optional
from dataclasses import replace
import warnings

from . import numerical
from .options import OptionContract, MarketState, OptionQuote, OptionType
from .errors import VolSolverError, VolSolverWarning

   
def black_scholes_price(contract: OptionContract, market: MarketState, sigma: float)->float:
    """
    Returns the Black-Scholes price from an Option object

    Parameters
    ----------
    contract : OptionContract
        The option contract
    market : MarketState
        The market state
    sigma : float
        Pricing volatility
    
    Returns
    -------
    A new OptionQuote object with the corresponding price
    """
    price= black_scholes(contract.type, market.S, contract.K, contract.T, sigma, market.r)
    return price
    # return OptionQuote(contract, market, model_price = price)

def black_scholes(o_type: OptionType, S: float, K: float, T: float, sigma: float, r:float):
    """
    Returns the price of a european option from the given 
    parameters according to the Black-Scholes model. All inputs 
    are expected to be scalar float

    Parameters
    ----------
    type : OptionType
        The option type (EuropeanCall or EuropeanPut)
    S : float
        The spot price of the underlying 
    K : float
        The strike price
    T : float
        Time to maturity in years
    sigma : float
        The pricing volatility
    r : float
        The risk-free rate
    
    Returns
    -------
    float
        The price as a float

    Raises
    ------
    ValueError
        If S, K are negative or sigma, T are not strictly positive
    
    Notes
    -----
    By default, the function computes time to expiry in years from today's date (datetime.now())
    For reproductibility, set the argument equal to the desired date 
    """

    if (S < 0):
        raise ValueError("S must be non-negative.")
    if (K < 0):
        raise ValueError("K must be non-negative")
    if (sigma <= 0):
        raise ValueError("sigma must be strictly positive.")
    if (T <= 0):
        raise ValueError("T must be strictly positive.")

    if o_type == OptionType.EUROPEAN_CALL: 
        return _call_price_bs(S, K, T, sigma, r)
    if o_type == OptionType.EUROPEAN_PUT:
        return _put_price_bs(S, K, T, sigma, r)
    
#------------------------------------- private pricing functions ---------------------------
def _call_price_bs(S: float, K: float, T: float, sigma: float, r:float):
    d1 = (np.log(S/K) + (r +( sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    return S*numerical.norm_cdf(d1) - K*np.exp(-r*T)*numerical.norm_cdf(d2)

def _put_price_bs(S: float, K: float, T: float, sigma: float, r:float):
    d1 = (np.log(S/K) + (r +( sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    return -S*numerical.norm_cdf(-d1) + K*np.exp(-r*T)*numerical.norm_cdf(-d2)

#-----------------------------------------------------------------------------------------
def iv_solver(market_price: float, contract: OptionContract, market: MarketState, sigma_min: float =  1e-6, sigma_max: float = 5, strict = False)->float:
    """
    Computes Black-Scholes implied volatility

    Parameters
    ----------
    market_price : float 
        The market price of the option
    contract : OptionContract 
        The option contract
    market : MarketState 
        Current market state
    sigma_min : float 
        Lower bound for solver
    sigma_max : float
        Max bound for solver
    strict : bool
        If set to True, raise IVSolverError if the 
        implied volatility can't be computed. Returns 
        np.nan otherwise.
    
    Returns
    -------
    float    
        The implied volatility

    Raises
    ------
    VolSolverError
        If brentq didn't find a solution for IV
    
    VolSolverWarning
        If parameters break no-arbitrage conditions

    Notes
    -----
    IV is computed using scipy.optimize.brentq.
    If IV can't be computed because of arbitrage condition, returned value is np.nan
    """
    
    if (contract.T is None) or (contract.T<0): 
        raise ValueError("dupire_tools.black_scholes.iv_solver : contract.T is not not valid - undefined or negative. Run option.compute_T() to compute T")
    if (market_price is None):
        raise ValueError("dupire_tools.black_scholes.iv_solver : market_price can't be None")
        
    #Arbitrage condition: 
    
    discK = contract.K * np.exp(-market.r*contract.T) #### K*e^(-rT)
    
    if contract.type==OptionType.EUROPEAN_CALL: 
        lower = max((market.S - discK), 0)
        upper = market.S
    if contract.type==OptionType.EUROPEAN_PUT: 
        lower = max((-market.S + discK), 0)
        upper = discK
        

    if(market_price > upper) or (market_price<lower):
        if strict:
            raise VolSolverError(f"Arbitrage Error : market price {market_price} for upper bound {upper} and lower bound {lower}")
        else:
            warnings.warn(f"Arbitrage Error : market price {market_price} for upper bound {upper} and lower bound {lower}. Returned value is np.nan", VolSolverWarning)
            return np.nan

    def residuals(sigma):
        return (black_scholes(contract.type,market.S,contract.K,contract.T,sigma,market.r) - market_price)

    try: 
        return brentq(residuals, sigma_min, sigma_max)
    except Exception as e:
        if strict:
            raise VolSolverError(f"iv_solver: brentq failed on [{sigma_min},{sigma_max}]") from e
        else:
            warnings.warn(f"Implied vol solver failed: {e}", VolSolverWarning)
            return np.nan

def compute_iv(market_price: float, quote: OptionQuote, sigma_min: float =  1e-6, sigma_max: float = 5)->OptionQuote:
    """
    Computes the BS implied volatility and returns an OptionQuote with corresponding IV
    
    Parameters
    ----------
    market_price : float
        Current market price of the option
    quote : OptionQuote 
        The current option quotde
    sigma_min : float 
        lower bound for solver
    sigma_max : float
        max bound for solver

    Returns
    --------
    OptionQuote 
        OptionQuote with implied volatility
    
    Notes
    -----
    IV is computed using scipy.optimize.brentq
    """
    iv = iv_solver(market_price = market_price, contract = quote.contract, market  =quote.market_state)
    return replace(quote, iv = iv)
    