r"""
  ___        _   _                 
 / _ \ _ __ | |_(_) ___  _ __  ___ 
| | | | '_ \| __| |/ _ \| '_ \/ __|
| |_| | |_) | |_| | (_) | | | \__ \
 \___/| .__/ \__|_|\___/|_| |_|___/
      |_| 
"""

from enum import StrEnum
from dataclasses import dataclass, replace
import pandas as pd
import yfinance as yf 
import numpy as np
from typing import Optional
from datetime import datetime
import warnings
from .errors import FetchedDataError, FetchedDataWarning


#------------------------------------------------------------#
#                                                            #
#                     CLASSES                                #---------------------------------------------------------------------------------------
#                                                            #
#------------------------------------------------------------#



class OptionType(StrEnum):
    EUROPEAN_CALL = "EuropeanCall"
    EUROPEAN_PUT = "EuropeanPut"


@dataclass(frozen=True, slots=True)
class OptionContract:
    """
    Representation of an option contract.

    Parameters
    ----------
    type : OptionType
        Type of the option (e.g. call or put).
    K : float
        Strike price of the option.
    maturity : str
        Expiration date of the option in ``YYYY-MM-DD`` format.
    T : float, optional
        Time to maturity expressed in years.
    ticker : str, optional
        Ticker symbol of the underlying asset.

    Notes
    -----
    OptionContract cannot be modified after instanciation
    """
    type:OptionType
    K:float
    maturity:str
    T: Optional[float] = None
    ticker: Optional[str] = None

    def __post_init__(self):
        if self.type is None:
            raise ValueError("Type can't be None")
        if self.K is None:
            raise ValueError("K can't be None")
        if self.maturity is None:
            raise ValueError("Maturity can't be None")

@dataclass(frozen=True, slots=True)
class MarketState:
    """
    Informations about current state of the market related to the option.    

    Parameters
    -----------
    S : float
        Current spot price
    r : float
        Risk-free rate

    Notes
    -----
    MarketState cannot be modified after instanciation
    """
    S:float
    r:float

    def __post_init__(self):
        if self.S is None:
            raise ValueError("S can't be None")
        if self.r is None:
            raise ValueError("r can't be None")


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """
    Snapshot of quote of the option
    
    Parameters
    -----------
    contract : OptionContract
        The option contract
    market_state : MarketState
        Current state of the market related to the option
    iv : float, optional
        The implied volatility
    market_price : float, optional 
        Current price of the option on the market
    model_price : float, optional
        Price according to model calculations

    Notes
    -----
    OptionQuote cannot be modified after instanciation
    """
    contract : OptionContract
    market_state: MarketState
    iv: Optional[float] = None
    market_price: Optional[float] = None
    model_price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None

    def __post_init__(self):
        if self.contract is None:
            raise ValueError("OptionContract can't be None")
        if self.market_state is None:
            raise ValueError("MarketState can't be None")

    @property
    def K(self) -> float:
        return self.contract.K
    @property
    def S(self) -> float:
        return self.market_state.S
    @property
    def type(self) -> OptionType:
        return self.contract.type
    @property
    def T(self) -> float:
        return self.contract.T
    @property
    def r(self) -> float:
        return self.market_state.r
    @property
    def maturity(self) -> str:
        return self.contract.maturity
    
    def __repr__(self):
        iv_str = f"{self.iv:.4f}" if self.iv is not None else "None"

        return (
            f"OptionQuote("
            f"type={self.contract.type}, "
            f"K={self.contract.K:.2f}, "
            f"T={self.contract.T:.4f}, "
            f"S={self.market_state.S:.2f}, "
            f"r={self.market_state.r:.4f}, "
            f"price={self.market_price:.4f}, "
            f"iv={iv_str}"
            f")"
        )


@dataclass
class OptionBook:
    """
    Creates a collection of OptionQuotes

    Parameters
    ----------
    quote_list :  list[OptionQuote]
        List of OptionQuote to be included in the book
    """
    quote_list: list[OptionQuote]
    
    def __post_init__(self):
        if self.quote_list is None:
            raise ValueError("List of OptionQuote can't be None")
        for q in self.quote_list:
            if q is None:
                raise ValueError("List of OptionQuote contains None values")

    def __repr__(self):
        return f"<OptionBook> with {len(self.quote_list)} quotes"

    def dropna_iv(self):
        """
        Returns a copy of the book without invalid values for IV 

        Returns
        -------
        OptionBook
            Returns an OptionBook with clean IV

        Notes
        -----
        np.nan and None are removed.  
        """
        quote_list_clean = [q for q in self.quote_list if (not q.iv is None) and (not np.isnan(q.iv)) and (np.isfinite(q.iv))]
        return OptionBook(quote_list_clean)


    def compute_iv(self, keep_nan: bool = True):
        """
        Compute implied volatility for each quote in the book.
        
        Parameters
        ----------
        keep_nan : bool
            Whether to keep quotes where arbitrage condition is not met for IV computation
        
        Returns
        -------
        OptionBook 
            Book with the same OptionQuotes with IV
        """
        from .black_scholes import compute_iv
        quote_list_iv = [compute_iv(q.market_price, q) for q in self.quote_list]

        if keep_nan: return OptionBook(quote_list_iv)
        else: return OptionBook(quote_list_iv).dropna_iv()

    def unique_T(self)-> list[float]:
        """
        Returns a list of the unique time to maturity in the book

        Returns
        -------
        list[float]
            The list of time to maturities in year
        """
        seen_T = set()
        seen_mat = set()
        for q in self.quote_list:
            if q.maturity not in seen_mat:
                seen_mat.add(q.maturity)
                seen_T.add(q.T)
        return sorted(list(seen_T))
    
    @property
    def F(self) -> np.ndarray:
        """
        Returns the forwards of the options 
        calculated as F = S * exp(r*T)
        """
        return self.S * np.exp(self.r * self.T)
    
    @property
    def LM(self) -> np.ndarray:
        """
        Returns the log-moneyness of the options
        calculated as k = log(K/F)
        """
        F = self.F
        return np.log(self.K/F)
    
    @property    
    def W(self) -> np.ndarray:
        """
        Returns the total variance of the options 
        calculated as W = (iv**2) * T
        """
        return (self.iv)**2 * self.T

    @property
    def T(self):
        return np.array([q.T for q in self.quote_list])
    
    @property
    def K(self):
        return np.array([q.K for q in self.quote_list])

    @property
    def iv(self):
        return np.array([q.iv for q in self.quote_list])

    @property
    def maturity(self):
        return np.array([q.maturity for q in self.quote_list])

    @property
    def S(self):
        return np.array([q.S for q in self.quote_list])
    
    @property
    def r(self):
        return np.array([q.r for q in self.quote_list])
    
    @property
    def bid(self):
        return np.array([q.bid for q in self.quote_list])

    @property
    def ask(self):
        return np.array([q.ask for q in self.quote_list])
    
    @property
    def spread(self):
        return np.array([q.ask -q.bid for q in self.quote_list])
    
    @property
    def market_prices(self):
        return np.array([q.market_price for q in self.quote_list])
    
    def __getitem__(self, key):
        quote_list = [q for q, keep in zip(self.quote_list, key) if keep]
        return OptionBook(quote_list)

    def __iter__(self):
        return iter(self.quote_list)

    def __len__(self):
        return len(self.quote_list)
    

#------------------------------------------------------------#
#                                                            #
#                     FUNCTIONS                              #----------------------------------------------------------------------------
#                                                            #
#------------------------------------------------------------#

def fetch_option_data(ticker:str, maturities: list[str], o_type: OptionType)->pd.DataFrame:
    """ 
    Return an option chain for a given ticker and maturities. 

    Parameters
    ----------
    ticker : str 
        The ticker of the stock of the underlying asset
    maturities : list[str]
        List of maturities in format yyyy-mm-dd
    type : OptionType 
        Type of the option

    Returns
    -------
    pandas.DataFrame 
        DataFrame containing option data

    Notes
    -----
    All the data comes from yfinance.
    The price of the underlying S is computed with with yfinance latest price. 
    """

    dfs = []
    stock = yf.Ticker(ticker)
    s = stock.fast_info["last_price"]
    for d in maturities:
        if o_type == OptionType.EUROPEAN_CALL:
            option = stock.option_chain(d).calls
        if o_type == OptionType.EUROPEAN_PUT:
            option = stock.option_chain(d).puts
        option["maturity"] = d
        dfs.append(option)

    df = pd.concat(dfs, ignore_index=True)
    if df.empty:
        raise FetchedDataError("options.fetch_option_data : could not retrieve any data from source.")
    if df.isna().sum().sum() > 0: 
        warnings.warn("options.fetch_option_data : fetched data contains nan values.", FetchedDataWarning)
    df["type"] = o_type
    df["S"] = s
    return df

def to_quote(df: pd.DataFrame, r: float, valuation_date: Optional[str] = None, ticker: Optional[str]= None) -> list[OptionQuote]:
    """
    Converts a fetched df from fetch_option_data to a list of OptionQuote

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the option chain
    r : float 
        The risk-free rate
    valuation_date : Optional[str]
        Date used to compute time to maturity in years; default will use today's date
    ticker : Optional[str]
        Ticker of the underlying asset

    Returns
    -------
    list[OptionQuote]
        A list of OptionQuote extracted from the df.
    """
    
    quote_list = []
    for _, row in df.iterrows():
        T = time_to_mat(row.maturity, valuation_date)
        contract = OptionContract(row.type, maturity=row.maturity, T=T, K=row.strike, ticker=ticker)
        market = MarketState(row.S, r)
        quote_list.append(OptionQuote(contract, market, market_price = (row.bid+row.ask)/2, bid=row.bid, ask=row.ask)) #compute mid price

    return quote_list

def compute_T(quote : OptionQuote, valuation_date: Optional[str] = None):
    """
    Returns a new OptionQuote with T computed.

    Parameters
    ----------
    quote : OptionQuote
        The OptionQuote on which to compute T
    valuation_date : Optional[str]
        Date from which to compute. Default will use today's date

    Returns
    -------
    OptionQuote
        A new OptionQuote with same characteristics and T
    """
    
    T = time_to_mat(quote.maturity, valuation_date)
    contract = replace(quote.contract, T=T)
    return replace(quote, contract = contract)
    

def time_to_mat( maturity: str, today: Optional[str] = None)->float:
    """
    Returns the time to maturity in years
    
    Parameters
    ----------
    maturity : str
        The date of expiry of the contract.
    today : Optional[str]
        Date from which to compute time to maturity. Default will compute from today's date

    Returns
    -------
    float
        The time to maturity in years.
    """
    expiry = datetime.strptime(maturity, "%Y-%m-%d")

    if not today:
        now = datetime.now()
    else:
        now = datetime.strptime(today, "%Y-%m-%d")
    return(expiry - now).total_seconds() / (365.0 * 24.0 * 3600.0)

