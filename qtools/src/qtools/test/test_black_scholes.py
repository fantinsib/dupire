r"""
 _     _            _                  _           _           
| |__ | | __ _  ___| | __     ___  ___| |__   ___ | | ___  ___ 
| '_ \| |/ _` |/ __| |/ /____/ __|/ __| '_ \ / _ \| |/ _ \/ __|
| |_) | | (_| | (__|   <_____\__ \ (__| | | | (_) | |  __/\__ \
|_.__/|_|\__,_|\___|_|\_\    |___/\___|_| |_|\___|_|\___||___/
                                                               
 _            _       
| |_ ___  ___| |_ ___ 
| __/ _ \/ __| __/ __|
| ||  __/\__ \ |_\__ \
 \__\___||___/\__|___/
                      
"""

import math

import numpy as np
import pytest

from qtools.black_scholes import black_scholes, black_scholes_price, compute_iv, iv_solver
from qtools.errors import VolSolverWarning
from qtools.options import MarketState, OptionContract, OptionQuote, OptionType


def test_black_scholes_call_price_known_value():
    price = black_scholes(
        OptionType.EUROPEAN_CALL,
        S=100.0,
        K=100.0,
        T=1.0,
        sigma=0.2,
        r=0.05,
    )
    assert price == pytest.approx(10.4506, rel=1e-4)


def test_black_scholes_put_call_parity():
    S = 100.0
    K = 110.0
    T = 0.5
    sigma = 0.25
    r = 0.03

    call = black_scholes(OptionType.EUROPEAN_CALL, S, K, T, sigma, r)
    put = black_scholes(OptionType.EUROPEAN_PUT, S, K, T, sigma, r)
    parity = call - put
    rhs = S - K * math.exp(-r * T)

    assert parity == pytest.approx(rhs, rel=1e-10)


@pytest.mark.parametrize(
    "S,K,T,sigma",
    [
        (-1.0, 100.0, 1.0, 0.2),
        (100.0, -1.0, 1.0, 0.2),
        (100.0, 100.0, 0.0, 0.2),
        (100.0, 100.0, 1.0, 0.0),
    ],
)
def test_black_scholes_invalid_parameters_raise(S, K, T, sigma):
    with pytest.raises(ValueError):
        black_scholes(OptionType.EUROPEAN_CALL, S, K, T, sigma, r=0.01)


def test_black_scholes_price_contract_wrapper():
    contract = OptionContract(
        type=OptionType.EUROPEAN_CALL,
        K=95.0,
        maturity="2030-01-01",
        T=1.5,
        ticker="TEST",
    )
    market = MarketState(S=100.0, r=0.02)
    price = black_scholes_price(contract, market, sigma=0.3)
    direct = black_scholes(OptionType.EUROPEAN_CALL, 100.0, 95.0, 1.5, 0.3, 0.02)
    assert price == pytest.approx(direct)


def test_iv_solver_recovers_sigma():
    contract = OptionContract(
        type=OptionType.EUROPEAN_PUT,
        K=100.0,
        maturity="2027-06-30",
        T=2.0,
    )
    market = MarketState(S=105.0, r=0.01)
    sigma = 0.35
    market_price = black_scholes(contract.type, market.S, contract.K, contract.T, sigma, market.r)

    iv = iv_solver(market_price, contract, market)
    assert iv == pytest.approx(sigma, rel=1e-6)


def test_iv_solver_arbitrage_warning_returns_nan():
    contract = OptionContract(
        type=OptionType.EUROPEAN_CALL,
        K=100.0,
        maturity="2026-12-31",
        T=1.0,
    )
    market = MarketState(S=100.0, r=0.0)
    market_price = 150.0

    with pytest.warns(VolSolverWarning):
        iv = iv_solver(market_price, contract, market, strict=False)

    assert np.isnan(iv)


def test_compute_iv_returns_new_quote_with_iv():
    contract = OptionContract(
        type=OptionType.EUROPEAN_CALL,
        K=120.0,
        maturity="2026-06-30",
        T=1.0,
    )
    market = MarketState(S=130.0, r=0.02)
    sigma = 0.22
    market_price = black_scholes(contract.type, market.S, contract.K, contract.T, sigma, market.r)
    quote = OptionQuote(contract=contract, market_state=market, market_price=market_price)

    updated = compute_iv(market_price, quote)

    assert updated.iv == pytest.approx(sigma, rel=1e-6)
    assert updated.contract == quote.contract
    assert updated.market_state == quote.market_state
