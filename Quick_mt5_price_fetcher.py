# mt5_bridge/price.py
import MetaTrader5 as mt5
from pricer.black_scholes import bs_call
from pricer.utils import get_market_data

def price_mt5_symbol(symbol: str, strike_pct: float = 1.05, days: int = 30):
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    info = mt5.symbol_info(symbol)
    if not info:
        raise ValueError(f"Symbol {symbol} not found")
    data = get_market_data(symbol.split(".")[0])   # strip suffix if any
    S, sigma, r = data["spot"], data["vol"], data["rate"]
    K = S * strike_pct
    T = days / 365.0
    price = bs_call(S, K, T, r, sigma)
    mt5.shutdown()
    return {"symbol": symbol, "price": price, "strike": K, "days": days}
