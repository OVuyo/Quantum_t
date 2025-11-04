"""
Be aware the code is an attempt , not an expert on this topic but it seems to be working 
""

import numpy as np
from scipy.stats import norm

def d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def bs_call(S, K, T, r, sigma):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def greeks(S, K, T, r, sigma):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    vega  = S * norm.pdf(d1) * np.sqrt(T)
    rho   = K * T * np.exp(-r * T) * norm.cdf(d2)
    return {"delta": delta, "gamma": gamma, "theta": theta/365,
            "vega": vega/100, "rho": rho/100}
