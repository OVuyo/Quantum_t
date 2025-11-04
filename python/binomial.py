American option pricing via binomial tree (Cox-Ross-Rubinstein).

Reference:
    Saxena, A., Mancilla, J., Montalban, I., Pere, C. (2023).
    *Financial Modeling Using Quantum Computing*, Packt Publishing.
    Chapter 4 – “Derivative Valuation” (binomial tree for early exercise).
"""

import numpy as np

def binomial_american(
    S: float, K: float, T: float, r: float, sigma: float,
    N: int = 200, option_type: str = "call"
) -> float:
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Stock price tree
    stock = np.asarray([S * (u ** j) * (d ** (i - j))
                        for i in range(N + 1)
                        for j in range(i + 1)])
    stock = stock.reshape((N + 1, N + 1))

    # Payoff at maturity
    payoff = np.maximum(stock[:, -1] - K, 0) if option_type == "call" \
        else np.maximum(K - stock[:, -1], 0)

    # Backward induction with early-exercise check
    for i in range(N - 1, -1, -1):
        exercise = np.maximum(stock[:i + 1, i] - K, 0) if option_type == "call" \
            else np.maximum(K - stock[:i + 1, i], 0)
        continuation = np.exp(-r * dt) * (p * payoff[:i + 1] + (1 - p) * payoff[1:i + 2])
        payoff = np.maximum(exercise, continuation)

    return payoff[0]
