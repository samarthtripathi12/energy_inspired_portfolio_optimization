import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ---- Parameters ----
mu = np.array([0.05, 0.08, 0.12, 0.07, 0.09])
Sigma = np.array([
    [0.005, 0.002, 0.001, 0.002, 0.001],
    [0.002, 0.006, 0.002, 0.001, 0.002],
    [0.001, 0.002, 0.007, 0.002, 0.001],
    [0.002, 0.001, 0.002, 0.006, 0.002],
    [0.001, 0.002, 0.001, 0.002, 0.005]
])

N = len(mu)
lambdas = np.linspace(0.01, 0.99, 40)

returns = []
risks = []

# ---- Constraints ----
def weight_sum_constraint(w):
    return np.sum(w) - 1

bounds = [(0, 1)] * N
constraints = {'type': 'eq', 'fun': weight_sum_constraint}

# ---- Optimization Loop ----
for lam in lambdas:
    def loss(w):
        risk = w.T @ Sigma @ w
        ret = mu @ w
        return lam * risk - (1 - lam) * ret

    w0 = np.ones(N) / N
    res = minimize(loss, w0, bounds=bounds, constraints=constraints)

    w_opt = res.x
    returns.append(mu @ w_opt)
    risks.append(w_opt.T @ Sigma @ w_opt)

# ---- Plot ----
plt.figure()
plt.plot(risks, returns, marker='o')
plt.xlabel("Portfolio Risk")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier")
plt.grid(True)

plt.savefig("plots/phase7_efficient_frontier.png")
plt.show()
