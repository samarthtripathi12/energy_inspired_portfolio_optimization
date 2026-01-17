import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -------------------------------
# 1. Simulated market data
# -------------------------------
np.random.seed(42)

n_assets = 5
n_days = 1000

# True (unknown) parameters
true_mu = np.array([0.0006, 0.0005, 0.0007, 0.0004, 0.0003])
true_cov = np.array([
    [0.00010, 0.00002, 0.00001, 0.00000, 0.00001],
    [0.00002, 0.00008, 0.00002, 0.00001, 0.00000],
    [0.00001, 0.00002, 0.00012, 0.00002, 0.00001],
    [0.00000, 0.00001, 0.00002, 0.00009, 0.00002],
    [0.00001, 0.00000, 0.00001, 0.00002, 0.00007]
])

returns = np.random.multivariate_normal(true_mu, true_cov, n_days)

mu = returns.mean(axis=0)
cov = np.cov(returns.T)

# -------------------------------
# 2. Optimization setup
# -------------------------------
risk_aversion = 5.0
uncertainty = 0.0002

def portfolio_variance(w, cov):
    return w.T @ cov @ w

def mean_variance_objective(w, mu, cov, lam):
    return -(mu @ w - lam * portfolio_variance(w, cov))

def robust_objective(w, mu, cov, lam, uncertainty):
    worst_mu = mu - uncertainty
    return -(worst_mu @ w - lam * portfolio_variance(w, cov))

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1) for _ in range(n_assets)]
x0 = np.ones(n_assets) / n_assets

# -------------------------------
# 3. Solve optimizations
# -------------------------------
mv_result = minimize(
    mean_variance_objective,
    x0,
    args=(mu, cov, risk_aversion),
    bounds=bounds,
    constraints=constraints
)

robust_result = minimize(
    robust_objective,
    x0,
    args=(mu, cov, risk_aversion, uncertainty),
    bounds=bounds,
    constraints=constraints
)

w_mv = mv_result.x
w_robust = robust_result.x

# -------------------------------
# 4. Performance comparison
# -------------------------------
mv_return = mu @ w_mv
robust_return = mu @ w_robust

mv_risk = np.sqrt(portfolio_variance(w_mv, cov))
robust_risk = np.sqrt(portfolio_variance(w_robust, cov))

# -------------------------------
# 5. Plot comparison
# -------------------------------
labels = ['Mean-Variance', 'Robust']
returns_plot = [mv_return, robust_return]
risks_plot = [mv_risk, robust_risk]

plt.figure(figsize=(8, 6))
plt.scatter(risks_plot, returns_plot, s=120)

for i, label in enumerate(labels):
    plt.annotate(label, (risks_plot[i], returns_plot[i]), xytext=(5, 5),
                 textcoords='offset points')

plt.xlabel("Risk (Volatility)")
plt.ylabel("Expected Return")
plt.title("Phase 7: Robust vs Classical Portfolio Optimization")
plt.grid(True)

plt.tight_layout()
plt.savefig("plots/phase7_robust_vs_standard.png")
plt.show()

# -------------------------------
# 6. Print results
# -------------------------------
print("\n=== Portfolio Weights ===")
print("Mean-Variance:", np.round(w_mv, 3))
print("Robust:", np.round(w_robust, 3))

print("\n=== Performance ===")
print(f"Mean-Variance Return: {mv_return:.6f}, Risk: {mv_risk:.6f}")
print(f"Robust Return:        {robust_return:.6f}, Risk: {robust_risk:.6f}")
