import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =============================
# Path handling (Windows safe)
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("Base directory:", BASE_DIR)
print("Data directory:", DATA_DIR)
print("Plots directory:", PLOTS_DIR)

# =============================
# Portfolio setup
# =============================
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
N = len(tickers)
T = 252  # trading days

np.random.seed(42)

# Daily return assumptions
mu_true = np.array([0.0005, 0.0007, 0.0009, 0.0006, 0.0008])
sigma_true = np.array([0.012, 0.013, 0.015, 0.011, 0.014])

# =============================
# Simulate price paths
# =============================
dates = pd.date_range("2023-01-01", periods=T, freq="B")
prices = pd.DataFrame(index=dates)

for i, ticker in enumerate(tickers):
    returns = np.random.normal(mu_true[i], sigma_true[i], T)
    prices[ticker] = 100 * np.cumprod(1 + returns)

prices.to_csv(os.path.join(DATA_DIR, "prices.csv"))
print("✔ prices.csv saved")

# =============================
# Compute returns
# =============================
returns = prices.pct_change().dropna()
returns.to_csv(os.path.join(DATA_DIR, "returns.csv"))
print("✔ returns.csv saved")

# =============================
# Expected return & covariance
# =============================
mu = returns.mean()
cov = returns.cov()

mu.to_csv(os.path.join(DATA_DIR, "mu.csv"), header=["Expected Return"])
cov.to_csv(os.path.join(DATA_DIR, "covariance.csv"))

print("✔ mu.csv saved")
print("✔ covariance.csv saved")

# =============================
# Visualizations
# =============================
plt.figure(figsize=(10, 5))
for ticker in tickers:
    plt.plot(prices[ticker], label=ticker)
plt.title("Simulated Asset Price Paths")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "price_paths.png"))
plt.close()
print("✔ price_paths.png saved")

plt.figure(figsize=(10, 5))
for ticker in tickers:
    plt.plot(returns[ticker], label=ticker)
plt.title("Daily Returns")
plt.xlabel("Date")
plt.ylabel("Return")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "returns_plot.png"))
plt.close()
print("✔ returns_plot.png saved")

print("\nPHASE 5 COMPLETED SUCCESSFULLY ✅")
