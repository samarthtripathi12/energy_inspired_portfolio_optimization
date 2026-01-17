# phase5_simulated_data.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create folders
os.makedirs("../data", exist_ok=True)
os.makedirs("../plots", exist_ok=True)

# Tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
N = len(tickers)
days = 252  # 1 year of trading days

# -----------------------------
# Simulate realistic adjusted prices
np.random.seed(42)
mu_true = np.array([0.0005, 0.0008, 0.0012, 0.0007, 0.0009])  # daily return
sigma_true = np.array([0.01, 0.012, 0.015, 0.011, 0.013])    # daily volatility

prices = pd.DataFrame(index=pd.date_range(start='2023-01-01', periods=days, freq='B'))
for i, ticker in enumerate(tickers):
    prices[ticker] = 100 * np.cumprod(1 + np.random.normal(mu_true[i], sigma_true[i], days))

prices.to_csv("../data/price_data.csv")
print("Saved simulated price data to data/price_data.csv")

# -----------------------------
# Compute daily returns
returns = prices.pct_change().dropna()
returns.to_csv("../data/returns_data.csv")
print("Saved returns data to data/returns_data.csv")

# Expected returns & covariance
mu = returns.mean().values
Sigma = returns.cov().values

pd.DataFrame(mu, index=tickers, columns=['Expected Return']).to_csv("../data/expected_returns.csv")
pd.DataFrame(Sigma, index=tickers, columns=tickers).to_csv("../data/covariance_matrix.csv")
print("Saved expected returns and covariance matrix to data/ folder")

# -----------------------------
# Plot daily returns
plt.figure(figsize=(10,5))
for ticker in tickers:
    plt.plot(returns[ticker], label=ticker)
plt.title("Daily Returns of Assets (Simulated)")
plt.xlabel("Date")
plt.ylabel("Return")
plt.legend()
plt.tight_layout()
plt.savefig("../plots/daily_returns.png")
plt.close()
print("Saved plot to plots/daily_returns.png")

print("Phase 5 complete ✅ - using simulated data")
