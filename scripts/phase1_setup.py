# phase1_setup.py
# Phase 1: Problem Setup for Energy-Inspired Portfolio Optimization

# -----------------------------
# Number of assets
N = 5

# Expected returns (simulated)
mu = [0.05, 0.08, 0.12, 0.07, 0.09]

# Covariance matrix (simulated)
Sigma = [
    [0.005, 0.002, 0.001, 0.002, 0.001],
    [0.002, 0.006, 0.002, 0.001, 0.002],
    [0.001, 0.002, 0.007, 0.002, 0.001],
    [0.002, 0.001, 0.002, 0.006, 0.002],
    [0.001, 0.002, 0.001, 0.002, 0.005]
]

# -----------------------------
# Hyperparameters for loss function
lambda_risk = 0.5  # risk vs return factor
gamma_energy = 0.1  # physics-inspired energy penalty

# Initial portfolio weights
w = [0.2, 0.2, 0.2, 0.2, 0.2]

# -----------------------------
# Optional: Advanced settings
allow_short = False  # True allows negative weights

# -----------------------------
# Verification print statements
print("Phase 1 setup complete!")
print("Number of assets:", N)
print("Initial weights:", w)
print("Expected returns:", mu)
print("Covariance matrix:", Sigma)
print("Lambda (risk factor):", lambda_risk)
print("Gamma (energy penalty):", gamma_energy)
print("Allow shorting:", allow_short)
