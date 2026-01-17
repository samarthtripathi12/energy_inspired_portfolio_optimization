# phase3_advanced_optimization.py
# Phase 3: Advanced Optimization - Multiple Scenarios & Saving Results

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# -----------------------------
# Phase 1 parameters (can import from phase1_setup if you want)
N = 5
mu = np.array([0.05, 0.08, 0.12, 0.07, 0.09])
Sigma = np.array([
    [0.005, 0.002, 0.001, 0.002, 0.001],
    [0.002, 0.006, 0.002, 0.001, 0.002],
    [0.001, 0.002, 0.007, 0.002, 0.001],
    [0.002, 0.001, 0.002, 0.006, 0.002],
    [0.001, 0.002, 0.001, 0.002, 0.005]
])
w_init = np.array([0.2]*N)

# Gradient descent parameters
learning_rate = 0.01
iterations = 1000

# Hyperparameter sets
lambda_list = [0.3, 0.5, 0.7]
gamma_list = [0.05, 0.1, 0.2]

# Ensure folders exist
os.makedirs("../plots", exist_ok=True)
os.makedirs("../data", exist_ok=True)
os.makedirs("../gifs", exist_ok=True)

# -----------------------------
# Prepare CSV for results
csv_file = "../data/optimized_portfolios.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["lambda", "gamma", "weights", "portfolio_return", "portfolio_risk", "energy_term"])

# -----------------------------
# Gradient descent functions
def loss(w, lam, gamma):
    energy_term = gamma * np.sum((w - 1/N)**2)
    return w.T @ Sigma @ w - lam * mu.T @ w + energy_term

def gradient(w, lam, gamma):
    grad_risk = 2 * Sigma @ w
    grad_return = -lam * mu
    grad_energy = 2 * gamma * (w - 1/N)
    return grad_risk + grad_return + grad_energy

# -----------------------------
# Loop over lambda and gamma
for lam in lambda_list:
    for gamma in gamma_list:
        w = w_init.copy()
        loss_history = []

        for i in range(iterations):
            grad = gradient(w, lam, gamma)
            w = w - learning_rate * grad
            # Project weights to sum=1 and no negatives
            w = np.maximum(w, 0)
            w = w / np.sum(w)
            loss_history.append(loss(w, lam, gamma))

        # Compute final metrics
        portfolio_return = mu.T @ w
        portfolio_risk = w.T @ Sigma @ w
        energy_term = gamma * np.sum((w - 1/N)**2)

        # Save results to CSV
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([lam, gamma, w.tolist(), portfolio_return, portfolio_risk, energy_term])

        # Save plot
        plt.figure(figsize=(8,5))
        plt.plot(loss_history, color='blue')
        plt.title(f"Loss vs Iterations (lambda={lam}, gamma={gamma})")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(f"../plots/loss_lambda{lam}_gamma{gamma}.png")
        plt.close()  # Close to avoid too many open figures

print("Phase 3 complete! All plots saved in 'plots/' and results in 'data/optimized_portfolios.csv'")
