# phase2_gradient_descent.py
# Phase 2: Gradient Descent Optimization for Energy-Inspired Portfolio

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Import Phase 1 parameters
# You can also copy them here directly if you want standalone script

# Number of assets
N = 5

# Expected returns
mu = np.array([0.05, 0.08, 0.12, 0.07, 0.09])

# Covariance matrix
Sigma = np.array([
    [0.005, 0.002, 0.001, 0.002, 0.001],
    [0.002, 0.006, 0.002, 0.001, 0.002],
    [0.001, 0.002, 0.007, 0.002, 0.001],
    [0.002, 0.001, 0.002, 0.006, 0.002],
    [0.001, 0.002, 0.001, 0.002, 0.005]
])

# Hyperparameters
lambda_risk = 0.5
gamma_energy = 0.1

# Initial weights
w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

# Gradient descent parameters
learning_rate = 0.01
iterations = 1000

# -----------------------------
# Loss function
def loss(w):
    energy_term = gamma_energy * np.sum((w - 1/N)**2)
    return w.T @ Sigma @ w - lambda_risk * mu.T @ w + energy_term

# Gradient of the loss function
def gradient(w):
    grad_risk = 2 * Sigma @ w
    grad_return = -lambda_risk * mu
    grad_energy = 2 * gamma_energy * (w - 1/N)
    return grad_risk + grad_return + grad_energy

# -----------------------------
# Gradient descent loop
loss_history = []

for i in range(iterations):
    grad = gradient(w)
    w = w - learning_rate * grad
    # Project weights to sum=1 and no negative weights
    w = np.maximum(w, 0)
    w = w / np.sum(w)
    loss_history.append(loss(w))

# -----------------------------
# Final Results
print("Optimized weights:", w)
portfolio_return = mu.T @ w
portfolio_risk = w.T @ Sigma @ w
energy = gamma_energy * np.sum((w - 1/N)**2)

print(f"Portfolio expected return: {portfolio_return:.4f}")
print(f"Portfolio risk (variance): {portfolio_risk:.4f}")
print(f"Energy penalty term: {energy:.4f}")

# -----------------------------
# Plot loss vs iterations
plt.figure(figsize=(8,5))
plt.plot(loss_history, color='blue')
plt.title("Loss vs Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
