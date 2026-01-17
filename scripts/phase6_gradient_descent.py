import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# =============================
# Paths
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# =============================
# Load Phase 5 data
# =============================
mu = pd.read_csv(os.path.join(DATA_DIR, "mu.csv"), index_col=0).values.flatten()
Sigma = pd.read_csv(os.path.join(DATA_DIR, "covariance.csv"), index_col=0).values

N = len(mu)

print("Loaded mu and Sigma for", N, "assets")

# =============================
# Hyperparameters
# =============================
lambda_risk = 0.5
gamma_penalty = 10.0
learning_rate = 0.05
iterations = 500

# =============================
# Initialize weights
# =============================
w = np.ones(N) / N
loss_history = []

# =============================
# Energy function
# =============================
def energy(w):
    risk = lambda_risk * w.T @ Sigma @ w
    ret = - mu.T @ w
    penalty = gamma_penalty * (np.sum(w) - 1)**2
    return risk + ret + penalty

# =============================
# Gradient of energy
# =============================
def gradient(w):
    grad_risk = 2 * lambda_risk * Sigma @ w
    grad_return = - mu
    grad_penalty = 2 * gamma_penalty * (np.sum(w) - 1)
    return grad_risk + grad_return + grad_penalty

# =============================
# Gradient Descent Loop
# =============================
for i in range(iterations):
    w = w - learning_rate * gradient(w)
    loss_history.append(energy(w))

    if i % 100 == 0:
        print(f"Iteration {i}, Energy = {loss_history[-1]:.6f}")

# =============================
# Normalize weights
# =============================
w = np.clip(w, 0, None)
w = w / np.sum(w)

# =============================
# Save results
# =============================
weights_df = pd.DataFrame(w, columns=["Weight"])
weights_df.to_csv(os.path.join(DATA_DIR, "optimized_weights.csv"))
print("\nOptimized weights saved")

# =============================
# Plot convergence
# =============================
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("Gradient Descent Convergence")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "energy_convergence.png"))
plt.close()

print("Energy convergence plot saved")
print("\nPHASE 6 COMPLETED SUCCESSFULLY ✅")
