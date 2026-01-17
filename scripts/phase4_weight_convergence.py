# phase4_weight_convergence.py
# Phase 4: Animated GIF of Weight Convergence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os

# -----------------------------
# Phase 1 parameters
N = 5
mu = np.array([0.05, 0.08, 0.12, 0.07, 0.09])
Sigma = np.array([
    [0.005, 0.002, 0.001, 0.002, 0.001],
    [0.002, 0.006, 0.002, 0.001, 0.002],
    [0.001, 0.002, 0.007, 0.002, 0.001],
    [0.002, 0.001, 0.002, 0.006, 0.002],
    [0.001, 0.002, 0.001, 0.002, 0.005]
])
w = np.array([0.2]*N)

# Hyperparameters
lambda_risk = 0.5
gamma_energy = 0.1
learning_rate = 0.01
iterations = 100

# Ensure gifs folder exists
os.makedirs("../gifs", exist_ok=True)

# -----------------------------
# Loss & gradient functions
def loss(w):
    return w.T @ Sigma @ w - lambda_risk * mu.T @ w + gamma_energy * np.sum((w - 1/N)**2)

def gradient(w):
    grad_risk = 2 * Sigma @ w
    grad_return = -lambda_risk * mu
    grad_energy = 2 * gamma_energy * (w - 1/N)
    return grad_risk + grad_return + grad_energy

# -----------------------------
# Store weights at each iteration
weights_history = []

for i in range(iterations):
    grad = gradient(w)
    w = w - learning_rate * grad
    w = np.maximum(w, 0)
    w = w / np.sum(w)
    weights_history.append(w.copy())

weights_history = np.array(weights_history)

# -----------------------------
# Create animation
fig, ax = plt.subplots(figsize=(8,5))
lines = [ax.plot([], [], label=f'Asset {i+1}')[0] for i in range(N)]
ax.set_xlim(0, iterations)
ax.set_ylim(0, 1)
ax.set_xlabel("Iteration")
ax.set_ylabel("Portfolio Weight")
ax.set_title("Portfolio Weights Convergence")
ax.legend()

def animate(i):
    for j, line in enumerate(lines):
        line.set_data(range(i+1), weights_history[:i+1, j])
    return lines

anim = FuncAnimation(fig, animate, frames=iterations, interval=50, blit=True)

# Save GIF
anim.save("../gifs/weights_convergence.gif", writer=PillowWriter(fps=20))
plt.close()

print("Phase 4 complete! GIF saved in 'gifs/weights_convergence.gif'")
