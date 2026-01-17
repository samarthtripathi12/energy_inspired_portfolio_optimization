import numpy as np
import matplotlib.pyplot as plt

# ---- Parameters ----
mu = np.array([0.05, 0.08, 0.12, 0.07, 0.09])
Sigma = np.array([
    [0.005, 0.002, 0.001, 0.002, 0.001],
    [0.002, 0.006, 0.002, 0.001, 0.002],
    [0.001, 0.002, 0.007, 0.002, 0.001],
    [0.002, 0.001, 0.002, 0.006, 0.002],
    [0.001, 0.002, 0.001, 0.002, 0.005]
])

lam = 0.5
alpha = 0.05
iterations = 200

def loss(w):
    return lam * (w.T @ Sigma @ w) - (1 - lam) * (mu @ w)

def grad(w):
    return 2 * lam * (Sigma @ w) - (1 - lam) * mu

# ---- Gradient Descent ----
w = np.ones(5) / 5
gd_losses = []

for _ in range(iterations):
    w -= alpha * grad(w)
    w = np.clip(w, 0, 1)
    w /= np.sum(w)
    gd_losses.append(loss(w))

# ---- Adam ----
w = np.ones(5) / 5
m = np.zeros(5)
v = np.zeros(5)
beta1, beta2 = 0.9, 0.999
eps = 1e-8
adam_losses = []

for t in range(1, iterations + 1):
    g = grad(w)
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * (g ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    w -= alpha * m_hat / (np.sqrt(v_hat) + eps)
    w = np.clip(w, 0, 1)
    w /= np.sum(w)

    adam_losses.append(loss(w))

# ---- Plot ----
plt.figure()
plt.plot(gd_losses, label="Gradient Descent")
plt.plot(adam_losses, label="Adam")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Optimizer Comparison")
plt.legend()
plt.grid(True)

plt.savefig("plots/phase6_optimizer_comparison.png")
plt.show()
