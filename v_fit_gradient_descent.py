"""
Fitting the "V" example data from AnnaCode with gradient descent.

Network (same shape AnnaCode's hint suggests):
    input x  ->  hidden layer: 2 neurons, ReLU  ->  output: 1 neuron, no activation

    z1 = w1*x + b1        h1 = ReLU(z1)
    z2 = w2*x + b2        h2 = ReLU(z2)
    y_hat = v1*h1 + v2*h2 + c

Loss = mean squared error, trained with full-batch gradient descent
(one averaged update per epoch, like AnnaCode's "Train" button).
x and y are normalized (z-scored) before training, also like AnnaCode.

To use YOUR data: in AnnaCode press Save (it downloads a .tsv) and set
DATA_FILE below to that file's path. Otherwise the script makes V data
with the same recipe AnnaCode uses: y = 2|x - 60| + noise, x in [20, 100].
"""
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = "annacode-v.tsv"   # saved from AnnaCode (Fitting mode, V); set to None to generate data
LEARNING_RATE = 0.05      # AnnaCode's default
EPOCHS = 2000
HIDDEN = 2                # neurons in the ReLU layer
SEED = 1

# ── 1. Data ────────────────────────────────────────────────────────────────
if DATA_FILE:
    d = np.loadtxt(DATA_FILE, delimiter="\t", skiprows=1)
    x, y = d[:, -2], d[:, -1]          # columns: Time (s), input, output
else:
    rng = np.random.default_rng(193)
    x = rng.uniform(20, 100, 120)
    y = 2 * np.abs(x - 60) + rng.normal(0, 2, 120)

x_mean, x_std = x.mean(), x.std()
y_mean, y_std = y.mean(), y.std()
xn = (x - x_mean) / x_std
yn = (y - y_mean) / y_std
N = len(xn)

# ── 2. Initialise weights (small random, zero bias) ────────────────────────
rng = np.random.default_rng(SEED)
w = rng.uniform(-1, 1, HIDDEN)     # hidden weights
b = np.zeros(HIDDEN)               # hidden biases
v = rng.uniform(-1, 1, HIDDEN)     # output weights
c = 0.0                            # output bias

def relu(z):
    return np.maximum(0, z)

def forward(xn):
    z = np.outer(xn, w) + b        # (N, HIDDEN) pre-activation
    h = relu(z)                    # (N, HIDDEN) post-activation
    y_hat = h @ v + c              # (N,)
    return z, h, y_hat

# ── 3. Train ───────────────────────────────────────────────────────────────
loss_history = []
for epoch in range(EPOCHS):
    z, h, y_hat = forward(xn)
    err = y_hat - yn
    loss_history.append(np.mean(err ** 2))

    # Backprop (chain rule)
    dL_dyhat = 2 * err / N                         # dL/dy_hat
    dL_dv = h.T @ dL_dyhat                         # dL/dv_j = sum 2/N * err * h_j
    dL_dc = dL_dyhat.sum()                         # dL/dc   = sum 2/N * err
    delta = np.outer(dL_dyhat, v) * (z > 0)        # error at each hidden neuron * ReLU'(z)
    dL_dw = (delta * xn[:, None]).sum(axis=0)      # dL/dw_j = sum delta_j * x
    dL_db = delta.sum(axis=0)                      # dL/db_j = sum delta_j

    # Gradient-descent step
    v -= LEARNING_RATE * dL_dv
    c -= LEARNING_RATE * dL_dc
    w -= LEARNING_RATE * dL_dw
    b -= LEARNING_RATE * dL_db

print(f"Final loss (normalized MSE): {loss_history[-1]:.4f}")
print(f"Final loss (raw units^2):    {loss_history[-1] * y_std**2:.2f}")

# Fold normalization back into raw-unit weights so you can compare with AnnaCode
w_raw = w / x_std
b_raw = b - w * x_mean / x_std
v_raw = v * y_std
c_raw = c * y_std + y_mean
for j in range(HIDDEN):
    print(f"hidden {j+1}: h = ReLU({w_raw[j]:.3f} x + {b_raw[j]:.3f})   output weight {v_raw[j]:.3f}")
print(f"output bias: {c_raw:.3f}")
for j in range(HIDDEN):
    if abs(w_raw[j]) > 1e-9:
        print(f"  neuron {j+1} turns on/off at x = {-b_raw[j] / w_raw[j]:.1f}")

# ── 4. Plots ───────────────────────────────────────────────────────────────
x_line = np.linspace(x.min(), x.max(), 400)
_, _, yn_line = forward((x_line - x_mean) / x_std)
y_line = yn_line * y_std + y_mean

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.scatter(x, y, s=14, alpha=0.6, label="V data")
ax1.plot(x_line, y_line, "r", lw=2, label=f"fit ({HIDDEN} ReLU neurons)")
ax1.set_xlabel("x (input)")
ax1.set_ylabel("y (output)")
ax1.set_title("Data and fit")
ax1.legend()

ax2.plot(loss_history)
ax2.set_yscale("log")
ax2.set_xlabel("epoch")
ax2.set_ylabel("loss (normalized MSE)")
ax2.set_title("Loss vs. epoch")

plt.tight_layout()
plt.savefig("v_fit_plots.png", dpi=150)
plt.show()
