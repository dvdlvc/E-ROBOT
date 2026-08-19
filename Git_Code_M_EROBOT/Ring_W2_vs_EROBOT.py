import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import ot

torch.manual_seed(20101978)
np.random.seed(1)

# ---------- Parameters ----------
sigma = 0.1
n_dims = 2
n_features = 2
n_iter = 200
size_batch = 400
eta = 0.05
outlier_mean = 3.0
outlier_std = 0.1


# E-ROBOT hyperparameters
epsilon_erobot = 1.0
lambda_cost_erobot = 0.3

# ---------- Data generation with contamination ----------
def get_data(n_samples, eta=0.0):
    c = torch.rand(size=(n_samples, 1))
    angle = c * 2 * np.pi
    x = torch.cat((torch.cos(angle), torch.sin(angle)), 1)
    x += torch.randn(n_samples, 2) * sigma
    if eta > 0:
        n_out = int(eta * n_samples)
        outliers = torch.randn(n_out, 2) * outlier_std + outlier_mean
        idx = torch.randperm(n_samples)[:n_out]
        x[idx] = outliers
    return x

# ---------- Generator ----------
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(n_features, 200)
        self.fc2 = nn.Linear(200, 500)
        self.fc3 = nn.Linear(500, n_dims)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ---------- E-ROBOT loss (debiased Sinkhorn + truncation) ----------
def sinkhorn_cost_log(x, y, epsilon, lambda_cost=None, num_iters=50):
    C = torch.cdist(x, y, p=2)
    if lambda_cost is not None:
        C = torch.min(C, torch.tensor(2.0 * lambda_cost, device=C.device))
    C = C ** 2

    n, m = x.shape[0], y.shape[0]
    a = torch.ones(n, device=x.device) / n
    b = torch.ones(m, device=y.device) / m
    K = -C / epsilon

    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)
    for _ in range(num_iters):
        log_u = torch.log(a) - torch.logsumexp(K + log_v, dim=1)
        log_v = torch.log(b) - torch.logsumexp(K.T + log_u, dim=1)

    log_P = log_u[:, None] + K + log_v[None, :]
    return torch.sum(torch.exp(log_P) * C)

def erobot_loss(x, y, epsilon, lambda_cost, num_iters=50):
    W_xy = sinkhorn_cost_log(x, y, epsilon, lambda_cost, num_iters)
    W_xx = sinkhorn_cost_log(x, x, epsilon, lambda_cost, num_iters)
    W_yy = sinkhorn_cost_log(y, y, epsilon, lambda_cost, num_iters)
    return W_xy - 0.5 * (W_xx + W_yy)

# ---------- Training function ----------
def train_gan(use_erobot=False, eta=0.0):
    G = Generator()
    optimizer = torch.optim.RMSprop(G.parameters(), lr=0.00019, eps=1e-5)

    n_visu = 100
    xnvisu = torch.randn(n_visu, n_features)
    xvisu = torch.zeros(n_iter, n_visu, n_dims)
    losses = []
    ab = torch.ones(size_batch) / size_batch

    for i in range(n_iter):
        xn = torch.randn(size_batch, n_features)
        xd = get_data(size_batch, eta=eta)
        xg = G(xn)
        xvisu[i, :, :] = G(xnvisu).detach()

        if use_erobot:
            loss = erobot_loss(xd, xg, epsilon_erobot, lambda_cost_erobot, num_iters=50)
        else:
            # ---- W2 APPROACH (exactly as in the original code) ----
            M = ot.dist(xg, xd)          # now M is a PyTorch tensor (if `ot` supports it)
            loss = ot.emd2(ab, ab, M)    # this returns a tensor with gradient )
        losses.append(float(loss.detach()))
        if i % 10 == 0:
            print(f"Iter {i:3d}, loss={losses[-1]:.4f}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return G, xvisu, losses

# ---------- Run experiments ----------
experiments = [
    ("W2 (exact), clean", False, 0.0),
    ("W2 (exact), contaminated", False, eta),
    ("E-ROBOT, clean", True, 0.0),
    ("E-ROBOT, contaminated", True, eta),
]

results = {}
for label, use_erobot, contam in experiments:
    print(f"\n--- Training: {label} ---")
    G, xvisu, losses = train_gan(use_erobot=use_erobot, eta=contam)
    results[label] = (G, xvisu, losses)

# ---------- Plotting ----------
# 1. Loss curves
plt.figure(figsize=(8,5))

# Define style mapping: (color, linestyle)
style_map = {
    "W2 (exact), clean": ('blue', 'dashdot'),
    "W2 (exact), contaminated": ('red', 'dashdot'),
    "E-ROBOT, clean": ('blue', '-'),
    "E-ROBOT, contaminated": ('red', '-'),
}

for label, (_, _, losses) in results.items():
    color, ls = style_map[label]
    plt.semilogy(losses, label=label, color=color, linestyle=ls, linewidth=2)

plt.grid()
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss curves")
plt.savefig("loss_curves.png")
plt.savefig("loss_curves.eps") 
plt.show()


# 2. Final generated samples
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

for idx, (label, (_, xvisu, _)) in enumerate(results.items()):
    ax = axes[idx]
    
    # Determine if this experiment is contaminated
    is_contaminated = "contaminated" in label.lower()
    
    if is_contaminated:
        # Generate a contaminated dataset (with outliers) for this subplot
        xd_contam = get_data(500, eta=eta)
        # Identify outliers: points with norm > 1.5 (ring radius ~1, noise ~0.1)
        norms = torch.norm(xd_contam, dim=1)
        outlier_mask = norms > 1.5
        inlier_mask = ~outlier_mask
        # Plot inliers in orange (clean-like)
        ax.scatter(xd_contam[inlier_mask, 0], xd_contam[inlier_mask, 1],
                   alpha=0.2, color='orange', label='Real (clean)')
        # Plot outliers in black
        ax.scatter(xd_contam[outlier_mask, 0], xd_contam[outlier_mask, 1],
                   alpha=0.8, color='black', s=20, label='Outliers')
    else:
        # Clean experiment: use fixed clean data
        xd_clean_fixed = get_data(500, eta=0.0)
        ax.scatter(xd_clean_fixed[:, 0], xd_clean_fixed[:, 1],
                   alpha=0.2, color='orange', label='Real (clean)')
    
    # Generated data: grey for W2, dodgerblue for E-ROBOT
    xg_final = xvisu[-1, :, :]
    if "E-ROBOT" in label:
        gen_color = 'dodgerblue'
    else:
        gen_color = 'grey'
    ax.scatter(xg_final[:, 0], xg_final[:, 1], alpha=0.5, color=gen_color, label='Generated')
    
    ax.set_title(label)
    ax.set_xlim(-3, 7)
    ax.set_ylim(-3, 7)
    ax.legend()
plt.tight_layout()
plt.savefig("generated_samples.png")
plt.savefig("generated_samples.eps")
plt.show()


# 3. Evolution of E-ROBOT contaminated (ptional, not in the paper)
xvisu_erobot_contam = results["E-ROBOT, contaminated"][1]
n_actual = xvisu_erobot_contam.shape[0]
ivisu = np.linspace(0, n_actual-1, 9, dtype=int)

fig, axes = plt.subplots(3, 3, figsize=(10,10))
axes = axes.flatten()
for i, it in enumerate(ivisu):
    ax = axes[i]
    xd_contam = get_data(500, eta=eta)
    norms = torch.norm(xd_contam, dim=1)
    outlier_mask = norms > 1.5
    inlier_mask = ~outlier_mask

    ax.scatter(xd_contam[inlier_mask, 0], xd_contam[inlier_mask, 1],
               alpha=0.2, color='gray', label='Real (clean)')
    ax.scatter(xd_contam[outlier_mask, 0], xd_contam[outlier_mask, 1],
               alpha=0.8, color='black', s=20, label='Outliers')

    xg = xvisu_erobot_contam[it, :, :]
    ax.scatter(xg[:, 0], xg[:, 1], alpha=0.6, color='dodgerblue', label='Generated')
    ax.set_title(f"Iter {it}")
    ax.set_xlim(-3, 5)
    ax.set_ylim(-3, 5)
    if i == 0:
        ax.legend()
plt.tight_layout()
plt.savefig("evolution_erobot_contaminated.png")
plt.savefig("evolution_erobot_contaminated.eps")
plt.show()