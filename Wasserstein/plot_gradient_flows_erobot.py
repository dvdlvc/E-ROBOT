
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from random import choices
from imageio import imread
#import imageio.v2 as imread

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# Load grayscale image and sample points
def load_image(fname):
    img = imread(fname, mode="F")
    img = (img[::-1, :]) / 255.0
    return 1 - img

def draw_samples(fname, n, dtype=torch.FloatTensor):
    A = load_image(fname)
    xg, yg = np.meshgrid(
        np.linspace(0, 1, A.shape[0]),
        np.linspace(0, 1, A.shape[1]),
        indexing="xy",
    )
    grid = list(zip(xg.ravel(), yg.ravel()))
    dens = A.ravel() / A.sum()
    dots = np.array(choices(grid, dens, k=n))
    dots += (0.5 / A.shape[0]) * np.random.standard_normal(dots.shape)
    return torch.from_numpy(dots).type(dtype)

def display_samples(ax, x, color):
    x_ = x.detach().cpu().numpy()
    ax.scatter(x_[:, 0], x_[:, 1], 25 * 500 / len(x_), color, edgecolors="none")

# Define E-ROBOT loss
class MySamplesLoss:
    def __init__(self, blur=0.05, lambda_val=0.15, debiased=True, n_iter=100):
        self.blur = blur
        self.eps = blur
        self.lambda_val = lambda_val
        self.debiased = debiased
        self.n_iter = n_iter

    def truncated_cost(self, x, y):
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        cost = torch.norm(x_col - y_lin, dim=-1)
        return torch.minimum(cost, torch.tensor(2 * self.lambda_val, device=cost.device))

    def sinkhorn(self, a, x, b, y):
        C = self.truncated_cost(x, y)
        K = torch.exp(-C / self.eps)
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(self.n_iter):
            u = a / (K @ v)
            v = b / (K.t() @ u)
        pi = torch.outer(u, v) * K
        loss_ab = torch.sum(pi * C)
        if self.debiased:
            loss_xx = self._self_sinkhorn(a, x)
            loss_yy = self._self_sinkhorn(b, y)
            loss = loss_ab - 0.5 * (loss_xx + loss_yy)
        else:
            loss = loss_ab
        return loss

    def _self_sinkhorn(self, a, x):
        C = self.truncated_cost(x, x)
        K = torch.exp(-C / self.eps)
        u = torch.ones_like(a)
        v = torch.ones_like(a)
        for _ in range(self.n_iter):
            u = a / (K @ v)
            v = a / (K.t() @ u)
        pi = torch.outer(u, v) * K
        return torch.sum(pi * C)

    def __call__(self, x, y, a=None, b=None):
        N, M = x.shape[0], y.shape[0]
        if a is None:
            a = torch.ones(N, device=x.device) / N
        if b is None:
            b = torch.ones(M, device=y.device) / M
        return self.sinkhorn(a, x, b, y)

# Gradient flow
def gradient_flow(loss, lr=0.05):
    Nsteps = int(5 / lr) + 1
    display_its = [int(t / lr) for t in [0, 0.25, 0.50, 1.0, 2.0, 5.0]]
    colors = (10 * X_i[:, 0]).cos() * (10 * X_i[:, 1]).cos()
    colors = colors.detach().cpu().numpy()
    x_i, y_j = X_i.clone(), Y_j.clone()
    x_i.requires_grad = True
    t_0 = time.time()
    plt.figure(figsize=(12, 8))
    k = 1
    for i in range(Nsteps):
        L_ab = loss(x_i, y_j)
        [g] = torch.autograd.grad(L_ab, [x_i])
        if i in display_its:
            ax = plt.subplot(2, 3, k)
            k += 1
            plt.set_cmap("hsv")
            plt.scatter([10], [10])
            display_samples(ax, y_j, [(0.55, 0.55, 0.95)])
            display_samples(ax, x_i, colors)
            ax.set_title("t = {:1.2f}".format(lr * i))
            plt.axis([0, 2, 0, 2])
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
        x_i.data -= lr * len(x_i) * g
    plt.title("t = {:1.2f}, elapsed time: {:.2f}s/it".format(lr * i, (time.time() - t_0) / Nsteps))

# Load data
N, M = 512, 512
X_i = draw_samples("density_a.png", N, dtype)
Y_j = draw_samples("density_b.png", M, dtype)

# Run gradient flow with E-ROBOT
gradient_flow(MySamplesLoss(blur=0.05, lambda_val=0.35, debiased=True, n_iter=100), lr=0.05)
