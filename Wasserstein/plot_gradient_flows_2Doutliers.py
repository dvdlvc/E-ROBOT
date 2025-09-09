#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 20:09:15 2025

@author: lavecchi
"""


import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from geomloss import SamplesLoss
from random import choices
from imageio import imread

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

def display_samples(ax, x, color, outlier_mask=None):
    x_ = x.detach().cpu().numpy()
    if outlier_mask is None:
        ax.scatter(x_[:, 0], x_[:, 1], 25 * 500 / len(x_), color, edgecolors="none")
    else:
        ax.scatter(x_[~outlier_mask, 0], x_[~outlier_mask, 1],
                   25 * 500 / len(x_), color[~outlier_mask], edgecolors="none")
        ax.scatter(x_[outlier_mask, 0], x_[outlier_mask, 1],
                   25 * 500 / len(x_), c='black', marker='*', edgecolors="black")

def gradient_flow(loss, lr=0.05, outlier_mask=None):
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
            display_samples(ax, x_i, colors, outlier_mask=outlier_mask)
            ax.set_title("t = {:1.2f}".format(lr * i))
            plt.axis([0, 2, 0, 2])
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
        x_i.data -= lr * len(x_i) * g
    plt.title("t = {:1.2f}, elapsed time: {:.2f}s/it".format(lr * i, (time.time() - t_0) / Nsteps))
    plt.savefig("FlowLaplacian.eps")
    plt.show()
    return x_i

# Load data
N, M = 512, 512
X_main = draw_samples("density_a.png", N, dtype)
Y_j = draw_samples("density_b.png", M, dtype)

# Add 20 outliers far from the main blob

torch.manual_seed(1978)
outliers = 1.8 + 0.1 * torch.randn(20, 2)
outliers = outliers.type(dtype)
X_i = torch.cat([X_main, outliers], dim=0)
outlier_mask = torch.zeros(X_i.shape[0], dtype=torch.bool)
outlier_mask[-20:] = True

# Run gradient flow with Sinkhorn loss
#gradient_flow(SamplesLoss("gaussian", blur=0.05),outlier_mask=outlier_mask)
#gradient_flow(SamplesLoss("sinkhorn", p=2, blur=0.05),outlier_mask=outlier_mask)

gradient_flow(SamplesLoss("laplacian", blur=0.1),outlier_mask=outlier_mask)