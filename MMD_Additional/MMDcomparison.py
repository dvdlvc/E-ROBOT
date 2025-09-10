#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 16:48:50 2025
@author: lavecchi

This code reproduces the gradient flows using MMD, as in the main
body of the paper
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from random import choices
import imageio.v2 as imageio
#from geomloss import SamplesLoss

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
dtype = torch.float32

# Load grayscale image and sample points
def load_image(fname):
    img = imageio.imread(fname, mode="F")
    img = (img[::-1, :]) / 255.0
    return 1 - img

def draw_samples(fname, n):
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
    return torch.from_numpy(dots).to(device=device, dtype=dtype)

def display_samples(ax, x, color, outlier_mask=None):
    x_ = x.detach().cpu().numpy()
    if outlier_mask is None:
        ax.scatter(x_[:, 0], x_[:, 1], 25 * 500 / len(x_), c=color, edgecolors="none")
    else:
        # Plot normal points
        ax.scatter(x_[~outlier_mask, 0], x_[~outlier_mask, 1],
                   25 * 500 / len(x_), c=color[~outlier_mask], edgecolors="none")
        # Plot outliers in black
        ax.scatter(x_[outlier_mask, 0], x_[outlier_mask, 1],
                   25 * 500 / len(x_), c='black', marker = '*', edgecolors="black")

# MMD Loss implementation
class MMDLoss:
    def __init__(self, blur=0.05, lambda_val=1):
        self.blur = blur
        self.eps = blur
        self.lambda_val = lambda_val

    def truncated_cost(self, x, y):
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        cost = torch.norm(x_col - y_lin, dim=-1)
        return torch.minimum(cost, torch.tensor(2 * self.lambda_val, device=cost.device))

    def kernel_matrix(self, x, y):
        C = self.truncated_cost(x, y)
        K = torch.exp(-C / self.eps)
        return K

    def __call__(self, x, y):
        K_xx = self.kernel_matrix(x, x)
        K_yy = self.kernel_matrix(y, y)
        K_xy = self.kernel_matrix(x, y)
        
        n = x.size(0)
        m = y.size(0)
        
        # Compute MMD^2 (squared MMD)
        term_xx = torch.sum(K_xx) / (n * n)
        term_yy = torch.sum(K_yy) / (m * m)
        term_xy = torch.sum(K_xy) / (n * m)
        
        mmd_squared = term_xx + term_yy - 2 * term_xy
        return mmd_squared  # Return MMD squared (better for gradient flow)




class MMDLossGauss:
    def __init__(self, sigma=0.05):
        self.sigma = sigma  # Gaussian kernel bandwidth
        self.sigma_sq = sigma * sigma  # Precompute sigma squared for efficiency

    def kernel_matrix(self, x, y):
        """
        Compute Gaussian kernel matrix: K(x,y) = exp(-||x-y||² / (2*sigma²))
        """
        x_col = x.unsqueeze(1)  # Shape: (n, 1, d)
        y_lin = y.unsqueeze(0)  # Shape: (1, m, d)
        
        # Compute squared Euclidean distances
        squared_dist = torch.sum((x_col - y_lin) ** 2, dim=-1)
        
        # Gaussian kernel
        K = torch.exp(-squared_dist / (2 * self.sigma_sq))
        return K

    def __call__(self, x, y):
        K_xx = self.kernel_matrix(x, x)
        K_yy = self.kernel_matrix(y, y)
        K_xy = self.kernel_matrix(x, y)
        
        n = x.size(0)
        m = y.size(0)
        
        # Compute MMD^2 (squared MMD)
        term_xx = torch.sum(K_xx) / (n * n)
        term_yy = torch.sum(K_yy) / (m * m)
        term_xy = torch.sum(K_xy) / (n * m)
        
        mmd_squared = term_xx + term_yy - 2 * term_xy
        return mmd_squared  # Return MMD squared (better for gradient flow)



# Gradient flow function
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
    plt.title("t = {:1.2f}".format(lr * i))
    plt.savefig("FlowLarge_MMDGauss.eps")
    plt.show()

# Load data
N, M = 512, 512
X_main = draw_samples("density_a.png", N)
Y_j = draw_samples("density_b.png", M)

# Add 20 outliers far from the main blob
torch.manual_seed(1978)
outliers = 1.8 + 0.1 * torch.randn(20, 2)
outliers = outliers.to(device=device, dtype=dtype)
X_i = torch.cat([X_main, outliers], dim=0)
outlier_mask = torch.zeros(X_i.shape[0], dtype=torch.bool)
outlier_mask[-20:] = True

# Run gradient flow with MMD loss
#gradient_flow(MMDLoss(blur=0.05, lambda_val=0.6), lr=0.05, outlier_mask=outlier_mask)



#Run gradient flow with MMD Gaussian loss (sigma small)
#gradient_flow(MMDLossGauss(sigma=0.05), lr=0.05, outlier_mask=outlier_mask)

#If the radius  of the kernel is too small, particles  won’t be attracted to the target, and may spread out 

# Run gradient flow with MMD Gaussian loss (sigma large)
gradient_flow(MMDLossGauss(sigma=0.65), lr=0.05, outlier_mask=outlier_mask)

