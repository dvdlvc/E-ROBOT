#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 16:48:50 2025

@author: lavecchi
This code produces the additional plots in Appendix B, about MMDs
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from random import choices

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
dtype = torch.float32

# ==================== SHAPE PARAMETERS - EASILY ADJUSTABLE ====================
# Adjust these parameters to change the positions and sizes of the shapes

# Square parameters
SQUARE_CENTER = (0.2, 0.4)  # Center position of the square (x, y)
SQUARE_SIZE = 0.2           # Size of the square (side length)

# Oval parameters  
OVAL_CENTER = (0.6, 0.6)    # Center position of the oval (x, y)
OVAL_WIDTH = 0.3           # Width of the oval
OVAL_HEIGHT = 0.1           # Height of the oval

# ==============================================================================

def create_square_density(size=100):
    """Create a square density matrix"""
    density = np.zeros((size, size))
    center = size // 2
    half_size = int(SQUARE_SIZE * size / 2)
    
    start_row = max(0, center - half_size)
    end_row = min(size, center + half_size)
    start_col = max(0, center - half_size)
    end_col = min(size, center + half_size)
    
    density[start_row:end_row, start_col:end_col] = 1.0
    
    # Apply Gaussian smoothing for nicer sampling
    from scipy.ndimage import gaussian_filter
    density = gaussian_filter(density, sigma=1)
    return density

def create_oval_density(size=100):
    """Create an oval density matrix"""
    y, x = np.ogrid[-0.5:0.5:size*1j, -0.5:0.5:size*1j]
    
    # Oval equation: (x/width)^2 + (y/height)^2 <= 1
    oval_mask = ((x / OVAL_WIDTH)**2 + (y / OVAL_HEIGHT)**2 <= 1)
    density = oval_mask.astype(float)
    
    # Apply Gaussian smoothing
    from scipy.ndimage import gaussian_filter
    density = gaussian_filter(density, sigma=1)
    return density

def draw_samples_from_density(density, n, center=(0.5, 0.5)):
    """
    Draw samples from a given density matrix with specified center
    """
    size = density.shape[0]
    xg, yg = np.meshgrid(
        np.linspace(center[0] - 0.5, center[0] + 0.5, size),
        np.linspace(center[1] - 0.5, center[1] + 0.5, size),
        indexing="xy",
    )
    grid = list(zip(xg.ravel(), yg.ravel()))
    dens = density.ravel() / (density.sum() + 1e-10)
    dots = np.array(choices(grid, dens, k=n))
    dots += (0.5 / size) * np.random.standard_normal(dots.shape)
    return torch.from_numpy(dots).to(device=device, dtype=dtype)

def display_samples(ax, x, color, outlier_mask=None):
    x_ = x.detach().cpu().numpy()
    if outlier_mask is None:
        ax.scatter(x_[:, 0], x_[:, 1], 25 * 500 / len(x_), c=color, edgecolors="none")
    else:
        # Plot normal points with rainbow colors
        ax.scatter(x_[~outlier_mask, 0], x_[~outlier_mask, 1],
                   25 * 500 / len(x_), c=color[~outlier_mask], edgecolors="none")
        # Plot outliers in black
        ax.scatter(x_[outlier_mask, 0], x_[outlier_mask, 1],
                   25 * 500 / len(x_), c='black', marker='*', edgecolors="black")

# MMDLaplace Loss implementation
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

# MMD Loss implementation
class MMDLossGauss:
    def __init__(self, sigma=0.05):
        self.sigma = sigma
        self.sigma_sq = sigma * sigma

    def kernel_matrix(self, x, y):
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        squared_dist = torch.sum((x_col - y_lin) ** 2, dim=-1)
        K = torch.exp(-squared_dist / (2 * self.sigma_sq))
        return K

    def __call__(self, x, y):
        K_xx = self.kernel_matrix(x, x)
        K_yy = self.kernel_matrix(y, y)
        K_xy = self.kernel_matrix(x, y)
        
        n = x.size(0)
        m = y.size(0)
        
        term_xx = torch.sum(K_xx) / (n * n)
        term_yy = torch.sum(K_yy) / (m * m)
        term_xy = torch.sum(K_xy) / (n * m)
        
        mmd_squared = term_xx + term_yy - 2 * term_xy
        return mmd_squared

# Gradient flow function
def gradient_flow(loss, lr=0.05, outlier_mask=None, filename_suffix=""):
    Nsteps = int(5 / lr) + 1
    display_its = [int(t / lr) for t in [0, 0.25, 0.50, 1.0, 2.0, 5.0]]
    
    # Create rainbow colors for the square points
    colors = np.zeros((len(X_i), 3))
    if len(X_i) > 0:
        # Rainbow gradient based on x-position for square points
        x_norm = (X_i[:, 0].detach().cpu().numpy() - X_i[:, 0].min().item()) / (X_i[:, 0].max().item() - X_i[:, 0].min().item() + 1e-10)
        for i in range(len(X_i)):
            if not outlier_mask[i] if outlier_mask is not None else True:
                # HSV to RGB conversion for rainbow effect
                h = x_norm[i] * 0.7  # 0.7 gives nice rainbow range
                s, v = 0.8, 0.8
                c = v * s
                x_h = h * 6
                m = v - c
                
                if 0 <= x_h < 1:
                    r, g, b = c, x_h * c, 0
                elif 1 <= x_h < 2:
                    r, g, b = (2 - x_h) * c, c, 0
                elif 2 <= x_h < 3:
                    r, g, b = 0, c, (x_h - 2) * c
                elif 3 <= x_h < 4:
                    r, g, b = 0, (4 - x_h) * c, c
                elif 4 <= x_h < 5:
                    r, g, b = (x_h - 4) * c, 0, c
                else:
                    r, g, b = c, 0, (6 - x_h) * c
                
                colors[i] = [r + m, g + m, b + m]
    
    x_i, y_j = X_i.clone(), Y_j.clone()
    x_i.requires_grad = True
    
    plt.figure(figsize=(12, 8))
    k = 1
    for i in range(Nsteps):
        L_ab = loss(x_i, y_j)
        [g] = torch.autograd.grad(L_ab, [x_i])
        if i in display_its:
            ax = plt.subplot(2, 3, k)
            k += 1
            display_samples(ax, y_j, [(0.55, 0.55, 0.95)])  # Blue for oval
            display_samples(ax, x_i, colors, outlier_mask=outlier_mask)  # Rainbow for square
            ax.set_title("t = {:1.2f}".format(lr * i))
            plt.axis([-0.5, 1, -0.5, 1])
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xticks([], [])
            plt.yticks([], [])
            plt.tight_layout()
        x_i.data -= lr * len(x_i) * g
    
    # Save the final plot as EPS
    eps_filename = f"gradient_flow_{filename_suffix}.eps"
    plt.savefig(eps_filename, format='eps')
    print(f"Saved: {eps_filename}")
    plt.show()

# Create densities and draw samples
N, M = 512, 512
size = 100

print("Creating square and oval densities...")
square_density = create_square_density(size)
oval_density = create_oval_density(size)

# Draw samples
print("Drawing samples...")
X_main = draw_samples_from_density(square_density, N, center=SQUARE_CENTER)
Y_j = draw_samples_from_density(oval_density, M, center=OVAL_CENTER)

# Add outliers
torch.manual_seed(1978)
outliers = 0.05 + 0.1 * torch.randn(20, 2)
outliers = torch.clamp(outliers, 0.1, 0.4)
outliers = outliers.to(device=device, dtype=dtype)
X_i = torch.cat([X_main, outliers], dim=0)
outlier_mask = torch.zeros(X_i.shape[0], dtype=torch.bool)
outlier_mask[-20:] = True

# Show initial distribution
plt.figure(figsize=(8, 6))
x_np = X_i.detach().cpu().numpy()
y_np = Y_j.detach().cpu().numpy()

# Create rainbow colors for visualization
rainbow_colors = np.zeros((len(x_np), 3))
x_min, x_max = x_np[:, 0].min(), x_np[:, 0].max()
for i in range(len(x_np)):
    if not outlier_mask[i]:
        norm_x = (x_np[i, 0] - x_min) / (x_max - x_min + 1e-10)
        hue = norm_x * 0.7
        rainbow_colors[i] = plt.cm.hsv(hue)[:3]

plt.scatter(x_np[~outlier_mask, 0], x_np[~outlier_mask, 1], 
            alpha=0.7, s=20, c=rainbow_colors[~outlier_mask], label='Square (X_i)')
plt.scatter(x_np[outlier_mask, 0], x_np[outlier_mask, 1], 
            alpha=0.9, s=60, c='black', marker='*', label='Outliers')
plt.scatter(y_np[:, 0], y_np[:, 1], 
            alpha=0.7, s=20, c='blue', label='Oval (Y_j)')
plt.legend()
plt.title('Initial Distributions - Square vs Oval')
plt.axis([-0.5, 1, -0.5, 1])
plt.savefig('initial_square_oval.eps', format='eps')
plt.show()

print("Running gradient flow...")
# Run gradient flow with different parameters and save each as separate EPS files
gradient_flow(MMDLoss(blur=0.25, lambda_val=4), lr=0.05, outlier_mask=outlier_mask, filename_suffix="laplace_small")
gradient_flow(MMDLoss(blur=1, lambda_val=4), lr=0.05, outlier_mask=outlier_mask, filename_suffix="laplace_large")
gradient_flow(MMDLossGauss(sigma=0.25), lr=0.05, outlier_mask=outlier_mask, filename_suffix="gauss_small")
gradient_flow(MMDLossGauss(sigma=0.55), lr=0.05, outlier_mask=outlier_mask, filename_suffix="gauss_large")
