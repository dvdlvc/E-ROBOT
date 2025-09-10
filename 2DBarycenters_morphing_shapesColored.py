#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Morphing between 2D shapes using truncated (EMD) Wasserstein-1 distance.
This script generates a red circle and a blue square on white backgrounds,
and interpolates between them using entropic barycenters.
More precisely, this file generates clean and contaminated images and then computes the morphism
between them, via OT (obtained using large values of lambda_trunc, 
e.g lambda_trunc= 40000) and ROBOT (obtained using small values of 
lambda_trunc, e.g. lambda_trunc=4). You may increase the number
of images in between the original and final images changing the variable
nb_images) to have a better view of the geodesic from Shape 1 to Shape 2. 
                                    
       

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from ot.bregman import barycenter as entropic_barycenter

def generate_outlier_coords(shape, num_outliers=10, corner='bottom-right', corner_size_ratio=0.25, seed=42):
    np.random.seed(seed)
    if corner == 'bottom-right':
        x_start = int(shape[0] * (1 - corner_size_ratio))
        y_start = int(shape[1] * (1 - corner_size_ratio))
        random_x = np.random.randint(x_start, shape[0], num_outliers)
        random_y = np.random.randint(y_start, shape[1], num_outliers)
    elif corner == 'top-right':
        x_end = int(shape[0] * corner_size_ratio)
        y_start = int(shape[1] * (1 - corner_size_ratio))
        random_x = np.random.randint(0, x_end, num_outliers)
        random_y = np.random.randint(y_start, shape[1], num_outliers)
    else:
        raise ValueError("Unsupported corner type.")
    return list(zip(random_x, random_y))

def add_outliers(img_array, outlier_coords):
    for x, y in outlier_coords:
        img_array[x, y] = 1.0
    return img_array

def create_circle(shape=(32, 32), radius=4.5, outlier_coords=None):
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    center = (shape[0] // 2, shape[1] // 2)
    mask = (X - center[0])**2 + (Y - center[1])**2 < radius**2
    circle_img = mask.astype(float)
    return add_outliers(circle_img.copy(), outlier_coords)

def create_square(shape=(32, 32), size=9, outlier_coords=None):
    img = np.zeros(shape)
    start = (shape[0] - size) // 2
    img[start:start+size, start:start+size] = 1.0
    return add_outliers(img.copy(), outlier_coords)

def normalize(img):
    total = img.sum()
    return img / total if total > 0 else img

#def nonlinear_transform(img, gamma=1.95, epsilon=1e-10):  # for lambda small (Fig Barycenters top plot)
def nonlinear_transform(img, gamma=0.55, epsilon=1e-10): # for lambda large  (Fig Barycenters bottom plot)
#def nonlinear_transform(img, gamma=2.95, epsilon=1e-10):  # for lambda small (GIF)
#def nonlinear_transform(img, gamma=0.95, epsilon=1e-10):   # for lambda large (GIF)
    transformed = np.power(img + epsilon, gamma)
    max_val = transformed.max()
    return transformed / max_val if max_val > 0 else transformed

# Settings
shape = (32, 32)
num_outliers = 10
num_plots = 5
reg = 0.15
sizesq = 9
radcer = 4.5
#lambda_trunc = 4  # this is lambda small
lambda_trunc = 40000000 # this is lambda large


circle_outliers = generate_outlier_coords(shape, num_outliers, corner='top-right')
square_outliers = generate_outlier_coords(shape, num_outliers, corner='bottom-right')

img1 = normalize(create_circle(shape, radius=radcer, outlier_coords=circle_outliers))
img2 = normalize(create_square(shape, size=sizesq, outlier_coords=square_outliers))

a = img1.flatten()
b = img2.flatten()
A = np.vstack([a, b]).T

x = np.array([[i, j] for i in range(shape[0]) for j in range(shape[1])])
M = cdist(x, x, metric="euclidean")
M[M > lambda_trunc] = lambda_trunc  # truncated cost matrix

t_values = np.linspace(0, 1, num_plots)
weights_list = [[1 - t, t] for t in t_values]

barycenters = []
for i, weights in enumerate(weights_list):
    if i == 0:
        barycenters.append(img1)
    elif i == len(weights_list) - 1:
        barycenters.append(img2)
    else:
        bc = entropic_barycenter(A, M, reg, weights)
        barycenters.append(bc.reshape(shape))

# Apply nonlinear transformation and normalize each image
transformed_images = [nonlinear_transform(img) for img in barycenters]

# Plotting
fig, axs = plt.subplots(1, num_plots, figsize=(15, 4), facecolor='white')

for ax, img, t in zip(axs, transformed_images, t_values):
    color = np.array([1 - t, 0, t])  # Blend red → blue
    img_rgb = np.ones((*img.shape, 3))  # White background

    # Mask where shape exists
    mask = img > 1e-2

    # Color only the shape
    for c in range(3):
        img_rgb[..., c][mask] = img[mask] * color[c] + (1 - img[mask]) * 1.0  # Blend with white

    ax.imshow(img_rgb)
    ax.set_facecolor("white")
    ax.set_title(f"t = {t:.2f}")
    ax.axis("off")

plt.tight_layout()
plt.show()





"""
To create a GIF run the next lines, otherwise comment them
"""

import imageio

# List to store image paths for GIF creation
filenames = []

# Plotting and saving frames
for i, (img, t) in enumerate(zip(transformed_images, t_values)):
    fig, ax = plt.subplots(figsize=(5, 5), facecolor='white') # Create a new figure for each frame
    color = np.array([1 - t, 0, t])  # Blend red → blue
    img_rgb = np.ones((*img.shape, 3))  # White background

    # Mask where shape exists
    mask = img > 1e-2

    # Color only the shape
    for c in range(3):
        img_rgb[..., c][mask] = img[mask] * color[c] + (1 - img[mask]) * 1.0  # Blend with white

    ax.imshow(img_rgb)
    ax.set_facecolor("white")
    ax.set_title(f"t = {t:.2f}")
    ax.axis("off")

    # Save the current figure as a PNG file
    filename = f'frame_{i:03d}.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.1)
    filenames.append(filename)
    plt.close(fig) # Close the figure to free up memory

# Create the GIF
gif_filename = 'morphing_animation.gif'
with imageio.get_writer(gif_filename, mode='I', fps=2) as writer: # fps controls animation speed
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

print(f"GIF saved as {gif_filename}")

# Optional: Clean up individual frame files
import os
for filename in filenames:
    os.remove(filename)

