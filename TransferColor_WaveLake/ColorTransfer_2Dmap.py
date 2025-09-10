#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  7 10:29:33 2025

@author: lavecchi
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # Use PIL instead of skimage
import ot

def crop_to_square_and_resize(image_path, target_size=(100, 100)):
    """
    Crop image to square shape and resize to target size
    """
    img = Image.open(image_path)
    
    # Get current dimensions
    width, height = img.size
    
    # Calculate crop dimensions for square
    min_dim = min(width, height)
    
    # Calculate crop coordinates (center crop)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    
    # Crop to square
    img_cropped = img.crop((left, top, right, bottom))
    
    # Resize to target size
    img_resized = img_cropped.resize(target_size, Image.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img_resized).astype(np.float64) / 255.0
    
    # Remove alpha channel if present
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    return img_array

# Load and preprocess image (now uses the cropping function)
def load_and_preprocess_image(path, target_size=(100, 100)):
    """Load, crop to square, and preprocess an image using PIL"""
    return crop_to_square_and_resize(path, target_size)

# Convert image to matrix (one pixel per line)
def im2mat(img):
    return img.reshape((img.shape[0] * img.shape[1], -1))

# Convert matrix back to image
def mat2im(X, shape):
    return X.reshape(shape)

# Truncated cost function to handle outliers
def truncated_cost_matrix(X1, X2, truncation_value=2.0):
    """Compute cost matrix with truncated values to handle outliers"""
    M = ot.dist(X1, X2, metric='euclidean')
    # Truncate large distances to handle outliers
    M_truncated = np.minimum(M, truncation_value)
    return M_truncated

# Load your source and target images with consistent square cropping
source_path = "Onda2.jpg"  # Replace with your source image path
target_path = "stprex.jpeg"  # Replace with your target image path

# Use the same target size for both images to ensure consistent definition
target_size = (100, 100)  # You can adjust this size as needed

img1 = load_and_preprocess_image(source_path, target_size)
img2 = load_and_preprocess_image(target_path, target_size)

# Verify both images have the same shape
print(f"Source image shape: {img1.shape}")
print(f"Target image shape: {img2.shape}")

# Convert images to matrices
X1 = im2mat(img1)
X2 = im2mat(img2)

# Compute data distributions
n1 = X1.shape[0]
n2 = X2.shape[0]

# Uniform distributions on samples
a = np.ones((n1,)) / n1
b = np.ones((n2,)) / n2

# Compute truncated cost matrix to handle potential outliers
truncation_value = 20  # Adjust this value based on your data
M_truncated = truncated_cost_matrix(X1, X2, truncation_value)

# EMD computation with non-truncated cost matrix
M_full = ot.dist(X1, X2, metric='euclidean')
G0 = ot.emd(a, b, M_full)

# Transported source
transp_X1 = G0.T @ X1 / a.reshape((-1, 1))
transp_X1 = mat2im(transp_X1, img2.shape)

# Convert transported image back to matrix for scatter plot
transp_X1_mat = im2mat(transp_X1)

# Sinkhorn transport with truncated cost (regularized OT)
varseps = 0.01
Gs = ot.sinkhorn(a, b, M_truncated, varseps, verbose=False)

# Transported source using Sinkhorn with truncated cost
transp_X1_sinkhorn = Gs.T @ X1 / a.reshape((-1, 1))
transp_X1_sinkhorn = mat2im(transp_X1_sinkhorn, img2.shape)

# Convert EROBOT transported image back to matrix for scatter plot
transp_X1_sinkhorn_mat = im2mat(transp_X1_sinkhorn)

# Figure 1: Original images and color scatter plots
plt.figure(1, figsize=(12, 10))

# Top left: Original source image
plt.subplot(2, 2, 1)
plt.imshow(img1)
plt.axis('off')
plt.title('Source Image')

# Top right: Original target image
plt.subplot(2, 2, 2)
plt.imshow(img2)
plt.axis('off')
plt.title('Target Image')

# Bottom left: Color scatter plot for source image
plt.subplot(2, 2, 3)
plt.scatter(X1[:, 0], X1[:, 2], c=X1)  # Red vs Blue channels of source
plt.axis([0, 1, 0, 1])
plt.xlabel("Red")
plt.ylabel("Blue")
plt.title('Source Image Color Distribution')

# Bottom right: Color scatter plot for target image
plt.subplot(2, 2, 4)
plt.scatter(X2[:, 0], X2[:, 2], c=X2)  # Red vs Blue channels of target
plt.axis([0, 1, 0, 1])
plt.xlabel("Red")
plt.ylabel("Blue")
plt.title('Target Image Color Distribution')

plt.tight_layout()
plt.savefig('original_images.eps', format='eps', dpi=300)  # Save Figure 1 as EPS

plt.show()

# Figure 2: Transported images
plt.figure(2, figsize=(12, 10))

# Left: EMD transported image
plt.subplot(2, 2, 1)
plt.imshow(transp_X1)
plt.axis('off')
plt.title('OT Transported')

# Right: EROBOT transported image
plt.subplot(2, 2, 2)
plt.imshow(transp_X1_sinkhorn)
plt.axis('off')
plt.title('EROBOT Transported')

#plt.tight_layout()
#plt.show()

# Figure 3: Color distributions of transported images
#plt.figure(3, figsize=(12, 5))

# Left: Color scatter plot for EMD transported image
plt.subplot(2, 2, 3)
plt.scatter(transp_X1_mat[:, 0], transp_X1_mat[:, 2], c=transp_X1_mat)
plt.axis([0, 1, 0, 1])
plt.xlabel("Red")
plt.ylabel("Blue")
plt.title('OT: Transferred Color Distribution')

# Right: Color scatter plot for EROBOT transported image
plt.subplot(2, 2, 4)
plt.scatter(transp_X1_sinkhorn_mat[:, 0], transp_X1_sinkhorn_mat[:, 2], c=transp_X1_sinkhorn_mat)
plt.axis([0, 1, 0, 1])
plt.xlabel("Red")
plt.ylabel("Blue")
plt.title('EROBOT: Transferred Color Distribution')

plt.tight_layout()
plt.savefig('transferred_results.eps', format='eps', dpi=300)  # Save Figure 2 as EPS
plt.show()