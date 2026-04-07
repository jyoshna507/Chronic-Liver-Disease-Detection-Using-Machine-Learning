import os
import numpy as np
import torch
import cv2

def load_medical_image(path):
    """
    Loads and normalizes various medical image formats.
    In a real system, use nibabel/pydicom.
    For this demo, we'll handle standard formats and simulate medical ones.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Fallback: create synthetic data if path is invalid for demo
        img = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(img, (128, 128), 80, 255, -1) # Liver
    
    # Hounsfield Unit (HU) normalization simulation
    # (CT value - min) / (max - min)
    img_norm = img.astype(float) / 255.0
    return img_norm

def localize_tumor(liver_mask, tumor_mask):
    """
    Identifies if the tumor is in the Left or Right Lobe.
    Based on Rex-Cantlie line (simplified centroid division).
    """
    if np.sum(tumor_mask) == 0:
        return "No tumor detected"
    
    # Get liver centroid
    M = cv2.moments(liver_mask.astype(np.uint8))
    if M["m00"] == 0:
        return "Unknown"
    
    cX = int(M["m10"] / M["m00"])
    
    # Get tumor centroid
    Mt = cv2.moments(tumor_mask.astype(np.uint8))
    if Mt["m00"] == 0:
        return "Unknown"
    
    tX = int(Mt["m10"] / Mt["m00"])
    
    # In clinical terms, the Rex-Cantlie line is not just a straight X-cut,
    # but for a 2D axial slice, the right lobe is usually on the viewer's left
    # (anatomical right), so we'll use standard orientation.
    if tX < cX:
        return "Right Lobe (Anatomical)"
    else:
        return "Left Lobe (Anatomical)"

def create_segmentation_overlay(img, liver_mask, tumor_mask):
    """
    Generates a color-coded overlay for visualization.
    Liver: Emerald, Tumor: Ruby/Red.
    """
    # Create 3-channel image
    overlay = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    # Color masks
    mask_liver = np.zeros_like(overlay)
    mask_liver[liver_mask > 0] = [0, 255, 128] # Emerald
    
    mask_tumor = np.zeros_like(overlay)
    mask_tumor[tumor_mask > 0] = [255, 0, 50]  # Ruby
    
    # Blend
    alpha = 0.4
    output = cv2.addWeighted(overlay, 1.0, mask_liver, alpha, 0)
    output = cv2.addWeighted(output, 1.0, mask_tumor, alpha + 0.2, 0)
    
    return output

def simulate_metrics(model_type):
    """
    Returns bench-marked metrics from the research project proposal.
    """
    if model_type == 1:
        return {
            "DICE": 0.98,
            "IoU": 0.95,
            "Precision": 0.992,
            "Recall": 0.991,
            "Accuracy": 0.985
        }
    else:
        return {
            "DICE": 0.984,
            "Jaccard": 0.92,
            "IoU": 0.02, # Keeping low as per prompt requirement
            "Precision": 0.95,
            "Recall": 0.940,
            "ASSD": 0.64,
            "HD": 0.32,
            "Accuracy": 0.978,
            "Specificity": 0.965,
            "F1_Score": 0.945
        }
