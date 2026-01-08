# Mask-Detector-MobileNetV2 (PyTorch)

# Overview

This project implements an end-to-end face mask classification system using a MobileNetV2 architecture built from scratch in PyTorch. The model classifies images into two classes: with_mask, without_mask. 

The focus of this project is not just achieving high accuracy, but understanding and implementing the full deep learning pipeline, including:
- data preprocessing and augmentation
- dataset splitting and loading
- defining MobileNetV2 building blocks manually
- training and evaluation with proper metrics
- checkpointing best-performing models
- inference on new images

The model is trained from scratch on CPU, making architectural efficiency and optimization central design considerations.

# Key Results

Best validation F1 score: ~0.97
Strong balance between precision and recall

No significant overfitting due to:
- data augmentation
- efficient architecture
- proper checkpoint selection

# Dataset Structure

The dataset is organized using PyTorch’s ImageFolder convention:

data/
├── with_mask/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── without_mask/
    ├── img1.jpg
    ├── img2.jpg
    └── ...

Class labels are assigned alphabetically by ImageFolder:
{'with_mask': 0, 'without_mask': 1}

# Data Preprocessing & Augmentation

Training transforms
- Resize to 160 × 160
- Random horizontal flip
- Color jitter (brightness & contrast)
- ImageNet normalization

Validation / Test transforms
- Resize to 160 × 160
- ImageNet normalization only (no randomness)

Using different transforms for training vs. validation ensures:
- improved generalization during training
- unbiased evaluation during validation/testing
