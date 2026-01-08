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

# Model Architecture: MobileNetV2

This project implements MobileNetV2 manually instead of importing it from torchvision, with the goal of deeply understanding its internal design and architectural choices.

Why MobileNetV2?
- Designed for efficiency on mobile and CPU devices
- Uses depthwise separable convolutions to reduce computation
- Employs inverted residual blocks with linear bottlenecks
- Provides an excellent speed–accuracy tradeoff for real-time inference

# Core Building Blocks

1. Conv–BN–ReLU6 Block

A lightweight building unit used throughout the network:

Conv2D → BatchNorm → ReLU6

ReLU6 is used instead of standard ReLU for improved numerical stability, especially in low-precision and mobile-oriented settings.

2. Inverted Residual (Bottleneck) Block

Each MobileNetV2 bottleneck follows this structure:

1×1 Expansion Conv
↓
3×3 Depthwise Conv
↓
1×1 Linear Projection Conv
↓
(Residual connection if shape allows)

Key ideas behind this design:
- Expansion temporarily increases channel capacity to learn richer features
- Depthwise convolution performs spatial filtering efficiently, one channel at a time
- Linear bottleneck avoids information loss caused by non-linear activations in low-dimensional spaces
- Residual connections are used only when stride = 1 and input/output channels match

# Classification Head

Instead of large fully connected layers, MobileNetV2 uses:
- A 1×1 convolution to expand channel depth
- Global Average Pooling to collapse spatial dimensions
- A single linear layer to produce class logits

This design significantly reduces the number of parameters while improving generalization.

# Training Details

Framework: PyTorch
Loss Function: CrossEntropyLoss (raw logits)
Optimizer: AdamW
Metrics: Accuracy, Precision, Recall, F1 Score
Checkpointing: Best model saved based on validation F1
Device: CPU-only training

The model is trained from scratch and evaluated using validation and test splits.

# Inference Pipeline

After training, the model can be used for inference by following these steps:
- Reconstruct the model architecture
- Load the saved state_dict from the checkpoint
- Apply the same preprocessing used during validation
- Run a forward pass through the model
- Apply softmax to obtain class probabilities

# Citation

If you reference or build upon this work, please cite the original MobileNetV2 paper:
@article{sandler2018mobilenetv2,
  title={MobileNetV2: Inverted Residuals and Linear Bottlenecks},
  author={Sandler, Mark and Howard, Andrew and Zhu, Menglong and Zhmoginov, Andrey and Chen, Liang-Chieh},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}



