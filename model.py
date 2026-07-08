"""
model.py

A small CNN classifier over MFCC features, trained from scratch (no
pretrained weights, no transfer learning). This is intentionally a small,
from-scratch architecture appropriate to a small synthetic dataset -- not
a claim of foundation-model-scale training.
"""

import torch
import torch.nn as nn


class SpeechCommandCNN(nn.Module):
    def __init__(self, n_mfcc=40, max_frames=48, num_classes=14):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (16, n_mfcc/2, max_frames/2)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (32, n_mfcc/4, max_frames/4)
        )
        flat_dim = 32 * (n_mfcc // 4) * (max_frames // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.classifier(x)
