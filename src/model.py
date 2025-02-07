# -*- coding: utf-8 -*-
import torch
from torchvision import models
import torch.nn as nn


class VideoResNet(nn.Module):
    def __init__(self, base_model, num_classes):
        super(VideoResNet, self).__init__()
        self.base_model = base_model
        # Remove the final fully connected layer
        self.base_model.fc = nn.Identity()

        # Add temporal processing layers
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # x shape: [batch, frames, channels, height, width]
        b, f, c, h, w = x.shape

        # Reshape input to process all frames
        x = x.reshape(b * f, c, h, w)

        # Process through base ResNet (except final fc)
        x = self.base_model(x)

        # Reshape back to include temporal dimension
        x = x.reshape(b, f, -1)  # [batch, frames, features]

        # Average across temporal dimension
        x = torch.mean(x, dim=1)  # [batch, features]

        # Final classification
        x = self.fc(x)

        return x


def build_model(fine_tune=True, num_classes=10):
    # Load base ResNet model
    base_model = models.resnet50(weights="DEFAULT")

    if fine_tune:
        print("[INFO]: Fine-tuning all layers...")
        for params in base_model.parameters():
            params.requires_grad = True
    else:
        print("[INFO]: Freezing hidden layers...")
        for params in base_model.parameters():
            params.requires_grad = False

    # Create video model
    model = VideoResNet(base_model, num_classes)

    return model
