# models/backbones/efficientnetv2.py

import torch
import torch.nn as nn
import timm

from configs.config import Config


class ImageEncoder(nn.Module):
    """
    Image Encoder (MI)

    Inputs:
        - Dermoscopic image
        - Clinical image

    Output:
        - Unified image feature fI
    """

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0
        )

        self.feature_dim = self.backbone.num_features

        # Combine the two image features
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT)
        )

    def forward(self, dermoscopic_image, clinical_image):

        derm_feature = self.backbone(dermoscopic_image)

        clinical_feature = self.backbone(clinical_image)

        fused = torch.cat(
            [derm_feature, clinical_feature],
            dim=1
        )

        image_feature = self.fusion(fused)

        return image_feature