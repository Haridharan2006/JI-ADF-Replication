# models/adf.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config


class AdaptiveDecisionFusion(nn.Module):
    """
    Adaptive Decision Fusion (ADF)

    Implements Equations (5) and (6) from the paper.
    """

    def __init__(self):
        super().__init__()

        self.gating_network = nn.Linear(
            Config.NUM_CLASSES * 3,
            3
        )

    def forward(
        self,
        image_logits,
        metadata_logits,
        joint_logits
    ):

        # ---------------------------------
        # Concatenate logits
        # Shape: (B, 33)
        # ---------------------------------

        concat_logits = torch.cat(
            [
                image_logits,
                joint_logits,
                metadata_logits
            ],
            dim=1
        )

        # ---------------------------------
        # Adaptive weights
        # Shape: (B,3)
        # ---------------------------------

        weights = self.gating_network(
            concat_logits
        )

        weights = F.softmax(
            weights,
            dim=1
        )

        # ---------------------------------
        # Convert logits to probabilities
        # ---------------------------------

        image_prob = F.softmax(image_logits, dim=1)
        metadata_prob = F.softmax(metadata_logits, dim=1)
        joint_prob = F.softmax(joint_logits, dim=1)

        # ---------------------------------
        # Weighted logits
        # ---------------------------------

        final_logits = (
            weights[:, 0:1] * image_logits +
            weights[:, 1:2] * joint_logits +
            weights[:, 2:3] * metadata_logits
        )

        # ---------------------------------
        # Final probabilities
        # ---------------------------------

        final_prob = F.softmax(final_logits, dim=1)

        return (
            final_logits,
            final_prob,
            weights
        )