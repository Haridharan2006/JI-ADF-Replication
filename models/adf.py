# models/adf.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config


class AdaptiveDecisionFusion(nn.Module):
    """
    Adaptive Decision Fusion (ADF)

    Implements Equations (5) and (6)
    from the JI-ADF paper.

    The gating network receives the concatenated
    logits from the image, joint, and metadata heads
    and produces sample-specific fusion weights.

    Final prediction is a convex combination of
    the three branch posterior probabilities.
    """

    def __init__(
        self,
        hidden_dim=128,
    ):
        super().__init__()

        input_dim = Config.NUM_CLASSES * 3

        # Equation (5):
        #
        # alpha = softmax(
        #     W2 * sigma(W1*s + b1) + b2
        # )
        #
        self.gating_network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim,
                3,
            ),
        )

    def forward(
        self,
        image_logits,
        metadata_logits,
        joint_logits,
    ):

        # -----------------------------------------
        # Concatenate branch logits
        # s = [zI || zIM || zM]
        # -----------------------------------------

        concat_logits = torch.cat(
            [
                image_logits,
                joint_logits,
                metadata_logits,
            ],
            dim=1,
        )

        # -----------------------------------------
        # Adaptive fusion weights
        # alpha = softmax(
        #     W2 sigma(W1s + b1) + b2
        # )
        # -----------------------------------------

        weights = self.gating_network(
            concat_logits
        )

        weights = F.softmax(
            weights,
            dim=1,
        )

        # -----------------------------------------
        # Convert branch logits to posteriors
        # -----------------------------------------

        image_prob = F.softmax(
            image_logits,
            dim=1,
        )

        joint_prob = F.softmax(
            joint_logits,
            dim=1,
        )

        metadata_prob = F.softmax(
            metadata_logits,
            dim=1,
        )

        # -----------------------------------------
        # Equation (6)
        #
        # Pfinal =
        #   alpha_I  * P_I
        # + alpha_IM * P_IM
        # + alpha_M  * P_M
        # -----------------------------------------

        final_prob = (
            weights[:, 0:1] * image_prob
            + weights[:, 1:2] * joint_prob
            + weights[:, 2:3] * metadata_prob
        )

        # -----------------------------------------
        # Convert final probability to logits
        #
        # This is needed because CrossEntropyLoss
        # expects logits.
        # -----------------------------------------

        final_logits = torch.log(
            final_prob.clamp(min=1e-8)
        )

        return (
            final_logits,
            final_prob,
            weights,
        )