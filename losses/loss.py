# losses/loss.py

import torch
import torch.nn as nn


class JIADFLoss(nn.Module):
    """
    Loss function for JI-ADF.

    Total Loss =
        λ1 * Image Loss +
        λ2 * Metadata Loss +
        λ3 * Joint Loss +
        λ4 * Final Prediction Loss
    """

    def __init__(
        self,
        image_weight=1.0,
        metadata_weight=1.0,
        joint_weight=1.0,
        final_weight=1.0
    ):
        super().__init__()

        self.image_weight = image_weight
        self.metadata_weight = metadata_weight
        self.joint_weight = joint_weight
        self.final_weight = final_weight

        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs,
        labels
    ):

        image_loss = self.ce(
            outputs["image_logits"],
            labels
        )

        metadata_loss = self.ce(
            outputs["metadata_logits"],
            labels
        )

        joint_loss = self.ce(
            outputs["joint_logits"],
            labels
        )

        # prediction is probability from ADF
        # Convert to log-probabilities for NLLLoss
        prediction_loss = self.ce(
            outputs["prediction_logits"],
            labels
        )

        total_loss = (

            self.image_weight * image_loss +

            self.metadata_weight * metadata_loss +

            self.joint_weight * joint_loss +

            self.final_weight * prediction_loss

        )

        losses = {

            "total_loss": total_loss,

            "image_loss": image_loss,

            "metadata_loss": metadata_loss,

            "joint_loss": joint_loss,

            "prediction_loss": prediction_loss

        }

        return losses