# losses/loss.py

import torch.nn as nn


class JIADFLoss(nn.Module):
    """
    Loss function for JI-ADF.

    Total Loss =
        0.25 * Image Loss +
        0.25 * Metadata Loss +
        0.50 * Joint Loss +
        1.00 * Final Prediction Loss
    """

    def __init__(
        self,
        image_weight=0.25,
        metadata_weight=0.25,
        joint_weight=0.50,
        final_weight=1.00,
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
        labels,
    ):

        # Individual image branch loss
        image_loss = self.ce(
            outputs["image_logits"],
            labels,
        )

        # Individual metadata branch loss
        metadata_loss = self.ce(
            outputs["metadata_logits"],
            labels,
        )

        # Joint multimodal branch loss
        joint_loss = self.ce(
            outputs["joint_logits"],
            labels,
        )

        # Final Adaptive Decision Fusion prediction loss
        prediction_loss = self.ce(
            outputs["prediction_logits"],
            labels,
        )

        # Weighted total loss
        total_loss = (
            self.image_weight * image_loss
            + self.metadata_weight * metadata_loss
            + self.joint_weight * joint_loss
            + self.final_weight * prediction_loss
        )

        losses = {
            "total_loss": total_loss,
            "image_loss": image_loss,
            "metadata_loss": metadata_loss,
            "joint_loss": joint_loss,
            "prediction_loss": prediction_loss,
        }

        return losses