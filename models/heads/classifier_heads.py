# models/heads/classifier_heads.py

import torch.nn as nn

from configs.config import Config


class ClassifierHeads(nn.Module):
    """
    Three classifier heads used in JI-ADF.

    Outputs raw logits only.
    Softmax is applied later inside ADF and Loss.
    """

    def __init__(
        self,
        image_dim=1280,
        metadata_dim=256,
        joint_dim=512
    ):
        super().__init__()

        self.image_head = nn.Linear(
            image_dim,
            Config.NUM_CLASSES
        )

        self.metadata_head = nn.Linear(
            metadata_dim,
            Config.NUM_CLASSES
        )

        self.joint_head = nn.Linear(
            joint_dim,
            Config.NUM_CLASSES
        )

    def forward(
        self,
        image_feature,
        metadata_feature,
        joint_feature
    ):

        image_logits = self.image_head(
            image_feature
        )

        metadata_logits = self.metadata_head(
            metadata_feature
        )

        joint_logits = self.joint_head(
            joint_feature
        )

        return (
            image_logits,
            metadata_logits,
            joint_logits
        )