# models/ji_adf.py

import torch.nn as nn

from models.backbones.efficientnetv2 import ImageEncoder
from models.metadata_encoder import MetadataEncoder
from models.fusion.mmfa import MMFA
from models.heads.classifier_heads import ClassifierHeads
from models.adf import AdaptiveDecisionFusion


class JIADFModel(nn.Module):
    """
    Complete JI-ADF Network
    """

    def __init__(
        self,
        metadata_input_dim,
        image_dim=1280,
        metadata_dim=256,
        joint_dim=512
    ):
        super().__init__()

        # Image encoder
        self.image_encoder = ImageEncoder()

        # Metadata encoder
        self.metadata_encoder = MetadataEncoder(
            input_dim=metadata_input_dim
        )

        # MMFA
        self.mmfa = MMFA(
            image_dim=image_dim,
            metadata_dim=metadata_dim,
            fused_dim=joint_dim
        )

        # Three classifier heads
        self.classifier_heads = ClassifierHeads(
            image_dim=image_dim,
            metadata_dim=metadata_dim,
            joint_dim=joint_dim
        )

        # Adaptive Decision Fusion
        self.adf = AdaptiveDecisionFusion()

    def forward(
        self,
        dermoscopic_image,
        clinical_image,
        metadata
    ):

        # -----------------------------
        # Feature Extraction
        # -----------------------------

        image_feature = self.image_encoder(
            dermoscopic_image,
            clinical_image
        )

        metadata_feature = self.metadata_encoder(
            metadata
        )

        # -----------------------------
        # Multi-modal Fusion
        # -----------------------------

        joint_feature = self.mmfa(
            image_feature,
            metadata_feature
        )

        # -----------------------------
        # Classifier Heads
        # -----------------------------

        (
            image_logits,
            metadata_logits,
            joint_logits
        ) = self.classifier_heads(
            image_feature,
            metadata_feature,
            joint_feature
        )

        # -----------------------------
        # Adaptive Decision Fusion
        # -----------------------------

        prediction_logits, prediction_probs, weights = self.adf(
            image_logits,
            metadata_logits,
            joint_logits
        )

        return {

            "image_feature": image_feature,
            "metadata_feature": metadata_feature,
            "joint_feature": joint_feature,

            "image_logits": image_logits,
            "metadata_logits": metadata_logits,
            "joint_logits": joint_logits,

            "prediction_logits": prediction_logits,
            "prediction_probs": prediction_probs,

            "weights": weights

        }