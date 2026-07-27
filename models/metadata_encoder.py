# models/metadata_encoder.py

import torch
import torch.nn as nn

from configs.config import Config


class MetadataEncoder(nn.Module):
    """
    Encodes patient metadata into a 256-dimensional embedding.
    """

    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(input_dim, 512),

            nn.ReLU(inplace=True),

            nn.Dropout(Config.DROPOUT),

            nn.Linear(512, Config.METADATA_EMBED_DIM)

        )

    def forward(self, metadata):
        """
        Args:
            metadata: Tensor (B, input_dim)

        Returns:
            embedding: Tensor (B, 256)
        """
        embedding = self.encoder(metadata)
        return embedding