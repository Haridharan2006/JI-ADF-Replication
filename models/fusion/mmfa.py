# models/fusion/mmfa.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMFA(nn.Module):
    """
    Multi-Modal Fusion Attention (MMFA)
    Based on Equations (8)–(11) of the JI-ADF paper.
    """

    def __init__(
        self,
        image_dim=1280,
        metadata_dim=256,
        fused_dim=512,
        num_heads=8,
        head_dim=64
    ):
        super().__init__()

        self.image_dim = image_dim
        self.metadata_dim = metadata_dim
        self.fused_dim = fused_dim

        self.num_heads = num_heads
        self.head_dim = head_dim

        # -------------------------
        # Q Projection
        # -------------------------

        self.q_image = nn.Linear(image_dim, num_heads * head_dim)
        self.q_meta = nn.Linear(metadata_dim, num_heads * head_dim)

        # -------------------------
        # K Projection
        # -------------------------

        self.k_image = nn.Linear(image_dim, num_heads * head_dim)
        self.k_meta = nn.Linear(metadata_dim, num_heads * head_dim)

        # -------------------------
        # V Projection
        # -------------------------

        self.v_image = nn.Linear(image_dim, num_heads * head_dim)
        self.v_meta = nn.Linear(metadata_dim, num_heads * head_dim)

        # -------------------------
        # Output Projection
        # -------------------------

        self.output_projection = nn.Linear(
            2 * num_heads * head_dim,
            fused_dim
        )

        # -------------------------
        # Skip Connection
        # -------------------------

        self.skip_projection = nn.Linear(
            image_dim + metadata_dim,
            fused_dim
        )

        # g(.) from Equation (11)
        self.activation = nn.ReLU()

    def forward(self, image_feature, metadata_feature):

        batch_size = image_feature.size(0)

        # -------------------------
        # Q K V
        # -------------------------

        q_img = self.q_image(image_feature)
        q_meta = self.q_meta(metadata_feature)

        k_img = self.k_image(image_feature)
        k_meta = self.k_meta(metadata_feature)

        v_img = self.v_image(image_feature)
        v_meta = self.v_meta(metadata_feature)

        # -------------------------
        # Reshape
        # (B,H,D)
        # -------------------------

        q_img = q_img.view(batch_size, self.num_heads, self.head_dim)
        q_meta = q_meta.view(batch_size, self.num_heads, self.head_dim)

        k_img = k_img.view(batch_size, self.num_heads, self.head_dim)
        k_meta = k_meta.view(batch_size, self.num_heads, self.head_dim)

        v_img = v_img.view(batch_size, self.num_heads, self.head_dim)
        v_meta = v_meta.view(batch_size, self.num_heads, self.head_dim)

        attention_outputs = []

        # -------------------------
        # Multi-head Attention
        # -------------------------

        for h in range(self.num_heads):

            Q = torch.stack(
                [
                    q_img[:, h, :],
                    q_meta[:, h, :]
                ],
                dim=1
            )

            K = torch.stack(
                [
                    k_img[:, h, :],
                    k_meta[:, h, :]
                ],
                dim=1
            )

            V = torch.stack(
                [
                    v_img[:, h, :],
                    v_meta[:, h, :]
                ],
                dim=1
            )

            scores = torch.matmul(
                Q,
                K.transpose(-2, -1)
            ) / math.sqrt(self.head_dim)

            attention = F.softmax(scores, dim=-1)

            U = torch.matmul(attention, V)

            attention_outputs.append(U.reshape(batch_size, -1))

        # -------------------------
        # Concatenate Heads
        # -------------------------

        attention_outputs = torch.cat(
            attention_outputs,
            dim=1
        )

        # -------------------------
        # Output Projection
        # -------------------------

        attention_output = self.output_projection(
            attention_outputs
        )

        attention_output = self.activation(
            attention_output
        )

        # -------------------------
        # Skip Connection
        # -------------------------

        skip = self.skip_projection(

            torch.cat(
                [
                    image_feature,
                    metadata_feature
                ],
                dim=1
            )

        )

        # -------------------------
        # Final Fusion
        # -------------------------

        fused_feature = skip + attention_output

        return fused_feature