# datasets/milk10k_dataset.py

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MILK10kDataset(Dataset):
    """
    MILK10k Dataset

    Returns:
    {
        "clinical_image": Tensor,
        "dermoscopy_image": Tensor,
        "metadata": Tensor,
        "label": int,
        "lesion_id": str
    }
    """

    def __init__(
        self,
        image_dir,
        metadata_csv,
        groundtruth_csv=None,
        metadata_processor=None,
        transform=None,
    ):

        self.image_dir = Path(image_dir)
        self.transform = transform
        self.metadata_processor = metadata_processor

        # -------------------------
        # Read metadata
        # -------------------------

        self.metadata = pd.read_csv(metadata_csv)
        # -------------------------
        # Fast metadata lookup
        # -------------------------

        self.metadata_lookup = {
            lesion_id: group
            for lesion_id, group in self.metadata.groupby("lesion_id")
        }
        # Training dataset?
        self.is_train = groundtruth_csv is not None

        if self.is_train:

            self.labels = pd.read_csv(groundtruth_csv)

            # Convert one-hot encoding -> class index
            class_columns = self.labels.columns[1:]

            self.labels["label"] = (
                self.labels[class_columns]
                .values
                .argmax(axis=1)
            )

            self.samples = self.labels.copy()
            self.labels_array = self.samples["label"].to_numpy()

        else:

            self.samples = (
                self.metadata[["lesion_id"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        # -------------------------
        # Lesion ID
        # -------------------------

        lesion_id = self.samples.iloc[idx]["lesion_id"]

        # -------------------------
        # Metadata rows
        # -------------------------

        lesion_rows = self.metadata_lookup[lesion_id]

        if len(lesion_rows) != 2:
            raise ValueError(
                f"{lesion_id} has {len(lesion_rows)} metadata rows "
                "instead of 2."
            )

        # -------------------------
        # Clinical row
        # -------------------------

        clinical_row = lesion_rows[
            lesion_rows["image_type"]
            .str.contains(
                "clinical",
                case=False,
                na=False
            )
        ]

        if len(clinical_row) == 0:
            raise ValueError(
                f"Clinical image not found for {lesion_id}"
            )

        clinical_row = clinical_row.iloc[0]

        # -------------------------
        # Dermoscopy row
        # -------------------------

        derm_row = lesion_rows[
            lesion_rows["image_type"]
            .str.contains(
                "dermo",
                case=False,
                na=False
            )
        ]

        if len(derm_row) == 0:
            raise ValueError(
                f"Dermoscopic image not found for {lesion_id}"
            )

        derm_row = derm_row.iloc[0]

        # -------------------------
        # Image Paths
        # -------------------------

        clinical_path = (
            self.image_dir
            / lesion_id
            / f"{clinical_row['isic_id']}.jpg"
        )

        derm_path = (
            self.image_dir
            / lesion_id
            / f"{derm_row['isic_id']}.jpg"
        )

        if not clinical_path.exists():
            raise FileNotFoundError(clinical_path)

        if not derm_path.exists():
            raise FileNotFoundError(derm_path)

        # -------------------------
        # Read Images
        # -------------------------

        with Image.open(clinical_path) as img:
            clinical_image = img.convert("RGB")

        with Image.open(derm_path) as img:
            dermoscopy_image = img.convert("RGB")

        # -------------------------
        # Image Transform
        # -------------------------

        if self.transform is not None:

            clinical_image = self.transform(
                clinical_image
            )

            dermoscopy_image = self.transform(
                dermoscopy_image
            )

        # -------------------------
        # Metadata
        # -------------------------

        metadata = clinical_row.drop(
            labels=[
                "lesion_id",
                "isic_id",
                "image_type",
            ],
            errors="ignore",
        ).to_frame().T

        if self.metadata_processor is not None:
            metadata = self.metadata_processor.transform(metadata)

        metadata = torch.tensor(
            metadata.squeeze(),
            dtype=torch.float32,
        )

        # -------------------------
        # Label
        # -------------------------

        if self.is_train:

            label = int(self.labels_array[idx])

        else:

            label = -1

        # -------------------------
        # Return
        # -------------------------

        return {

            "clinical_image": clinical_image,

            "dermoscopy_image": dermoscopy_image,

            "metadata": metadata,

            "label": label,

            "lesion_id": lesion_id

        }