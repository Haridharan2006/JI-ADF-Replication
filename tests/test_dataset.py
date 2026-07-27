from datasets.milk10k_dataset import MILK10kDataset
from datasets.transforms import get_train_transforms
from datasets.metadata_processor import MetadataProcessor
from configs.paths import *

import pandas as pd

metadata_df = pd.read_csv(TRAIN_METADATA)

metadata_df = (
    metadata_df
    .drop_duplicates(subset=["lesion_id"])
    .drop(columns=["lesion_id", "isic_id", "image_type"], errors="ignore")
)

processor = MetadataProcessor()
processor.fit(metadata_df)

dataset = MILK10kDataset(
    image_dir=TRAIN_IMAGE_DIR,
    metadata_csv=TRAIN_METADATA,
    groundtruth_csv=TRAIN_GROUND_TRUTH,
    metadata_processor=processor,
    transform=get_train_transforms()
)

sample = dataset[0]

print("Lesion ID:", sample["lesion_id"])
print("Clinical:", sample["clinical_image"].shape)
print("Dermoscopy:", sample["dermoscopy_image"].shape)
print("Metadata:", sample["metadata"].shape)
print("Label:", sample["label"])