# configs/paths.py

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "MILK10K"

TRAIN_IMAGE_DIR = DATASET_DIR / "MILK10K_Training_Input"
TEST_IMAGE_DIR = DATASET_DIR / "MILK10K_Test_Input"

TRAIN_METADATA = DATASET_DIR / "MILK10k_Training_Metadata.csv"
TEST_METADATA = DATASET_DIR / "MILK10k_Test_Metadata.csv"

TRAIN_GROUND_TRUTH = DATASET_DIR / "MILK10k_Training_GroundTruth.csv"

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
OUTPUT_DIR = ROOT_DIR / "outputs"
LOG_DIR = ROOT_DIR / "logs"