import torch


class Config:

    # Dataset
    IMAGE_SIZE = 384
    NUM_CLASSES = 11

    # Metadata
    METADATA_EMBED_DIM = 256

    # Training
    BATCH_SIZE = 8
    EPOCHS = 50            # Changed from 1 to 50

    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Model
    BACKBONE = "tf_efficientnetv2_s"
    PRETRAINED = True
    DROPOUT = 0.3

    # Misc
    RANDOM_SEED = 42

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"