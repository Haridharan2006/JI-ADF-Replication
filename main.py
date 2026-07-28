import os
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from configs.config import Config
from configs.paths import *

from datasets.transforms import (
    get_train_transforms,
    get_val_transforms,
)
from datasets.metadata_processor import MetadataProcessor
from datasets.milk10k_dataset import MILK10kDataset

from models.ji_adf import JIADFModel

from losses.loss import JIADFLoss

from trainer.train import train_one_epoch
from trainer.validate import validate
from trainer.test import test

from utils.seed import set_seed
from utils.logger import get_logger
from utils.checkpoint import save_checkpoint, load_checkpoint


def main():

    ####################################################
    # Seed & Device
    ####################################################
    set_seed(Config.RANDOM_SEED)

    logger = get_logger()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info(f"Using device: {device}")

    ####################################################
    # Metadata Processor
    ####################################################
    metadata_df = pd.read_csv(TRAIN_METADATA)

    metadata_df = (
        metadata_df
        .drop_duplicates(subset=["lesion_id"])
        .drop(
            columns=[
                "lesion_id",
                "isic_id",
                "image_type",
            ],
            errors="ignore",
        )
    )

    metadata_processor = MetadataProcessor()
    metadata_processor.fit(metadata_df)

    ####################################################
    # Full Training Dataset
    ####################################################
    full_dataset = MILK10kDataset(
        image_dir=TRAIN_IMAGE_DIR,
        metadata_csv=TRAIN_METADATA,
        groundtruth_csv=TRAIN_GROUND_TRUTH,
        metadata_processor=metadata_processor,
        transform=get_train_transforms(),
    )

    ####################################################
    # Train / Validation Split
    ####################################################
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(
        Config.RANDOM_SEED
    )

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,
    )

    ####################################################
    # Validation Transform (optional)
    ####################################################
    # If you want different transforms for validation,
    # create another dataset using the validation transform.
    # For now, we keep the split simple.

    ####################################################
    # DataLoaders
    ####################################################
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    ####################################################
    # Test Dataset
    ####################################################
    test_dataset = MILK10kDataset(
        image_dir=TEST_IMAGE_DIR,
        metadata_csv=TEST_METADATA,
        groundtruth_csv=None,
        metadata_processor=metadata_processor,
        transform=get_val_transforms(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    ####################################################
    # Dynamic Metadata Dimension
    ####################################################
    sample = full_dataset[0]
    metadata_dim = sample["metadata"].shape[0]

    logger.info(f"Metadata Dimension: {metadata_dim}")

    ####################################################
    # Model
    ####################################################
    model = JIADFModel(
        metadata_input_dim=metadata_dim
    ).to(device)

    ####################################################
    # Loss
    ####################################################
    criterion = JIADFLoss()

    ####################################################
    # Optimizer
    ####################################################
    optimizer = AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    ####################################################
    # Scheduler
    ####################################################
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    ####################################################
    # Training
    ####################################################
    best_f1 = 0.0

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(Config.EPOCHS):

        logger.info(
            f"\nEpoch {epoch + 1}/{Config.EPOCHS}"
        )

        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        logger.info(f"Train Loss : {train_loss:.4f}")
        logger.info(f"Train Accuracy : {train_acc:.4f}")

        logger.info(f"Validation Loss : {val_loss:.4f}")
        logger.info(metrics)

        scheduler.step(metrics["f1_score"])

        if metrics["f1_score"] > best_f1:

            best_f1 = metrics["f1_score"]

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=best_f1,
                filepath=CHECKPOINT_DIR / "best_model.pth",
            )

            logger.info("Best model saved.")

    ####################################################
    # Load Best Model
    ####################################################
    model, optimizer, epoch, best_metric = load_checkpoint(
        model=model,
        optimizer=optimizer,
        filepath=CHECKPOINT_DIR / "best_model.pth",
        device=device,
    )

    logger.info(
        f"Loaded best model (Epoch {epoch + 1}, F1 = {best_metric:.4f})"
    )

    ####################################################
    # Test
    ####################################################
    RUN_TEST = False

    if RUN_TEST:
        logger.info("Testing...")

        test_loss, test_metrics = test(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
        )

    logger.info(test_metrics)

    logger.info(test_metrics)


if __name__ == "__main__":
    main()