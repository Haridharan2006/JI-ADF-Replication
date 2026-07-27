# trainer/test.py

import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.metrics import compute_metrics
from utils.visualization import plot_confusion_matrix


@torch.no_grad()
def test(
    model,
    test_loader,
    criterion,
    device,
    class_names=None,
    save_predictions=True,
):
    """
    Evaluate the model on a test dataset.

    Supports both:
        1. Labelled datasets
        2. Unlabelled datasets (official MILK10k test set)

    Returns:
        avg_loss
        metrics (None if labels unavailable)
    """

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    has_labels = False

    for batch in tqdm(
        test_loader,
        desc="Testing",
        leave=False,
    ):

        clinical = batch["clinical_image"].to(
            device,
            non_blocking=True,
        )

        dermoscopy = batch["dermoscopy_image"].to(
            device,
            non_blocking=True,
        )

        metadata = batch["metadata"].float().to(
            device,
            non_blocking=True,
        )

        outputs = model(
            dermoscopy,
            clinical,
            metadata,
        )

        probabilities = outputs["prediction_probs"]

        predictions = probabilities.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().tolist()
        )

        # ------------------------------------
        # If labels exist, compute loss/metrics
        # ------------------------------------

        if "label" in batch and batch["label"] is not None:

            has_labels = True

            labels = batch["label"].long().to(
                device,
                non_blocking=True,
            )

            losses = criterion(
                outputs,
                labels,
            )

            running_loss += losses["total_loss"].item()

            all_labels.extend(
                labels.cpu().tolist()
            )

    # ==========================================
    # Save predictions
    # ==========================================

    if save_predictions:

        os.makedirs(
            "outputs/predictions",
            exist_ok=True,
        )

        if has_labels:

            df = pd.DataFrame({

                "True Label": all_labels,

                "Predicted Label": all_predictions,

            })

        else:

            df = pd.DataFrame({

                "Predicted Label": all_predictions,

            })

        df.to_csv(
            "outputs/predictions/test_predictions.csv",
            index=False,
        )

    # ==========================================
    # Unlabelled dataset
    # ==========================================

    if not has_labels:

        print("No ground-truth labels found.")
        print("Predictions saved successfully.")

        return None, None

    # ==========================================
    # Compute metrics
    # ==========================================

    avg_loss = running_loss / max(
        len(test_loader),
        1,
    )

    metrics = compute_metrics(

        np.array(all_labels),

        np.array(all_predictions),

        np.array(all_probabilities),

    )

    # ==========================================
    # Confusion Matrix
    # ==========================================

    if class_names is not None:

        plot_confusion_matrix(

            np.array(metrics["confusion_matrix"]),

            class_names,

        )

    return avg_loss, metrics