import numpy as np
import torch
from tqdm import tqdm

from utils.metrics import compute_metrics


@torch.no_grad()
def validate(
    model,
    val_loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    for batch in tqdm(
        val_loader,
        desc="Validation",
        leave=False,
    ):

        clinical = batch["clinical_image"].to(
            device, non_blocking=True
        )

        dermoscopy = batch["dermoscopy_image"].to(
            device, non_blocking=True
        )

        metadata = batch["metadata"].float().to(
            device, non_blocking=True
        )

        labels = batch["label"].long().to(
            device, non_blocking=True
        )

        outputs = model(
            dermoscopy,
            clinical,
            metadata,
        )

        losses = criterion(outputs, labels)

        running_loss += losses["total_loss"].item()

        probabilities = outputs["prediction_probs"]
        predictions = probabilities.argmax(dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

    avg_loss = running_loss / max(len(val_loader), 1)

    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities),
    )

    return avg_loss, metrics