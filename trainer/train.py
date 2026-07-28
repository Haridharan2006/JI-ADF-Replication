import torch
from tqdm import tqdm


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    scaler=None,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(
        train_loader,
        desc="Training",
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

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:

            with torch.cuda.amp.autocast():

                outputs = model(
                    dermoscopy,
                    clinical,
                    metadata,
                )

                losses = criterion(outputs, labels)
                loss = losses["total_loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:

            outputs = model(
                dermoscopy,
                clinical,
                metadata,
            )

            losses = criterion(outputs, labels)
            loss = losses["total_loss"]

            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        prediction = outputs["prediction_probs"].argmax(dim=1)

        correct += (prediction == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / max(len(train_loader), 1)
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    return avg_loss, accuracy