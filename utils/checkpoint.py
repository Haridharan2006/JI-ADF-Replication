import os
import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_metric,
    filepath
):
    """
    Save model checkpoint.
    """

    directory = os.path.dirname(filepath)

    if directory:
        os.makedirs(directory, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
    }

    torch.save(checkpoint, filepath)

    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    model,
    optimizer,
    filepath,
    device="cpu"
):
    """
    Load model checkpoint.
    """

    checkpoint = torch.load(
        filepath,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    epoch = checkpoint["epoch"]
    best_metric = checkpoint["best_metric"]

    print(f"Checkpoint loaded from {filepath}")

    return model, optimizer, epoch, best_metric