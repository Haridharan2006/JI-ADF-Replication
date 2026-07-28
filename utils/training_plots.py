from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_history(
    history_csv,
    output_dir,
):
    """
    Generate training curves from the training_history.csv file.
    """

    history = pd.read_csv(history_csv)

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------
    # Loss Curve
    # -------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Train Loss",
        linewidth=2,
    )

    plt.plot(
        history["epoch"],
        history["validation_loss"],
        label="Validation Loss",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "loss_curve.png",
        dpi=300,
    )

    plt.close()

    # -------------------------
    # Accuracy Curve
    # -------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["epoch"],
        history["train_accuracy"],
        label="Train Accuracy",
        linewidth=2,
    )

    plt.plot(
        history["epoch"],
        history["accuracy"],
        label="Validation Accuracy",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "accuracy_curve.png",
        dpi=300,
    )

    plt.close()