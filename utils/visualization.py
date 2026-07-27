# utils/visualization.py

import os

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_loss(
    train_losses,
    val_losses,
    save_path="outputs/plots/loss_curve.png"
):
    """
    Plot training and validation loss.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def plot_accuracy(
    train_acc,
    val_acc,
    save_path="outputs/plots/accuracy_curve.png"
):
    """
    Plot training and validation accuracy.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.title("Accuracy Curve")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def plot_confusion_matrix(
    cm,
    class_names,
    save_path="outputs/plots/confusion_matrix.png"
):
    """
    Plot confusion matrix.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        xticks_rotation=45,
        cmap="Blues"
    )

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()