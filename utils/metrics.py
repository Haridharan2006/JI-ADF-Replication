from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_prob=None):

    metrics = {}

    metrics["accuracy"] = accuracy_score(
        y_true,
        y_pred,
    )

    metrics["balanced_accuracy"] = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    metrics["precision"] = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics["recall"] = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics["f1_score"] = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics["confusion_matrix"] = (
        confusion_matrix(
            y_true,
            y_pred,
        ).tolist()
    )

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="weighted",
            )
        except ValueError:
            metrics["roc_auc"] = None

    return metrics