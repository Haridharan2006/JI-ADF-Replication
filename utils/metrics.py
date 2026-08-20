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

    # ------------------------------------------------
    # Accuracy
    # ------------------------------------------------

    metrics["accuracy"] = accuracy_score(
        y_true,
        y_pred,
    )

    # ------------------------------------------------
    # Balanced Accuracy
    # ------------------------------------------------

    metrics["balanced_accuracy"] = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    # ------------------------------------------------
    # Macro Precision
    # ------------------------------------------------

    metrics["precision"] = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # ------------------------------------------------
    # Macro Sensitivity / Recall
    # ------------------------------------------------

    metrics["recall"] = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # ------------------------------------------------
    # Macro F1
    # ------------------------------------------------

    metrics["f1_score"] = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # ------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------

    metrics["confusion_matrix"] = (
        confusion_matrix(
            y_true,
            y_pred,
        ).tolist()
    )

    # ------------------------------------------------
    # Macro AUC
    # ------------------------------------------------

    if y_prob is not None:

        try:

            metrics["roc_auc"] = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
            )

        except ValueError:

            metrics["roc_auc"] = None

    else:

        metrics["roc_auc"] = None

    return metrics