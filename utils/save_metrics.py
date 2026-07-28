from pathlib import Path
import pandas as pd


def save_metrics(metrics, output_dir):
    """
    Save evaluation metrics to a CSV file.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        [metrics]
    )

    df.to_csv(
        output_dir / "metrics.csv",
        index=False,
    )