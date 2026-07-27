# datasets/metadata_processor.py

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class MetadataProcessor:
    """
    Preprocesses MILK10k metadata.

    - Label encodes categorical columns
    - Standardizes numerical columns
    """

    def __init__(self):

        self.label_encoders = {}
        self.scaler = StandardScaler()

        self.categorical_columns = []
        self.numerical_columns = []

    def fit(self, dataframe):

        df = dataframe.copy()

        # Automatically detect categorical and numerical columns
        for column in df.columns:

            if df[column].dtype == object:

                self.categorical_columns.append(column)

            else:

                self.numerical_columns.append(column)

        # -------------------------
        # Fit Label Encoders
        # -------------------------

        for column in self.categorical_columns:

            encoder = LabelEncoder()

            values = (
                df[column]
                .fillna("Unknown")
                .astype(str)
            )

            encoder.fit(values)

            self.label_encoders[column] = encoder

        # -------------------------
        # Fit StandardScaler
        # -------------------------

        if len(self.numerical_columns) > 0:

            numeric_df = (
                df[self.numerical_columns]
                .fillna(0)
                .astype(float)
            )

            self.scaler.fit(numeric_df)

    def transform(self, dataframe):

        if isinstance(dataframe, np.ndarray):

            dataframe = pd.DataFrame(
                dataframe,
                columns=self.categorical_columns +
                        self.numerical_columns
            )

        df = dataframe.copy()

        # -------------------------
        # Encode categoricals
        # -------------------------

        encoded = []

        for column in self.categorical_columns:

            encoder = self.label_encoders[column]

            values = (
                df[column]
                .fillna("Unknown")
                .astype(str)
            )

            # Handle unseen categories
            values = values.where(
                values.isin(encoder.classes_),
                "Unknown"
            )

            if "Unknown" not in encoder.classes_:

                encoder.classes_ = np.append(
                    encoder.classes_,
                    "Unknown"
                )

            encoded.append(
                encoder.transform(values)
            )

        # -------------------------
        # Scale numerical columns
        # -------------------------

        if len(self.numerical_columns) > 0:

            numeric = (
                df[self.numerical_columns]
                .fillna(0)
                .astype(float)
            )

            numeric = self.scaler.transform(numeric)

        else:

            numeric = np.empty((len(df), 0))

        # -------------------------
        # Combine
        # -------------------------

        if len(encoded) > 0:

            categorical = np.stack(
                encoded,
                axis=1
            )

            features = np.concatenate(
                [categorical, numeric],
                axis=1
            )

        else:

            features = numeric

        return features

    def fit_transform(self, dataframe):

        self.fit(dataframe)

        return self.transform(dataframe)