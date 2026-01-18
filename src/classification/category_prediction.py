from typing import Any, Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from src.classification.data_processing import (prepare_classification_dataset,
                                                group_text_and_train_val_test_split,
                                                build_feature_encoder_pipeline_tfidf)

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report


class TrainLogisticRegression:
    def __init__(self, tickets_data):
        self.tickets_data = tickets_data

        # Will be set after training
        self.preprocessor = None
        self.model = None

    def train(self):
        # 1) Prepare dataset
        classification_data = prepare_classification_dataset(self.tickets_data)
        classification_data = pd.DataFrame(classification_data)

        train_df, val_df, test_df = group_text_and_train_val_test_split(classification_data)

        label_cols = ["subcategory", "category"]

        # 2) Split X/y
        X_train = train_df.drop(columns=label_cols, errors="ignore")
        X_val   = val_df.drop(columns=label_cols, errors="ignore")
        X_test  = test_df.drop(columns=label_cols, errors="ignore")

        # If you're training only category (as you do below):
        y_train = train_df["category"]
        y_val   = val_df["category"]
        y_test  = test_df["category"]

        cat_cols = ["product", "product_module", "priority", "channel", "customer_tier"]
        text_col = "text"

        # 3) Fit/transform features
        self.preprocessor = build_feature_encoder_pipeline_tfidf(
            cat_cols=cat_cols,
            text_col=text_col
        )

        X_train_t = self.preprocessor.fit_transform(X_train)
        X_val_t   = self.preprocessor.transform(X_val)
        X_test_t  = self.preprocessor.transform(X_test)

        # 4) Train model
        self.model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=None
        )
        self.model.fit(X_train_t, y_train)

        # 5) Evaluate
        val_pred = self.model.predict(X_val_t)
        val_f1 = f1_score(y_val, val_pred, average="weighted")
        print("\n--- VALIDATION ---")
        print("Weighted F1:", round(val_f1, 4))
        print(classification_report(y_val, val_pred, digits=3))

        test_pred = self.model.predict(X_test_t)
        test_f1 = f1_score(y_test, test_pred, average="weighted")
        print("\n--- TEST ---")
        print("Weighted F1:", round(test_f1, 4))
        print(classification_report(y_test, test_pred, digits=3))

        return self  # allows chaining

    def predict(self, df: pd.DataFrame):
        if self.preprocessor is None or self.model is None:
            raise RuntimeError("Model not trained. Call .train() first.")

        X_t = self.preprocessor.transform(df)
        return self.model.predict(X_t)

