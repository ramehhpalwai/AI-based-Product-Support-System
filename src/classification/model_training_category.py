from typing import Any, Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report
from src.classification.feature_builder import BuildClassificationFeatures
from src.data.ingestion import load_tickets

from src.classification.data_processing import (prepare_classification_dataset,
                                                group_text_and_train_val_test_split,
                                                build_feature_encoder_pipeline_tfidf)
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





def train_and_save_category_model(json_data,text_mode="tfidf", save_path="xgb_ticket_model_category.joblib"):
    builder = BuildClassificationFeatures(
        json_data,
        label_mode="category",
        text_mode=text_mode,
    )
    features = builder.build(sample_sizes=None)

    num_class = len(features.label_encoder.classes_)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_class,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=42,
    )

    model.fit(features.X_train, features.y_train)

    # VAL
    val_pred_enc = model.predict(features.X_val)
    val_pred = features.label_encoder.inverse_transform(val_pred_enc)
    val_true = features.label_encoder.inverse_transform(features.y_val)
    print("VAL weighted F1:", round(f1_score(val_true, val_pred, average="weighted"), 4))
    print(classification_report(val_true, val_pred, digits=3))

    # TEST
    test_pred_enc = model.predict(features.X_test)
    test_pred = features.label_encoder.inverse_transform(test_pred_enc)
    test_true = features.label_encoder.inverse_transform(features.y_test)
    print("TEST weighted F1:", round(f1_score(test_true, test_pred, average="weighted"), 4))
    print(classification_report(test_true, test_pred, digits=3))

    if text_mode == 'tfidf':
        bundle = {
            "model": model,
            "label_encoder": features.label_encoder,
            "ohe": features.ohe,
            "embedder": features.embedder,          # fitted TfidfVectorizer
            "cat_cols": features.cat_cols,
            "text_col": features.text_col,
            "text_mode": builder.text_mode,
            "label_mode": builder.label_mode,
            # optional: store tfidf params used
            "tfidf_params": {
                "max_features": builder.tfidf_max_features,
                "ngram_range": builder.tfidf_ngram_range,
                "min_df": builder.tfidf_min_df,
            },
        }
    if text_mode == "transformer":
        bundle = {
            "model": model,
            "label_encoder": features.label_encoder,
            "ohe": features.ohe,

            # DO NOT save SentenceTransformer
            "embedder": None,

            # Save how to recreate it
            "text_mode": "transformer",
            "model_name": builder.model_name,
            "device": builder.device,
            "batch_size": builder.batch_size,
            "normalize_embeddings": builder.normalize_embeddings,

            "cat_cols": features.cat_cols,
            "text_col": features.text_col,
            "label_mode": builder.label_mode,
        }
        joblib.dump(bundle, save_path)
        print(f"Saved to: {save_path}")

    return bundle



json_data = load_tickets("data/raw/support_tickets.json")
train_and_save_category_model(json_data[0][:5000],save_path="trained_models/xgb_ticket_model_category.joblib")