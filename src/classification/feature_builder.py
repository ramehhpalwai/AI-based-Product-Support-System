from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix, hstack
from src.classification.data_processing import (prepare_classification_dataset,
                                                group_text_and_train_val_test_split,
                                                )

from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class ClassificationFeatures:
    """
    Container for ready-to-train features.
    """
    X_train: csr_matrix
    X_val: csr_matrix
    X_test: csr_matrix
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder
    ohe: OneHotEncoder
    embedder: object  # SentenceTransformer or TfidfVectorizer
    cat_cols: List[str]
    text_col: str


class BuildClassificationFeatures:
    """
    Build features with:
      - Categorical: OneHotEncoder
      - Text: choose ONE of:
          * SentenceTransformer embeddings  (text_mode="transformer")
          * TF-IDF vectors                  (text_mode="tfidf")

    label_mode:
      - "category"
      - "category|subcategory"
    """

    def __init__(
        self,
        tickets_data: Any,
        label_mode: str = "category",
        text_mode: str = "transformer",  # "transformer" or "tfidf"
        text_col: str = "text",
        cat_cols: Optional[List[str]] = None,
        # transformer config
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 256,
        device: str = "cpu",
        normalize_embeddings: bool = True,
        # tfidf config
        tfidf_max_features: int = 80_000,
        tfidf_ngram_range: Tuple[int, int] = (1, 2),
        tfidf_min_df: int = 2,
    ):
        self.tickets_data = tickets_data
        self.label_mode = label_mode
        self.text_mode = text_mode
        self.text_col = text_col
        self.cat_cols = cat_cols or ["product", "product_module", "priority", "channel", "customer_tier"]

        # transformer
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings

        # tfidf
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tfidf_ngram_range
        self.tfidf_min_df = tfidf_min_df

        # fitted objects
        self.le = LabelEncoder()
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.embedder = None  # SentenceTransformer or TfidfVectorizer (set in build)

    def build(self, sample_sizes: Optional[Tuple[int, int, int]] = None) -> ClassificationFeatures:
        # 1) Prepare dataset
        df = pd.DataFrame(prepare_classification_dataset(self.tickets_data))
        train_df, val_df, test_df = group_text_and_train_val_test_split(df)

        if sample_sizes is not None:
            train_n, val_n, test_n = sample_sizes
            train_df = train_df.head(train_n)
            val_df = val_df.head(val_n)
            test_df = test_df.head(test_n)

        # 2) Patch missing categorical cols
        for c in self.cat_cols:
            for d in (train_df, val_df, test_df):
                if c not in d.columns:
                    d[c] = ""

        # 3) Labels
        y_train_raw = self._build_labels(train_df)
        y_val_raw = self._build_labels(val_df)
        y_test_raw = self._build_labels(test_df)

        y_train = self.le.fit_transform(y_train_raw)
        y_val = self.le.transform(y_val_raw)
        y_test = self.le.transform(y_test_raw)

        # 4) X dfs (drop labels)
        label_cols = ["category", "subcategory"]
        X_train_df = train_df.drop(columns=label_cols, errors="ignore")
        X_val_df = val_df.drop(columns=label_cols, errors="ignore")
        X_test_df = test_df.drop(columns=label_cols, errors="ignore")

        # 5) Text features (choose mode)
        X_train_text, X_val_text, X_test_text = self._build_text_features(
            X_train_df[self.text_col].tolist(),
            X_val_df[self.text_col].tolist(),
            X_test_df[self.text_col].tolist(),
        )

        # 6) Categorical features
        X_train_cat = self.ohe.fit_transform(X_train_df[self.cat_cols])
        X_val_cat = self.ohe.transform(X_val_df[self.cat_cols])
        X_test_cat = self.ohe.transform(X_test_df[self.cat_cols])

        # 7) Combine
        X_train = hstack([X_train_text, X_train_cat], format="csr")
        X_val = hstack([X_val_text, X_val_cat], format="csr")
        X_test = hstack([X_test_text, X_test_cat], format="csr")

        return ClassificationFeatures(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            label_encoder=self.le,
            ohe=self.ohe,
            embedder=self.embedder,
            cat_cols=self.cat_cols,
            text_col=self.text_col,
        )

    def _build_text_features(
        self, train_texts: List[str], val_texts: List[str], test_texts: List[str]
    ) -> Tuple[csr_matrix, csr_matrix, csr_matrix]:
        train_texts = [t if isinstance(t, str) else "" for t in train_texts]
        val_texts = [t if isinstance(t, str) else "" for t in val_texts]
        test_texts = [t if isinstance(t, str) else "" for t in test_texts]

        if self.text_mode == "tfidf":
            self.embedder = TfidfVectorizer(
                max_features=self.tfidf_max_features,
                ngram_range=self.tfidf_ngram_range,
                min_df=self.tfidf_min_df,
            )
            X_train_text = self.embedder.fit_transform(train_texts)
            X_val_text = self.embedder.transform(val_texts)
            X_test_text = self.embedder.transform(test_texts)
            return X_train_text.tocsr(), X_val_text.tocsr(), X_test_text.tocsr()

        if self.text_mode == "transformer":
            self.embedder = SentenceTransformer(self.model_name, device=self.device)
            X_train = self.embedder.encode(
                train_texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )
            X_val = self.embedder.encode(
                val_texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )
            X_test = self.embedder.encode(
                test_texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )
            return csr_matrix(X_train), csr_matrix(X_val), csr_matrix(X_test)

        raise ValueError("text_mode must be either 'transformer' or 'tfidf'")

    def _build_labels(self, df: pd.DataFrame) -> pd.Series:
        if self.label_mode == "category":
            return df["category"].astype(str).fillna("")
        
        if self.label_mode == "subcategory":
            return df["subcategory"].astype(str).fillna("")

        if self.label_mode == "category|subcategory":
            cat = df["category"].astype(str).fillna("")
            sub = df["subcategory"].astype(str).fillna("")
            return pd.Series(np.where(sub != "", cat + "|" + sub, cat), index=df.index)

        raise ValueError("label_mode must be 'category' or 'category|subcategory'")


    def transform_new(self, tickets_data: Any) -> csr_matrix:
        """
        Transform new/unseen tickets into model-ready X using
        already-fitted embedder + ohe (no re-fitting).
        """
        if self.embedder is None:
            raise ValueError("Embedder is not fitted. Call build() first.")
        if self.ohe is None:
            raise ValueError("OHE is not fitted. Call build() first.")

        df = pd.DataFrame(prepare_classification_dataset(tickets_data))

        # ensure missing categorical columns exist
        for c in self.cat_cols:
            if c not in df.columns:
                df[c] = ""

        # ensure missing text column exists
        if self.text_col not in df.columns:
            df[self.text_col] = ""

        texts = [t if isinstance(t, str) else "" for t in df[self.text_col].tolist()]

        # text features
        if self.text_mode == "tfidf":
            X_text = self.embedder.transform(texts).tocsr()
        else:
            X_text_dense = self.embedder.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )
            X_text = csr_matrix(X_text_dense)

        # categorical features
        X_cat = self.ohe.transform(df[self.cat_cols])

        # combine
        return hstack([X_text, X_cat], format="csr")
