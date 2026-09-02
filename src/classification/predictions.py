from dataclasses import dataclass
from typing import Any, List, Optional

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.classification.data_processing import prepare_classification_dataset


@dataclass
class TicketFeaturizer:
    text_mode: str  # "tfidf" or "transformer"
    text_col: str
    cat_cols: List[str]
    ohe: OneHotEncoder
    embedder: Optional[object] = None  # fitted TfidfVectorizer OR SentenceTransformer
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 256
    normalize_embeddings: bool = True

    def transform(self, tickets_data: Any) -> csr_matrix:
        df = pd.DataFrame(prepare_classification_dataset(tickets_data))

        for c in self.cat_cols:
            if c not in df.columns:
                df[c] = ""
        if self.text_col not in df.columns:
            df[self.text_col] = ""

        texts = [t if isinstance(t, str) else "" for t in df[self.text_col].tolist()]

        if self.text_mode == "tfidf":
            if self.embedder is None:
                raise ValueError("TF-IDF vectorizer (embedder) is missing.")
            X_text = self.embedder.transform(texts).tocsr()

        elif self.text_mode == "transformer":
            if self.embedder is None:
                from sentence_transformers import SentenceTransformer

                self.embedder = SentenceTransformer(self.model_name, device=self.device)

            X_dense = self.embedder.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
            )
            X_text = csr_matrix(X_dense)

        else:
            raise ValueError("text_mode must be 'tfidf' or 'transformer'")

        X_cat = self.ohe.transform(df[self.cat_cols])
        return hstack([X_text, X_cat], format="csr")


@dataclass
class TicketCategoryPredictor:
    model: object
    featurizer: TicketFeaturizer
    label_encoder: LabelEncoder

    def predict(self, tickets_data: Any) -> List[str]:
        X = self.featurizer.transform(tickets_data)
        pred_enc = self.model.predict(X)
        return self.label_encoder.inverse_transform(pred_enc).tolist()

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "TicketCategoryPredictor":
        return joblib.load(path)


def load_category_predictor(path: str) -> TicketCategoryPredictor:
    """Load either a TicketCategoryPredictor or the dict bundle saved by training."""
    loaded = joblib.load(path)
    if isinstance(loaded, TicketCategoryPredictor):
        return loaded
    if not isinstance(loaded, dict):
        raise TypeError(f"Unsupported category model artifact: {type(loaded)!r}")

    featurizer = TicketFeaturizer(
        text_mode=loaded["text_mode"],
        text_col=loaded["text_col"],
        cat_cols=loaded["cat_cols"],
        ohe=loaded["ohe"],
        embedder=loaded.get("embedder"),
        model_name=loaded.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        device=loaded.get("device", "cpu"),
        batch_size=loaded.get("batch_size", 256),
        normalize_embeddings=loaded.get("normalize_embeddings", True),
    )
    return TicketCategoryPredictor(
        model=loaded["model"],
        featurizer=featurizer,
        label_encoder=loaded["label_encoder"],
    )
