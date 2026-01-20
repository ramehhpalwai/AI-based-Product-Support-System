from dataclasses import dataclass
from typing import Any, List, Optional
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from src.classification.data_processing import prepare_classification_dataset


@dataclass
class TicketFeaturizer:
    text_mode: str                  # "tfidf" or "transformer"
    text_col: str
    cat_cols: List[str]
    ohe: OneHotEncoder
    embedder: Optional[object] = None   # fitted TfidfVectorizer OR SentenceTransformer
    # only needed for transformer mode
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 256
    normalize_embeddings: bool = True

    def transform(self, tickets_data: Any) -> csr_matrix:
        df = pd.DataFrame(prepare_classification_dataset(tickets_data))

        # ensure cols exist
        for c in self.cat_cols:
            if c not in df.columns:
                df[c] = ""
        if self.text_col not in df.columns:
            df[self.text_col] = ""

        texts = [t if isinstance(t, str) else "" for t in df[self.text_col].tolist()]

        # text features
        if self.text_mode == "tfidf":
            if self.embedder is None:
                raise ValueError("TF-IDF vectorizer (embedder) is missing.")
            X_text = self.embedder.transform(texts).tocsr()

        elif self.text_mode == "transformer":
            # lazy-load transformer if not provided
            if self.embedder is None:
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

        # categorical
        X_cat = self.ohe.transform(df[self.cat_cols])

        return hstack([X_text, X_cat], format="csr")


from dataclasses import dataclass
from typing import Any, List
import joblib

from sklearn.preprocessing import LabelEncoder

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


features = builder.build()

predictor = TicketCategoryPredictor(
    model=xgb,
    featurizer=TicketFeaturizer(
        text_mode=builder.text_mode,
        text_col=features.text_col,
        cat_cols=features.cat_cols,
        ohe=features.ohe,
        embedder=features.embedder,           # TF-IDF: fitted vectorizer (good)
        model_name=builder.model_name,        # Transformer: config
        device=builder.device,
        batch_size=builder.batch_size,
        normalize_embeddings=builder.normalize_embeddings,
    ),
    label_encoder=features.label_encoder,
)

predictor.save("ticket_category_predictor.joblib")
