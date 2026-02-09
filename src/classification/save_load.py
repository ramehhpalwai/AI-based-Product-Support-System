from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

import joblib

from src.classification.feature_builder import BuildClassificationFeatures

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINED_MODELS_DIR = PROJECT_ROOT / "src" / "classification" / "trained_models"

def _safe_name(s: str) -> str:
    """Make a filesystem-safe name."""
    return (
        s.strip()
        .lower()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def save_category_artifacts(
    *,
    category: str,
    model: Any,
    features: BuildClassificationFeatures,
    text_mode: str,
    out_dir: str = "src/classification/trained_models",
    transformer_model_name: str | None = None,
    transformer_batch_size: int | None = None,
    transformer_device: str | None = None,
    normalize_embeddings: bool | None = None,
) -> Dict[str, str]:
    """
    Saves:
      - model (.joblib)
      - feature artifacts/pipeline (.joblib) for TF-IDF, or lightweight config for transformer
      - meta (.json)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cat_key = _safe_name(category)

    model_path = os.path.join(out_dir, f"xgb_subcategory__{cat_key}.joblib")
    pipe_path = os.path.join(out_dir, f"features_pipeline__{cat_key}.joblib")
    meta_path = os.path.join(out_dir, f"meta__{cat_key}.json")

    # 1) Save model
    joblib.dump(model, model_path)

    # 2) Save "pipeline"
    # We store ONLY what we need for transform_new + label decode
    pipeline_obj = {
        "label_encoder": features.label_encoder,
        "ohe": features.ohe,
        "cat_cols": features.cat_cols,
        "text_col": features.text_col,
        "text_mode": text_mode,
    }

    if text_mode == "tfidf":
        pipeline_obj["embedder"] = features.embedder  # fitted TfidfVectorizer
        joblib.dump(pipeline_obj, pipe_path)
    else:
        # transformer: don't dump the SentenceTransformer object; store config
        pipeline_obj["transformer"] = {
            "model_name": transformer_model_name,
            "batch_size": transformer_batch_size,
            "device": transformer_device,
            "normalize_embeddings": normalize_embeddings,
        }
        joblib.dump(pipeline_obj, pipe_path)

    # 3) Save meta (optional but useful)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "category": category,
                "model_path": model_path,
                "pipeline_path": pipe_path,
                "text_mode": text_mode,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {"model_path": model_path, "pipeline_path": pipe_path, "meta_path": meta_path}


def load_category_artifacts(
    category: str,
    out_dir: str = "trained_models",
):
    """Loads model + pipeline dict."""
    cat_key = _safe_name(category)

    model_path = os.path.join(out_dir, f"xgb_subcategory__{cat_key}.joblib")
    pipe_path = os.path.join(out_dir, f"features_pipeline__{cat_key}.joblib")

    model = joblib.load(model_path)
    pipeline = joblib.load(pipe_path)
    return model, pipeline
