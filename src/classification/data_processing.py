from typing import Any, Dict, List
import re
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



def clean_error_logs_preserve_codes(text: str) -> str:
    """
    Remove:
      - ISO timestamps (2024-11-27T18:17:26)
      - request IDs (ID-12345, req-123)

    Keep:
      - severity + error code structure: 'ERROR ERROR_SERVER_500:'
      - durations like '60 seconds'
    """
    if not isinstance(text, str):
        return ""

    # 1) remove ISO timestamps
    text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "", text)

    # 2) remove request ids
    text = re.sub(r"\b(id|req)[-_]?\d+\b", "", text, flags=re.IGNORECASE)

    # 3) normalize whitespace/newlines (preserve underscores + colons)
    text = text.replace("\n", " ")
    text = re.sub(r"[ \t]+", " ", text).strip()

    return text


def prepare_classification_dataset(records: List[Any]) -> pd.DataFrame:
    data = []
    for r in records:
        if hasattr(r, "model_dump"):     # Pydantic
            r = r.model_dump()
        elif not isinstance(r, dict):
            r = dict(r)
        r['error_logs'] = clean_error_logs_preserve_codes(r.get('error_logs',''))

        text = f"{r.get('subject','')}\n{r.get('description','')}\n{r.get('error_logs','')}"

        data.append({
            "text": text,
            "product": r.get("product", ""),
            "product_module": r.get("product_module", ""),
            "priority": r.get("priority", ""),
            "channel": r.get("channel", ""),
            "customer_tier": r.get("customer_tier", ""),
            "region": r.get("region", ""),
            "category": r.get("category", ""),
            "subcategory": r.get("subcategory", ""),
            "subject": r.get("subject", ""),
            "description": r.get("description", ""),
            "error_logs": r.get("error_logs", "")
        })

    return pd.DataFrame(data)






def group_text_and_train_val_test_split(
    df: pd.DataFrame,
    label_col: str = "category",
    subject_col: str = "subject",
    description_col: str = "description",
    error_logs_col: str = "error_logs",
    test_size: float = 0.30,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform group-aware train/val/test split to prevent duplicate text leakage.
    it is because data is synthetic. 

    Groups are created from the description text.
    Final split: 70% train, 15% val, 15% test.
    """

    df = df.copy()

    # Create grouping key (keeps duplicates together)
    df["text_key"] = df[description_col].fillna("").astype(str)
    #df = df.drop_duplicates(subset=['text'],keep='first')
    df["group_id"] = pd.util.hash_pandas_object(df["text_key"], index=False)

    # Build combined text field
    df["text"] = (
        df[subject_col].fillna("")
        + " "
        + df[description_col].fillna("")
        + " "
        + df[error_logs_col].fillna("")
    )

    X = df.drop(columns=[label_col, subject_col, description_col, error_logs_col])
    y = df[label_col]
    groups = df["group_id"]

    # Train vs temp (70 / 30)
    gss = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, tmp_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_tmp, y_tmp = X.iloc[tmp_idx], y.iloc[tmp_idx]
    groups_tmp = groups.iloc[tmp_idx]

    # Validation vs test (15 / 15)
    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=random_state + 1,
    )
    val_idx_rel, test_idx_rel = next(
        gss2.split(X_tmp, y_tmp, groups=groups_tmp)
    )

    X_val, y_val = X_tmp.iloc[val_idx_rel], y_tmp.iloc[val_idx_rel]
    X_test, y_test = X_tmp.iloc[test_idx_rel], y_tmp.iloc[test_idx_rel]

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    col_duplicates = ["text","product",
        "product_module",
        "priority",
        "channel",
        "customer_tier",]

    train_df = train_df.drop_duplicates(subset=col_duplicates,keep='first')
    val_df = val_df.drop_duplicates(subset=col_duplicates,keep='first')
    test_df = test_df.drop_duplicates(subset=col_duplicates,keep='first')


    return (train_df,val_df,test_df)



def build_feature_encoder_pipeline_tfidf(
    text_col: str = "text",
    cat_cols: List[str] = None,
    max_features: int = 80_000,
    ngram_range: tuple = (1, 2),
    min_df: int = 2,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer for text + categorical features.

    - TF-IDF for text column
    - One-hot encoding for categorical columns
    """

    if cat_cols is None:
        cat_cols = [
            "product",
            "product_module",
            "priority",
            "channel",
            "customer_tier"
        ]

    text_transformer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
    )

    cat_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("text", text_transformer, text_col),
            ("cat", cat_transformer, cat_cols),
        ],
        remainder="drop",
    )

    return preprocess


# train_df, val_df, test_df = group_text_and_train_val_test_split(classification_data)

# print(len(train_df), len(val_df), len(test_df))
