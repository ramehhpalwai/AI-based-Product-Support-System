from typing import Any, Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

def prepare_classification_dataset(records: List[Any]) -> pd.DataFrame:
    data = []
    for r in records:
        if hasattr(r, "model_dump"):     # Pydantic
            r = r.model_dump()
        elif not isinstance(r, dict):
            r = dict(r)

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
        })

    return pd.DataFrame(data)



# IMPORTANT: pass the list of records (not json_data[0] unless nested)
classification_data = prepare_classification_dataset(json_data[0])

X = classification_data.drop("category", axis=1)
y = classification_data["category"]

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
)

preprocess = ColumnTransformer(
    transformers=[
        ("text", TfidfVectorizer(max_features=80_000, ngram_range=(1, 2), min_df=2), "text"),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         ["product","product_module","priority","channel","customer_tier","region"]),
    ],
)

X_train_t = preprocess.fit_transform(X_train)
X_val_t = preprocess.transform(X_val)
X_test_t = preprocess.transform(X_test)

model = LogisticRegression(max_iter=2000, class_weight="balanced")
model.fit(X_train_t, y_train)

val_pred = model.predict(X_val_t)
print("VAL weighted F1:", round(f1_score(y_val, val_pred, average="weighted"), 4))
print(classification_report(y_val, val_pred, digits=3))

test_pred = model.predict(X_test_t)
print("TEST weighted F1:", round(f1_score(y_test, test_pred, average="weighted"), 4))
print(classification_report(y_test, test_pred, digits=3))
