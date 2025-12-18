import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score, classification_report
)
from catboost import CatBoostClassifier

DATA_PATH = Path("applications_master_synthetic_v3_1200_resume_versions.csv")
MODEL_PATH = Path("catboost_interview_v1.cbm")

TARGET = "y_interview"

# Все колонки, которые точно нельзя использовать как фичи (утечки / ID / постфактум)
DROP_COLS = [
    "application_id",
    "candidate_id",
    "resume_id",
    "status_raw",
    "status_norm",      # <-- УТЕЧКА
    "funnel_stage",     # если появится потом
    "y_offer",          # если появится потом
    TARGET
]

# Категориальные признаки
CAT_COLS = ["platform", "vacancy_title", "resume_domain"]

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df.get("application_date"), errors="coerce")
    df["apply_month"] = dt.dt.month.fillna(0).astype(int)
    df["apply_dow"] = dt.dt.dayofweek.fillna(0).astype(int)  # 0=Mon
    return df

def main():
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Нет колонки {TARGET} в CSV")

    df = add_date_features(df)

    y = df[TARGET].astype(int)

    # Удаляем недопустимые колонки (если каких-то нет — ок)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Исходную дату не используем напрямую
    if "application_date" in X.columns:
        X = X.drop(columns=["application_date"])

    # CatBoost требует список cat_features как имена или индексы
    cat_cols = [c for c in CAT_COLS if c in X.columns]

    # Сплит
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=200
    )

    model.fit(
        X_train, y_train,
        cat_features=cat_cols,
        eval_set=(X_test, y_test),
        use_best_model=True
    )

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)
    f1 = f1_score(y_test, pred)
    acc = accuracy_score(y_test, pred)

    print("\n=== Metrics (no leakage) ===")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC : {pr:.4f}")
    print(f"F1     : {f1:.4f} (threshold=0.5)")
    print(f"ACC    : {acc:.4f}\n")
    print(classification_report(y_test, pred, digits=4))

    # Важность фич
    importances = model.get_feature_importance()
    fi = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values("importance", ascending=False)
    print("\n=== Feature importance ===")
    print(fi.to_string(index=False))

    model.save_model(str(MODEL_PATH))
    print(f"\nSaved model to: {MODEL_PATH.resolve()}")

if __name__ == "__main__":
    main()