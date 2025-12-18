import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, classification_report
from catboost import CatBoostClassifier

DATA_PATH = Path("applications_master_synthetic.csv") 
MODEL_PATH = Path("catboost_interview_v0.cbm")

TARGET = "y_interview"

# Эти колонки НЕ используем как фичи
DROP_COLS = [
    "application_id",
    "candidate_id",  # чтобы не было утечки через "запоминание кандидата"
    "resume_id",
    TARGET,
]

# Категориальные фичи (CatBoost умеет их напрямую)
CAT_COLS = ["platform", "vacancy_title", "status_norm", "resume_domain"]

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["application_date"], errors="coerce")
    df["apply_month"] = dt.dt.month.fillna(0).astype(int)
    df["apply_dow"] = dt.dt.dayofweek.fillna(0).astype(int)  # 0=Mon
    # исходную дату оставим, но как строку/категорию не используем
    return df

def main():
    df = pd.read_csv(DATA_PATH)

    # Базовая валидация
    if TARGET not in df.columns:
        raise ValueError(f"Нет колонки {TARGET} в CSV")

    df = add_date_features(df)

    # X/y
    y = df[TARGET].astype(int)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Убедимся, что cat cols существуют
    cat_cols = [c for c in CAT_COLS if c in X.columns]
    # Удалим исходную дату (мы уже извлекли признаки)
    if "application_date" in X.columns:
        X = X.drop(columns=["application_date"])

    # Train/test split (для реальных данных лучше делать time-based split; тут synthetic — обычный)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # CatBoost
    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=100
    )

    model.fit(
        X_train, y_train,
        cat_features=cat_cols,
        eval_set=(X_test, y_test),
        use_best_model=True
    )

    # Предикты
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Метрики
    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)
    f1 = f1_score(y_test, pred)
    acc = accuracy_score(y_test, pred)

    print("\n=== Metrics ===")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC : {pr:.4f}")
    print(f"F1     : {f1:.4f} (threshold=0.5)")
    print(f"ACC    : {acc:.4f}\n")
    print(classification_report(y_test, pred, digits=4))

    # Важность фич
    importances = model.get_feature_importance()
    fi = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values("importance", ascending=False)
    print("\n=== Feature importance (top 20) ===")
    print(fi.head(20).to_string(index=False))

    # Сохранение
    model.save_model(str(MODEL_PATH))
    print(f"\nSaved model to: {MODEL_PATH.resolve()}")

if __name__ == "__main__":
    main()