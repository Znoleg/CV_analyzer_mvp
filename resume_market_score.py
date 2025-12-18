import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier, Pool

DATA_PATH = Path("applications_master_synthetic_v3_1200_resume_versions.csv")
MODEL_PATH = Path("catboost_interview_v1.cbm")

CAT_COLS = ["platform", "vacancy_title", "resume_domain"]

DROP_COLS = [
    "application_id",
    "candidate_id",
    "resume_id",
    "status_norm",
    "y_interview",
    "application_date",
]

def load_model():
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    return model

def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X = X.drop(columns=[c for c in DROP_COLS if c in X.columns], errors="ignore")
    return X

def score_resume_market(df: pd.DataFrame, candidate_id: str, resume_id: str):
    model = load_model()

    subset = df[
        (df["candidate_id"] == candidate_id) &
        (df["resume_id"] == resume_id)
    ]

    if subset.empty:
        raise ValueError("Нет строк для данного резюме")

    X = prepare_X(subset)

    cat_cols = [c for c in CAT_COLS if c in X.columns]

    pool = Pool(
        data=X,
        cat_features=cat_cols
    )

    probs = model.predict_proba(pool)[:, 1]

    return {
        "candidate_id": candidate_id,
        "resume_id": resume_id,
        "n_vacancies": int(len(probs)),
        "avg_prob": float(np.mean(probs)),
        "p25": float(np.percentile(probs, 25)),
        "p50": float(np.percentile(probs, 50)),
        "p75": float(np.percentile(probs, 75)),
    }

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    for resume_id in ["R1", "R2", "R3"]:
        result = score_resume_market(df, "cand_01", resume_id)
        print(result)
