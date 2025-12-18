import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier, Pool

DATA_PATH = Path("applications_master_synthetic_v3_1200_resume_versions.csv")
MARKET_PATH = Path("market_vacancies_sampled.csv")
MODEL_PATH = Path("catboost_interview_v1.cbm")

CAT_COLS = ["platform", "vacancy_title", "resume_domain"]

def load_model():
    m = CatBoostClassifier()
    m.load_model(str(MODEL_PATH))
    return m

def build_similarity_lookup(df: pd.DataFrame):
    # средняя similarity по (candidate_id,resume_id,platform,vacancy_title)
    grp = (
        df.groupby(["candidate_id", "resume_id", "platform", "vacancy_title"])["similarity_resume_vacancy"]
          .mean()
          .reset_index()
    )
    return grp

def score_one(df_all: pd.DataFrame, market: pd.DataFrame, candidate_id: str, resume_id: str):
    model = load_model()

    # домен резюме (в синтетике всегда gamedev, но оставим как часть интерфейса)
    resume_domain = "gamedev"

    # lookup similarity из исторических строк (MVP proxy)
    sim_df = df_all[
        (df_all["candidate_id"] == candidate_id) &
        (df_all["resume_id"] == resume_id)
    ][["platform", "vacancy_title", "similarity_resume_vacancy"]]

    # fallback если вдруг нет совпадений
    if sim_df.empty:
        raise ValueError("Нет данных по этому candidate_id+resume_id в датасете")

    sim_map = (
        sim_df.groupby(["platform", "vacancy_title"])["similarity_resume_vacancy"]
              .mean()
              .to_dict()
    )

    # строим виртуальные строки рынка
    rows = []
    for _, r in market.iterrows():
        key = (r["platform"], r["vacancy_title"])
        sim = sim_map.get(key, float(sim_df["similarity_resume_vacancy"].mean()))  # fallback на среднее
        rows.append({
            "platform": r["platform"],
            "vacancy_title": r["vacancy_title"],
            "resume_domain": resume_domain,
            "similarity_resume_vacancy": sim,
            # date features можно фиксировать нейтрально
            "apply_month": 6,
            "apply_dow": 2
        })

    X = pd.DataFrame(rows)
    pool = Pool(X, cat_features=[c for c in CAT_COLS if c in X.columns])
    probs = model.predict_proba(pool)[:, 1]

    return {
        "candidate_id": candidate_id,
        "resume_id": resume_id,
        "n_market": len(probs),
        "avg_prob": float(np.mean(probs)),
        "p25": float(np.percentile(probs, 25)),
        "p50": float(np.percentile(probs, 50)),
        "p75": float(np.percentile(probs, 75)),
    }

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    market = pd.read_csv(MARKET_PATH)

    for rid in ["R1", "R2", "R3"]:
        print(score_one(df, market, "cand_01", rid))
