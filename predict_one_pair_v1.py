import pandas as pd
from catboost import CatBoostClassifier, Pool

from nlp_similarity_v1 import compute_similarity

MODEL_PATH = "catboost_interview_v1.cbm"

CAT_COLS = ["platform", "vacancy_title", "resume_domain"]

def load_model():
    m = CatBoostClassifier()
    m.load_model(MODEL_PATH)
    return m

def predict_interview_proba(
    resume_text: str,
    vacancy_text: str,
    platform: str,
    vacancy_title: str,
    resume_domain: str = "gamedev",
    apply_month: int = 6,
    apply_dow: int = 2,
) -> float:
    sim = compute_similarity(resume_text, vacancy_text)

    row = {
        "platform": platform,
        "vacancy_title": vacancy_title,
        "resume_domain": resume_domain,
        "similarity_resume_vacancy": sim,
        "apply_month": apply_month,
        "apply_dow": apply_dow,
    }

    X = pd.DataFrame([row])
    pool = Pool(X, cat_features=[c for c in CAT_COLS if c in X.columns])

    model = load_model()
    proba = model.predict_proba(pool)[:, 1][0]
    return float(proba)

if __name__ == "__main__":
    # Пример (замени на реальные тексты)
    resume_text = "Unity Developer, 5 years. Addressables, optimization, mobile F2P..."
    vacancy_text = "We need a Unity Developer for mobile F2P, Addressables, analytics, UI..."
    p = predict_interview_proba(
        resume_text=resume_text,
        vacancy_text=vacancy_text,
        platform="HH",
        vacancy_title="Unity Developer (Middle) - Mobile F2P",
    )
    print("P(interview) =", p)
