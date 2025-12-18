from __future__ import annotations
import argparse
import pandas as pd
from catboost import CatBoostClassifier, Pool

from text_extract_v1 import extract_text
from nlp_similarity_v1 import compute_similarity
from vacancy_title_infer_v1 import infer_vacancy_title

DEFAULT_MODEL = "catboost_interview_v1.cbm"
CAT_COLS = ["platform", "vacancy_title", "resume_domain"]

def load_model(path: str) -> CatBoostClassifier:
    m = CatBoostClassifier()
    m.load_model(path)
    return m

def predict_proba(model_path: str, resume_text: str, vacancy_text: str,
                  platform: str, vacancy_title: str,
                  resume_domain: str = "gamedev",
                  apply_month: int = 6, apply_dow: int = 2) -> tuple[float, float]:
    """
    Returns: (P(interview), similarity_resume_vacancy)
    """
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

    model = load_model(model_path)
    p = float(model.predict_proba(pool)[:, 1][0])
    return p, float(sim)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", required=True, help="Path to resume file: pdf/docx/txt")
    ap.add_argument("--vacancy", required=True, help="Path to vacancy file: txt/pdf/docx (лучше txt)")
    ap.add_argument("--platform", required=True, choices=["HH", "LinkedIn", "InGameJob"], help="Vacancy platform")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Path to catboost model .cbm")
    args = ap.parse_args()

    resume_text = extract_text(args.resume)
    vacancy_text = extract_text(args.vacancy)

    inferred_title, title_conf = infer_vacancy_title(vacancy_text)

    p, sim = predict_proba(
        model_path=args.model,
        resume_text=resume_text,
        vacancy_text=vacancy_text,
        platform=args.platform,
        vacancy_title=inferred_title
    )

    print(f"Inferred vacancy_title: {inferred_title} (cosine={title_conf:.3f})")
    print(f"similarity_resume_vacancy: {sim:.3f}")
    print(f"P(interview) = {p:.4f}")

    # предупреждение по уверенности классификации типа вакансии
    if title_conf < 0.45:
        print("WARNING: Low confidence vacancy type inference. Vacancy may be out of domain; prediction can be less reliable.")
    elif title_conf < 0.60:
        print("NOTE: Medium confidence vacancy type inference.")

if __name__ == "__main__":
    main()
