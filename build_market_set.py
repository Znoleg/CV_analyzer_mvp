import pandas as pd
from pathlib import Path

DATA_PATH = Path("applications_master_synthetic_v3_1200_resume_versions.csv")

OUT_WEIGHTED = Path("market_vacancies_weighted.csv")
OUT_SAMPLED = Path("market_vacancies_sampled.csv")

SAMPLE_SIZE = 1000  # сколько "рыночных" вакансий используем для оценки (можно 2000/5000)

VACANCY_KEYS = ["platform", "vacancy_title"]  # позже добавим vacancy_text/embedding

def main():
    df = pd.read_csv(DATA_PATH)

    # Частоты вакансий в данных = прокси распределения рынка
    weighted = (
        df.groupby(VACANCY_KEYS, dropna=False)
          .size()
          .reset_index(name="weight")
          .sort_values("weight", ascending=False)
    )

    weighted.to_csv(OUT_WEIGHTED, index=False)

    # Сэмплируем фиксированный рынок с повторениями по весам
    sampled = weighted.sample(
        n=SAMPLE_SIZE,
        replace=True,
        weights="weight",
        random_state=42
    ).reset_index(drop=True)

    sampled.to_csv(OUT_SAMPLED, index=False)

    print(f"Saved: {OUT_WEIGHTED.resolve()}")
    print(f"Saved: {OUT_SAMPLED.resolve()}  (n={len(sampled)})")
    print("\nTop weighted vacancies:")
    print(weighted.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
