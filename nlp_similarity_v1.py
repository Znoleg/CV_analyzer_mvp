from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

# Быстрая мультиязычная модель эмбеддингов
# (поддерживает схему "query:" / "passage:" как в примерах модели)
MODEL_NAME = "intfloat/multilingual-e5-small"

_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def compute_similarity(resume_text: str, vacancy_text: str) -> float:
    """
    Возвращает similarity в диапазоне примерно [-1..1] (обычно 0..1).
    Для E5 желательно префиксировать тексты как "query:" и "passage:".
    """
    model = _get_model()
    # В нашей задаче "резюме" можно трактовать как query, "вакансия" как passage (или наоборот — не критично)
    texts = [f"query: {resume_text}", f"passage: {vacancy_text}"]
    emb = model.encode(texts, normalize_embeddings=True)  # нормализация облегчает cosine
    return cosine_sim(emb[0], emb[1])