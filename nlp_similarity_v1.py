from __future__ import annotations
import os
import numpy as np
from sentence_transformers import SentenceTransformer

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_PATH = r"./models/multilingual-e5-small"

_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_PATH, local_files_only=True)
    return _model

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def compute_similarity(resume_text: str, vacancy_text: str) -> float:
    model = _get_model()
    texts = [f"query: {resume_text}", f"passage: {vacancy_text}"]
    emb = model.encode(texts, normalize_embeddings=True)
    return cosine_sim(emb[0], emb[1])