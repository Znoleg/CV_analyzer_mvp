from __future__ import annotations
import os
import numpy as np
from sentence_transformers import SentenceTransformer

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_PATH = r"./models/multilingual-e5-small"
_model = None
_cache = {}

KNOWN_VACANCY_TITLES = [
    "Unity Developer (Middle) - Mobile F2P",
    "Unity Developer (WebGL) - Casual Games",
    "Senior Unity Developer - Multiplayer (Photon)",
    "Senior Unity Developer - Architecture / Tools",
    "Unity Developer - Hypercasual / LiveOps",
    "Senior Unity Developer - VR/AR",
]

# Двуязычные прототипы для лучшего матчинга с RU/EN вакансиями
TITLE_PROTOTYPES = {
    "Unity Developer (Middle) - Mobile F2P":
        "Unity developer middle mobile free-to-play F2P analytics SDK UI Addressables IAP ads. "
        "Unity разработчик middle мобильные игры F2P монетизация реклама аналитика SDK UI Addressables",
    "Unity Developer (WebGL) - Casual Games":
        "Unity developer WebGL browser casual games performance optimization UI. "
        "Unity разработчик WebGL браузерные казуальные игры оптимизация производительности UI",
    "Senior Unity Developer - Multiplayer (Photon)":
        "Senior Unity developer multiplayer networking Photon PUN Fusion netcode synchronization optimization. "
        "Senior Unity разработчик мультиплеер сетевое взаимодействие Photon PUN Fusion netcode синхронизация оптимизация",
    "Senior Unity Developer - Architecture / Tools":
        "Senior Unity developer architecture tools frameworks DI Zenject refactoring editor tools CI pipelines. "
        "Senior Unity разработчик архитектура инструменты фреймворки DI Zenject рефакторинг editor tools CI пайплайны",
    "Unity Developer - Hypercasual / LiveOps":
        "Unity developer hypercasual liveops A/B tests monetization ads mediation remote config events. "
        "Unity разработчик гиперказуал liveops A/B тесты монетизация реклама медиация remote config ивенты",
    "Senior Unity Developer - VR/AR":
        "Senior Unity developer VR AR XR OpenXR Oculus Quest ARKit ARCore performance interaction. "
        "Senior Unity разработчик VR AR XR OpenXR Oculus Quest ARKit ARCore оптимизация взаимодействие",
}

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_PATH, local_files_only=True)
    return _model

def _clean(text: str, limit: int = 8000) -> str:
    # убираем лишние пробелы/переводы, режем хвост
    text = " ".join((text or "").split())
    return text[:limit]

def _embed(text: str) -> np.ndarray:
    model = _get_model()
    emb = model.encode([f"query: {text}"], normalize_embeddings=True)[0]
    return emb

def infer_vacancy_title(vacancy_text: str) -> tuple[str, float]:
    """
    Returns: (best_title, confidence_cosine)
    confidence_cosine ~ cosine similarity (0..1 чаще всего).
    """
    vacancy_text = _clean(vacancy_text)
    v_emb = _embed(vacancy_text)

    best_title = None
    best_score = -1.0

    for title, proto in TITLE_PROTOTYPES.items():
        if title not in _cache:
            _cache[title] = _embed(proto)
        score = float(np.dot(v_emb, _cache[title]))  # cosine for normalized vectors
        if score > best_score:
            best_score = score
            best_title = title

    # защита на всякий случай
    if best_title is None:
        best_title = KNOWN_VACANCY_TITLES[0]
        best_score = 0.0

    return best_title, best_score
