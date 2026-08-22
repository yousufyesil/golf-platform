"""Embedding + kNN-Vorhersage + Aggregation — portiert aus app.py.

Der Vision-Encoder (config.MODEL_ID) ist über alle Baureihen hinweg derselbe ->
einmal geladen, als Singleton gecacht. Nur der EmbeddingStore unterscheidet sich
je Baureihe und wird pro Baureihe gecacht; ein Rebuild invalidiert den jeweiligen
Cache-Eintrag über invalidate_cache(baureihe).

@author Yousuf Yesil <yousufyesil@icloud.com>
"""
import threading
from pathlib import Path

from . import baureihen_service, config
from .encoders import Encoder
from EmbeddingStore import EmbeddingStore

_encoder: Encoder | None = None
_stores: dict[str, EmbeddingStore] = {}
_lock = threading.Lock()


def get_encoder() -> Encoder:
    """Der eine konfigurierte Vision-Encoder, gemeinsam für alle Baureihen genutzt."""
    global _encoder
    with _lock:
        if _encoder is None:
            _encoder = Encoder(config.MODEL_ID)
        return _encoder


def store_prefix(baureihe: str) -> Path:
    return config.STORE_DIR / f"store_{baureihen_service.slug(baureihe)}"


def get_resources(baureihe: str) -> tuple[Encoder, EmbeddingStore]:
    """Lädt (und cached) den Store einer Baureihe + den gemeinsamen Encoder."""
    with _lock:
        store = _stores.get(baureihe)
    if store is not None:
        return get_encoder(), store

    prefix = store_prefix(baureihe)
    if not Path(f"{prefix}.npy").exists():
        raise FileNotFoundError(f"Kein Store für Baureihe '{baureihe}' vorhanden — zuerst 'Store neu generieren'.")

    store = EmbeddingStore.load(str(prefix))
    with _lock:
        _stores[baureihe] = store
    return get_encoder(), store


def invalidate_cache(baureihe: str) -> None:
    """Nach einem Rebuild den evtl. gecachten (dann veralteten) Store dieser Baureihe verwerfen."""
    with _lock:
        _stores.pop(baureihe, None)


def classify_embedding(emb, store: EmbeddingStore, k: int, metric: str):
    """kNN-Voting für ein einzelnes Embedding -> (pred, prob_dict, confidence, neighbors).

    neighbors sind die k tatsächlich einbezogenen Nachbar-Bilder (für die
    "Erweitert"-Ansicht), sortiert nach Ähnlichkeit, mit similarity als das im
    Voting verwendete Gewicht (cosine: Ähnlichkeit direkt; euclidean: 1/(1+distanz))."""
    neighbours = store.search(emb, k=k, metric=metric)
    scores: dict[str, float] = {}
    neighbor_info = []
    for s, meta in neighbours:
        weight = s if metric == "cosine" else 1.0 / (1.0 + s)
        scores[meta["class"]] = scores.get(meta["class"], 0.0) + weight
        neighbor_info.append({
            "file": meta.get("file"),
            "class": meta["class"],
            "perspektive": meta.get("perspektive"),
            "score": s,
            "similarity": weight,
        })

    pred = max(scores, key=scores.get)
    total = sum(scores.values()) or 1.0
    prob = {cls: s / total for cls, s in scores.items()}
    confidence = prob[pred]
    return pred, prob, confidence, neighbor_info


def aggregate_confidence_weighted(per_view: list[tuple[dict, float]]):
    """Aggregiert mehrere Ansichten, gewichtet nach View-Confidence (siehe app.py)."""
    agg: dict[str, float] = {}
    for prob, conf in per_view:
        for cls, p in prob.items():
            agg[cls] = agg.get(cls, 0.0) + conf * p
    total = sum(agg.values()) or 1.0
    agg = {cls: v / total for cls, v in agg.items()}
    pred = max(agg, key=agg.get)
    return pred, agg, agg[pred]


def run_inference(baureihe: str, images: list[tuple[str, "Image.Image"]], k: int, metric: str, group_mode: bool) -> dict:
    """images: Liste von (dateiname, PIL.Image). Liefert das vollständige Ergebnis für die API."""
    encoder, store = get_resources(baureihe)
    embeddings = encoder.embed([img for _, img in images])

    per_image = []
    for (name, _), emb in zip(images, embeddings):
        pred, prob, conf, neighbors = classify_embedding(emb, store, k=k, metric=metric)
        per_image.append({"name": name, "pred": pred, "prob": prob, "conf": conf, "neighbors": neighbors, "metric": metric})

    result = {"per_image": per_image, "aggregate": None}
    if group_mode and len(per_image) > 1:
        pred, agg, conf = aggregate_confidence_weighted([(d["prob"], d["conf"]) for d in per_image])
        result["aggregate"] = {"pred": pred, "prob": agg, "conf": conf}
    return result
