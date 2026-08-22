"""UMAP-Projektion der Store-Embeddings einer Baureihe auf 2D — zeigt im
Modell-Tab, wie klar die Klassen im Embedding-Raum getrennt sind (dieselben
L2-normalisierten Embeddings und dieselbe cosine-Metrik wie die kNN-Suche).

Rechnung ist teuer (Sekunden bis niedrige zehner Sekunden je nach Store-Größe)
-> pro Baureihe gecacht. Cache-Key ist die Store-Instanz selbst (id(store)):
ein Rebuild ersetzt die Instanz in inference_service.get_resources, eine
veraltete Projektion wird dadurch nie fälschlich weiterverwendet, ohne dass
dieses Modul store_service' Rebuild-Bookkeeping kennen muss.

@author Yousuf Yesil <yousufyesil@icloud.com>
"""
import numpy as np

from . import inference_service

MAX_POINTS = 3000  # UMAP-Laufzeit + SVG-Rendering bleiben bei sehr großen Stores handhabbar
MIN_POINTS = 5  # UMAP braucht genug Nachbarn für eine sinnvolle Projektion

_cache: dict[str, tuple[int, list[dict]]] = {}


def project(baureihe: str) -> list[dict]:
    """Liefert [{'x', 'y', 'class', 'perspektive', 'file'}, ...] für die Store-
    Embeddings der Baureihe, auf 2D projiziert."""
    _, store = inference_service.get_resources(baureihe)

    cached = _cache.get(baureihe)
    if cached and cached[0] == id(store):
        return cached[1]

    if store.n < MIN_POINTS:
        raise ValueError(
            f"Zu wenige Bilder im Store ({store.n}) für eine UMAP-Projektion — mindestens {MIN_POINTS} nötig."
        )

    import umap  # teuer -> erst bei tatsächlichem Bedarf importieren

    X, meta = store.X, store.meta
    if len(X) > MAX_POINTS:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X), MAX_POINTS, replace=False)
        X, meta = X[idx], [meta[i] for i in idx]

    n_neighbors = min(15, len(X) - 1)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(X)

    points = [
        {
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "class": m["class"],
            "perspektive": m.get("perspektive"),
            "file": m["file"],
        }
        for i, m in enumerate(meta)
    ]
    _cache[baureihe] = (id(store), points)
    return points
