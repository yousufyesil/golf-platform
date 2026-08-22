"""@author Yousuf Yesil <yousufyesil@icloud.com>"""

import numpy as np
import pandas as pd


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Zeilenweise auf Länge 1. Akzeptiert [d] (ein Vektor) oder [n, d] (Batch)."""
    x = np.atleast_2d(x).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class EmbeddingStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.X = np.empty((0, dim), dtype=np.float32)   # normalisierte Embeddings
        self.meta: list[dict] = []

    # ---- Aufbau ----
    def add(self, embeddings: np.ndarray, rows: list[dict]) -> None:
        emb = l2_normalize(embeddings)
        assert emb.shape[0] == len(rows), "Embeddings und Metadaten müssen gleich viele Zeilen haben"
        self.X = np.vstack([self.X, emb])
        self.meta.extend(rows)

    # ---- Score-Berechnung ----
    def _scores(self, q: np.ndarray, metric: str) -> tuple[np.ndarray, bool]:
        """
        Gibt (scores, higher_is_better) zurück.
        - cosine:    Cosinus-Ähnlichkeit,  größer = besser
        - euclidean: euklidische Distanz,  kleiner = besser
        """
        q = l2_normalize(q)[0]  # [d]
        if metric == "cosine":
            return self.X @ q, True
        elif metric == "euclidean":
            # ||x - q||_2, zeilenweise
            return np.linalg.norm(self.X - q, axis=1), False
        else:
            raise ValueError(f"Unbekannte Metrik: {metric!r}. Erlaubt: 'cosine', 'euclidean'.")

    # ---- Suche ----
    def search(
        self,
        q: np.ndarray,
        k: int = 10,
        metric: str = "cosine",
    ) -> list[tuple[float, dict]]:
        """Gibt die k ähnlichsten Bilder als (score, metadaten) zurück.
        Sortierung: bester Treffer zuerst (bei cosine absteigend, bei euclidean aufsteigend).
        """
        scores, higher_is_better = self._scores(q, metric)
        k = min(k, len(scores))

        # Vorzeichen so drehen, dass "kleiner Wert" = "besser" -> np.argpartition funktioniert einheitlich
        ranking_key = -scores if higher_is_better else scores
        top = np.argpartition(ranking_key, k - 1)[:k]
        top = top[np.argsort(ranking_key[top])]

        return [(float(scores[i]), self.meta[i]) for i in top]

    # ---- Klassenentscheidung (gewichtetes Voting) ----
    def predict(
        self,
        q: np.ndarray,
        k: int = 5,
        metric: str = "cosine",
    ) -> tuple[str, dict[str, float]]:
        """Gewichtetes Votum über die k nächsten Nachbarn.
        Gewicht:
        - cosine:    Ähnlichkeit direkt (0..1, größer = besser)
        - euclidean: 1 / (1 + distanz)  (näher = mehr Gewicht)
        """
        neighbours = self.search(q, k, metric=metric)
        scores: dict[str, float] = {}
        for s, meta in neighbours:
            weight = s if metric == "cosine" else 1.0 / (1.0 + s)
            scores[meta["class"]] = scores.get(meta["class"], 0.0) + weight
        best = max(scores, key=scores.get)
        return best, scores

    # ---- Persistenz ----
    def save(self, prefix: str) -> None:
        np.save(f"{prefix}.npy", self.X)
        pd.DataFrame(self.meta).to_parquet(f"{prefix}.parquet")

    def remove(self, indices) -> None:
        mask = np.ones(len(self.X), dtype=bool)
        mask[list(indices)] = False
        self.X = self.X[mask]
        self.meta = [m for keep, m in zip(mask, self.meta) if keep]

    @classmethod
    def load(cls, prefix: str) -> "EmbeddingStore":
        X = np.load(f"{prefix}.npy")
        store = cls(dim=X.shape[1])
        store.X = X.astype(np.float32)
        store.meta = pd.read_parquet(f"{prefix}.parquet").to_dict("records")
        assert len(store.meta) == store.X.shape[0], "Matrix und Metadaten sind nicht synchron!"
        return store

    @property
    def n(self) -> int:
        return len(self.meta)

    @property
    def klassen(self) -> list[str]:
        return sorted(set(m["class"] for m in self.meta))

    @property
    def klassen_verteilung(self) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        for m in self.meta:
            key = (m["class"], m.get("perspektive", "?"))
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def eintraege(self) -> list[tuple[int, dict]]:
        return list(enumerate(self.meta))

    def info(self) -> None:
        print(f"Store: {self.n} Embeddings, dim={self.dim}, {len(self.klassen)} Klassen")
        for (cls, persp), count in self.klassen_verteilung.items():
            print(f"  {cls:25s} {persp:20s} {count:5d}")