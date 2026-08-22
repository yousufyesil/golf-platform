"""Store-(Re-)Build als Hintergrund-Job — ein Store je Baureihe (z.B. Golf, Polo),
mit dem einen konfigurierten Encoder (config.MODEL_ID, siehe inference_service).

Embedded alle Bilder der Klassen einer Baureihe aus der verwalteten Galerie
(image_service.iter_included_files, nur handgeprüfte train/test-Bilder + Uploads)
batchweise und schreibt store_<baureihe-slug>.{npy,parquet,json}. Läuft in einem
Hintergrund-Thread, Fortschritt wird über einen Job-Dict abgefragt.

Sonderfall config.GLOBAL_BAUREIHE ("Gesamt"): kein Filter auf Klassen einer
Baureihe, sondern alle Bilder der gesamten Galerie -> ein Store über alle
Marken/Baureihen hinweg, z.B. für baureihenübergreifende Ähnlichkeitssuche.

@author Yousuf Yesil <yousufyesil@icloud.com>
"""
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from . import baureihen_service, config, image_service, inference_service, lot_service
from EmbeddingStore import EmbeddingStore

# Spalten aus dem Lot-Import (lot.csv), die je Bild in die Store-Metadaten
# geschrieben werden. Immer alle Spalten setzen, auch für Bilder ohne Lot-Bezug
# (dann leer) — sonst wird die Parquet-Datei je nach Bestand unterschiedlich breit.
STORE_LOT_FIELDS = lot_service.LOT_FIELDS + ("file_id",)

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_active_baureihen: set[str] = set()


def _prefix(baureihe: str) -> Path:
    return config.STORE_DIR / f"store_{baureihen_service.slug(baureihe)}"


def get_status(baureihe: str) -> dict:
    """Status des Stores einer Baureihe: existiert er, wie viele Bilder/Dim, wann gebaut."""
    prefix = _prefix(baureihe)
    npy, sidecar = Path(f"{prefix}.npy"), Path(f"{prefix}.json")

    if sidecar.exists():
        info = json.loads(sidecar.read_text(encoding="utf-8"))
    elif npy.exists():
        shape = np.load(npy, mmap_mode="r").shape
        info = {
            "model_id": config.MODEL_ID,
            "dim": int(shape[1]) if len(shape) > 1 else None,
            "n_images": int(shape[0]),
            "built_at": None,
        }
    else:
        info = {"model_id": config.MODEL_ID, "dim": None, "n_images": None, "built_at": None}

    info["baureihe"] = baureihe
    info["exists"] = npy.exists()
    return info


def delete_store(baureihe: str) -> None:
    """Löscht die Store-Dateien einer Baureihe (npy/parquet/json) und invalidiert den Cache."""
    prefix = _prefix(baureihe)
    for ext in ("npy", "parquet", "json"):
        f = Path(f"{prefix}.{ext}")
        if f.exists():
            f.unlink()
    inference_service.invalidate_cache(baureihe)


def get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def start_rebuild(baureihe: str) -> str:
    with _jobs_lock:
        if baureihe in _active_baureihen:
            raise RuntimeError(f"Für Baureihe '{baureihe}' läuft bereits ein Rebuild.")
        _active_baureihen.add(baureihe)

        job_id = uuid.uuid4().hex
        _jobs[job_id] = {
            "job_id": job_id,
            "baureihe": baureihe,
            "model_id": config.MODEL_ID,
            "status": "running",
            "processed": 0,
            "total": 0,
            "error": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
        }

    threading.Thread(target=_run_rebuild, args=(job_id, baureihe), daemon=True).start()
    return job_id


def _update(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


def _run_rebuild(job_id: str, baureihe: str) -> None:
    try:
        if baureihe == config.GLOBAL_BAUREIHE:
            rows = image_service.iter_included_files()
        else:
            klassen = set(baureihen_service.klassen_for_baureihe(baureihe))
            if not klassen:
                raise RuntimeError(f"Keine Klassen für Baureihe '{baureihe}' gefunden.")
            rows = [r for r in image_service.iter_included_files() if r["class"] in klassen]

        _update(job_id, total=len(rows))
        if not rows:
            raise RuntimeError(f"Keine handgeprüften Bilder für Baureihe '{baureihe}' gefunden.")

        encoder = inference_service.get_encoder()
        store: EmbeddingStore | None = None

        for i in range(0, len(rows), config.BATCH_SIZE):
            batch = rows[i:i + config.BATCH_SIZE]
            images = [Image.open(r["path"]).convert("RGB") for r in batch]
            embeddings = encoder.embed(images)

            if store is None:
                store = EmbeddingStore(dim=embeddings.shape[-1])

            meta_rows = [
                {
                    "class": r["class"],
                    "perspektive": r["perspektive"],
                    "quelle": r["quelle"],
                    "file": str(r["path"].relative_to(config.IMAGE_ROOT)),
                    **{f: str(r["lot"].get(f, "") or "") for f in STORE_LOT_FIELDS},
                }
                for r in batch
            ]
            store.add(embeddings, meta_rows)
            _update(job_id, processed=min(i + len(batch), len(rows)))

        prefix = _prefix(baureihe)
        store.save(str(prefix))
        Path(f"{prefix}.json").write_text(
            json.dumps({
                "model_id": config.MODEL_ID,
                "dim": store.dim,
                "n_images": store.n,
                "built_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        inference_service.invalidate_cache(baureihe)
        _update(job_id, status="done", finished_at=datetime.now(timezone.utc).isoformat())

    except Exception as e:  # Job-Fehler sollen den Prozess nicht crashen, nur im Job sichtbar sein
        _update(job_id, status="error", error=str(e), finished_at=datetime.now(timezone.utc).isoformat())
    finally:
        with _jobs_lock:
            _active_baureihen.discard(baureihe)
