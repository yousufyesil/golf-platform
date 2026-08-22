"""Bild-Verwaltung.

Nur handgeprüfte Bilder sind Teil der verwalteten Galerie: die in
train_data.json/test_data.json referenzierten (siehe verified_set.py) plus alle
über die Plattform hochgeladenen (der Nutzer ordnet sie beim Upload selbst einer
Klasse zu). Rohe, nie verifizierte Scraping-Dateien, die zufällig im selben
Ordnerbaum liegen, werden ausgeblendet.

metadata.json liefert Zusatzattribute für Uploads (Perspektive, Upload-Zeitpunkt),
die sich nicht aus dem Pfad ableiten lassen — bei Bildern aus einem Lot-Import
zusätzlich die CSV-Spalten der Anzeige unter "lot" (siehe lot_service.py), die
über den Store-Build in die Store-Metadaten wandern.

@author Yousuf Yesil <yousufyesil@icloud.com>
"""
import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import baureihen_service, config, verified_set

_lock = threading.Lock()


def _safe_component(name: str, label: str) -> str:
    name = (name or "").strip()
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"Ungültiger {label}: {name!r}")
    return name


def _load_metadata() -> dict:
    if config.METADATA_FILE.exists():
        return json.loads(config.METADATA_FILE.read_text(encoding="utf-8"))
    return {}


def _save_metadata(meta: dict) -> None:
    config.METADATA_FILE.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load_all_metadata() -> dict:
    """Öffentlicher Zugriff auf die komplette metadata.json, z.B. für den Store-Build."""
    return _load_metadata()


def _included_index() -> dict[str, dict]:
    """relpath -> {'perspektive', 'quelle', 'lot', 'lid', 'file_id'} für alle Bilder,
    die in der Galerie sichtbar sein sollen: handgeprüfte train/test-Bilder + Uploads.
    'lot' ist nur bei Bildern aus einem Lot-Import gefüllt; 'lid'/'file_id' kommen
    entweder aus train_data.json/test_data.json oder (bei Lot-Bildern) aus 'lot'."""
    index = {
        rel: {"perspektive": info["perspektive"], "quelle": info["source"], "lot": {},
              "lid": info.get("lid", ""), "file_id": info.get("file_id", "")}
        for rel, info in verified_set.verified_index().items()
    }
    for rel, info in _load_metadata().items():
        lot = info.get("lot") or {}
        index[rel] = {
            "perspektive": info.get("perspektive", "unbekannt"),
            "quelle": info.get("quelle", "upload"),
            "lot": lot,
            "lid": lot.get("lid", ""),
            "file_id": lot.get("file_id", ""),
        }
    return index


def iter_included_files() -> list[dict]:
    """Alle Bild-Dateien der verwalteten Galerie (verifiziert + Uploads), die
    tatsächlich auf der Platte liegen -> Basis für Grid-Anzeige und Store-Build."""
    rows = []
    for rel, info in _included_index().items():
        path = config.IMAGE_ROOT / rel
        if not path.is_file():
            continue
        klasse, filename = rel.rsplit("/", 1)
        rows.append({
            "class": klasse,
            "filename": filename,
            "perspektive": info["perspektive"],
            "quelle": info["quelle"],
            "lot": info["lot"],
            "lid": info["lid"],
            "file_id": info["file_id"],
            "path": path,
        })
    return rows


def list_classes() -> list[dict]:
    counts: dict[str, int] = {}
    for row in iter_included_files():
        counts[row["class"]] = counts.get(row["class"], 0) + 1
    return [{"name": k, "count": counts[k]} for k in sorted(counts)]


def list_images(klasse: str) -> list[dict]:
    klasse = _safe_component(klasse, "Klassenname")
    rows = sorted(
        (r for r in iter_included_files() if r["class"] == klasse),
        key=lambda r: r["filename"],
    )
    return [
        {"filename": r["filename"], "class": r["class"], "perspektive": r["perspektive"],
         "quelle": r["quelle"], "lot": r["lot"]}
        for r in rows
    ]


def add_image(klasse: str, filename: str, content: bytes, perspektive: str | None = None) -> dict:
    klasse = _safe_component(klasse, "Klassenname")
    if baureihen_service.baureihe_for_klasse(klasse) is None:
        raise ValueError(f"Klasse '{klasse}' ist keiner Baureihe zugeordnet — zuerst im Daten-Bereich anlegen.")

    ext = Path(filename).suffix.lower()
    if ext not in config.IMAGE_EXTENSIONS:
        raise ValueError(f"Nicht unterstützter Dateityp: {ext or '(keine Endung)'}")

    class_dir = config.IMAGE_ROOT / klasse
    class_dir.mkdir(parents=True, exist_ok=True)
    new_name = f"{uuid.uuid4().hex}{ext}"
    (class_dir / new_name).write_bytes(content)

    with _lock:
        meta = _load_metadata()
        meta[f"{klasse}/{new_name}"] = {
            "perspektive": perspektive or "unbekannt",
            "added_at": datetime.now(timezone.utc).isoformat(),
            "original_filename": filename,
        }
        _save_metadata(meta)

    return {"filename": new_name, "class": klasse}


def register_lot_image(klasse: str, src: Path, lot_meta: dict) -> str:
    """Verschiebt ein Bild aus dem Lot-Staging in die Klasse und hinterlegt die
    CSV-Spalten der Anzeige (lot_meta) am Bild. Gibt den Pfad relativ zu
    IMAGE_ROOT zurück (der Schlüssel in metadata.json).

    Der Dateiname bleibt die file_id (wie im bestehenden golf_images-Bestand),
    damit ein Bild in der Galerie seiner Anzeige zuzuordnen bleibt; bei Kollision
    wird durchnummeriert.
    """
    klasse = _safe_component(klasse, "Klassenname")
    if baureihen_service.baureihe_for_klasse(klasse) is None:
        raise ValueError(f"Klasse '{klasse}' ist keiner Baureihe zugeordnet — zuerst im Daten-Bereich anlegen.")

    class_dir = config.IMAGE_ROOT / klasse
    class_dir.mkdir(parents=True, exist_ok=True)

    dest = class_dir / src.name
    i = 1
    while dest.exists():
        dest = class_dir / f"{src.stem}_{i}{src.suffix}"
        i += 1
    shutil.move(str(src), str(dest))

    rel = f"{klasse}/{dest.name}"
    with _lock:
        meta = _load_metadata()
        meta[rel] = {
            "perspektive": "unbekannt",
            "added_at": datetime.now(timezone.utc).isoformat(),
            "original_filename": src.name,
            "quelle": "lot",
            "lot": lot_meta,
        }
        _save_metadata(meta)
    return rel


def forget_image(rel: str) -> None:
    """Entfernt nur den metadata.json-Eintrag eines Bilds, ohne die Datei zu
    löschen — für das Zurückholen eines Lot-Bilds ins Staging, wo die Datei
    verschoben statt gelöscht wird."""
    with _lock:
        meta = _load_metadata()
        if rel in meta:
            del meta[rel]
            _save_metadata(meta)


def delete_image(klasse: str, filename: str) -> None:
    klasse = _safe_component(klasse, "Klassenname")
    filename = _safe_component(filename, "Dateiname")
    dest = config.IMAGE_ROOT / klasse / filename
    if not dest.is_file():
        raise FileNotFoundError(f"{klasse}/{filename} nicht gefunden")
    dest.unlink()

    with _lock:
        meta = _load_metadata()
        rel = f"{klasse}/{filename}"
        if rel in meta:
            del meta[rel]
            _save_metadata(meta)


def image_path(klasse: str, filename: str) -> Path:
    klasse = _safe_component(klasse, "Klassenname")
    filename = _safe_component(filename, "Dateiname")
    path = config.IMAGE_ROOT / klasse / filename
    if not path.is_file():
        raise FileNotFoundError(f"{klasse}/{filename} nicht gefunden")
    return path


def has_verified_images(klasse: str) -> bool:
    """True, wenn die Klasse handgeprüfte train/test-Bilder enthält (deren
    Zuordnung fest in train_data.json/test_data.json auf den aktuellen
    Ordnernamen verdrahtet ist -> so eine Klasse darf nicht umbenannt werden).
    Uploads UND Lot-Importe sind dagegen frei umbenennbar -- ihre Zuordnung
    liegt in metadata.json und wird bei rename_class_folder() mitmigriert."""
    return any(r["class"] == klasse and r["quelle"] in ("train", "test") for r in iter_included_files())


def rename_class_folder(old_klasse: str, new_klasse: str) -> None:
    """Benennt den Upload-Ordner einer Klasse um und migriert die zugehörigen
    metadata.json-Einträge. Nur für Klassen ohne handgeprüfte Bilder sicher
    (siehe has_verified_images) — wird vom Aufrufer geprüft."""
    old_klasse = _safe_component(old_klasse, "Klassenname")
    new_klasse = _safe_component(new_klasse, "Klassenname")
    old_dir = config.IMAGE_ROOT / old_klasse
    new_dir = config.IMAGE_ROOT / new_klasse
    if new_dir.exists():
        raise ValueError(f"Ordner für Klasse '{new_klasse}' existiert bereits.")

    with _lock:
        if old_dir.is_dir():
            old_dir.rename(new_dir)

        meta = _load_metadata()
        prefix = f"{old_klasse}/"
        changed = False
        for rel in [r for r in meta if r.startswith(prefix)]:
            meta[f"{new_klasse}/{rel[len(prefix):]}"] = meta.pop(rel)
            changed = True
        if changed:
            _save_metadata(meta)


def delete_class_uploads(klasse: str) -> int:
    """Löscht alle hochgeladenen Bilder einer Klasse (Dateien + metadata.json-Einträge)
    und entfernt den Klassen-Ordner, falls er dadurch leer wird. Verifizierte
    train/test-Bilder werden NICHT von der Platte gelöscht (geteilte Daten).
    Gibt die Anzahl gelöschter Upload-Dateien zurück."""
    klasse = _safe_component(klasse, "Klassenname")
    deleted = 0
    with _lock:
        meta = _load_metadata()
        for rel in [r for r in meta if r.split("/", 1)[0] == klasse]:
            f = config.IMAGE_ROOT / rel
            if f.is_file():
                f.unlink()
                deleted += 1
            del meta[rel]
        _save_metadata(meta)

    class_dir = config.IMAGE_ROOT / klasse
    if class_dir.is_dir() and not any(class_dir.iterdir()):
        class_dir.rmdir()
    return deleted
