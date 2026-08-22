"""Lot-Import: Anzeigen aus einer lot.csv nachträglich einer Klasse zuordnen.

Ablauf:
1. lot.csv hochladen -> je Anzeige (lid) ein Datensatz in lots.json, mit allen
   Spalten aus dem Header und der Liste der zugehörigen file_ids.
2. Bilder hochladen (einzeln oder als Zip) -> landen unter ihrer file_id im
   Staging-Ordner (config.LOT_DIR), noch ausserhalb der Galerie. Reihenfolge egal:
   der Vorhanden-/Fehlt-Status einer Anzeige wird bei jeder Abfrage aus dem
   Ordnerinhalt abgeleitet, nicht in lots.json mitgeschrieben.
3. Zuordnung -> die Bilder der Anzeige wandern nach IMAGE_ROOT/<KLASSE>/ und sind
   ab da normale Galerie-Bilder. Die CSV-Spalten der Anzeige werden dabei in
   metadata.json am Bild hinterlegt und landen so über den Store-Build in den
   Store-Metadaten (siehe image_service.register_lot_image, store_service).

file_id <-> Dateiname: "36916,026a9b06872139374a" entspricht der Datei
"36916_026a9b06872139374a.webp" — das Komma ist im Dateinamen ein Unterstrich
(gleiche Konvention wie im bestehenden golf_images-Bestand).

@author Yousuf Yesil <yousufyesil@icloud.com>
"""
import csv
import io
import json
import shutil
import threading
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from . import config, image_service

# Spalten aus dem lot.csv-Header, die als Metadaten an der Anzeige (und später am
# Bild und im Store) hängen. Fehlt eine Spalte in der Datei, bleibt sie leer.
LOT_FIELDS = ("table", "lid", "title", "first_registration", "make", "model", "fuel", "body_type")

_lock = threading.Lock()


# ---- file_id <-> Dateiname ----

def slug_for_file_id(file_id: str) -> str:
    """file_id -> Dateiname ohne Endung. Nur Zeichen, die als Dateiname sicher
    sind; Komma (das Trennzeichen der file_id) wird zum Unterstrich."""
    return "".join(c if c.isalnum() else "_" for c in file_id.strip())


def _staging_index() -> dict[str, Path]:
    """slug -> Datei im Staging-Ordner. Einzige Quelle dafür, welches Bild
    tatsächlich vorliegt."""
    index = {}
    for p in config.LOT_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS:
            index[p.stem] = p
    return index


# ---- Persistenz ----

def _load() -> dict:
    if config.LOTS_FILE.exists():
        return json.loads(config.LOTS_FILE.read_text(encoding="utf-8"))
    return {}


def _save(lots: dict) -> None:
    config.LOTS_FILE.write_text(json.dumps(lots, indent=2, ensure_ascii=False), encoding="utf-8")


# ---- CSV-Import ----

def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Datei-Encoding nicht lesbar (weder UTF-8 noch Latin-1).")


def _reader(text: str) -> csv.DictReader:
    """DictReader mit erkanntem Trennzeichen (Tab, Komma oder Semikolon)."""
    head = text[:8192].splitlines()[0] if text.strip() else ""
    delimiter = max(("\t", ",", ";"), key=head.count) if head else ","
    return csv.DictReader(io.StringIO(text), delimiter=delimiter)


def _file_ids_from_row(row: dict) -> list[str]:
    """Eine file_id je Zeile (lot.csv) oder eine JSON-Liste in 'file_ids'
    (Format des bestehenden golf_models-Exports) — beides wird unterstützt."""
    raw = (row.get("file_id") or "").strip()
    if raw:
        return [raw]

    raw = (row.get("file_ids") or "").strip()
    if raw.startswith("["):
        try:
            return [str(f).strip() for f in json.loads(raw) if str(f).strip()]
        except json.JSONDecodeError:
            return []
    return [raw] if raw else []


def import_csv(raw: bytes) -> dict:
    """Liest eine lot.csv ein und legt/ergänzt je lid eine Anzeige an.

    Bereits zugeordnete Anzeigen werden nicht angefasst; bei bereits bekannten
    offenen Anzeigen kommen nur neue file_ids dazu (mehrfacher Import derselben
    Datei ist damit unschädlich).
    """
    text = _decode(raw)
    reader = _reader(text)
    if not reader.fieldnames or "lid" not in reader.fieldnames:
        raise ValueError("Spalte 'lid' fehlt — ist das eine lot.csv?")
    if "file_id" not in reader.fieldnames and "file_ids" not in reader.fieldnames:
        raise ValueError("Spalte 'file_id' fehlt — ist das eine lot.csv?")

    now = datetime.now(timezone.utc).isoformat()
    zeilen = 0
    # Je Anzeige zählen, nicht je Zeile — eine Anzeige steht mit einer Zeile je Bild
    # in der Datei und würde sonst mehrfach gezählt.
    neu: set[str] = set()
    ergaenzt: set[str] = set()
    uebersprungen: set[str] = set()

    with _lock:
        lots = _load()
        for row in reader:
            lid = (row.get("lid") or "").strip()
            file_ids = _file_ids_from_row(row)
            if not lid or not file_ids:
                continue
            zeilen += 1

            lot = lots.get(lid)
            if lot is None:
                lot = {f: (row.get(f) or "").strip() for f in LOT_FIELDS}
                lot.update({"lid": lid, "file_ids": [], "imported_at": now,
                            "klasse": None, "assigned_at": None, "assigned_files": {}})
                lots[lid] = lot
                neu.add(lid)
            elif lot.get("klasse"):
                uebersprungen.add(lid)
                continue

            for fid in file_ids:
                if fid not in lot["file_ids"]:
                    lot["file_ids"].append(fid)
                    if lid not in neu:
                        ergaenzt.add(lid)

        _save(lots)

    return {"zeilen": zeilen, "neue_anzeigen": len(neu), "ergaenzte_anzeigen": len(ergaenzt),
            "uebersprungen_zugeordnet": len(uebersprungen)}


# ---- Bild-Import ----

def _known_slugs() -> dict[str, str]:
    """slug -> lid für alle file_ids aller Anzeigen. Bilder ohne passende Anzeige
    werden beim Upload abgelehnt (sonst sammelt sich stiller Müll im Staging)."""
    return {slug_for_file_id(fid): lid for lid, lot in _load().items() for fid in lot["file_ids"]}


def import_images(files: list[tuple[str, bytes]]) -> dict:
    """Legt hochgeladene Bilder unter ihrer file_id im Staging-Ordner ab.
    Zip-Archive werden entpackt. Gibt Zähler + die ersten unbekannten Dateinamen
    zurück."""
    known = _known_slugs()
    gespeichert, ersetzt = 0, 0
    unbekannt: list[str] = []

    def store(name: str, content: bytes) -> None:
        nonlocal gespeichert, ersetzt
        path = Path(name)
        ext = path.suffix.lower()
        if ext not in config.IMAGE_EXTENSIONS:
            return
        # Dateiname trägt die file_id, ggf. noch mit Komma statt Unterstrich.
        stem = slug_for_file_id(path.stem)
        if stem not in known:
            if len(unbekannt) < 20:
                unbekannt.append(path.name)
            return
        dest = config.LOT_DIR / f"{stem}{ext}"
        existing = dest.exists()
        dest.write_bytes(content)
        if existing:
            ersetzt += 1
        else:
            gespeichert += 1

    for name, content in files:
        if Path(name).suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for info in zf.infolist():
                        if info.is_dir() or Path(info.filename).name.startswith("."):
                            continue
                        store(Path(info.filename).name, zf.read(info))
            except zipfile.BadZipFile:
                raise ValueError(f"'{name}' ist kein lesbares Zip-Archiv.")
        else:
            store(name, content)

    return {"gespeichert": gespeichert, "ersetzt": ersetzt,
            "unbekannt": len(unbekannt), "unbekannte_dateien": unbekannt}


# ---- Abfrage ----

def _lot_view(lot: dict, staging: dict[str, Path]) -> dict:
    """Anzeige inkl. Bild-Status für die API."""
    images = []
    for fid in lot["file_ids"]:
        slug = slug_for_file_id(fid)
        assigned = (lot.get("assigned_files") or {}).get(fid)
        if assigned:
            status = "zugeordnet"
        elif slug in staging:
            status = "vorhanden"
        else:
            status = "fehlt"
        images.append({"file_id": fid, "slug": slug, "status": status, "assigned_file": assigned})

    view = {f: lot.get(f, "") for f in LOT_FIELDS}
    view.update({
        "klasse": lot.get("klasse"),
        "assigned_at": lot.get("assigned_at"),
        "imported_at": lot.get("imported_at"),
        "images": images,
        "n_images": len(images),
        "n_fehlt": sum(1 for i in images if i["status"] == "fehlt"),
    })
    return view


def list_makes() -> list[dict]:
    """Marken aus der CSV (Vorfilter), mit Anzahl offener/gesamter Anzeigen."""
    counts: dict[str, dict] = {}
    for lot in _load().values():
        marke = lot.get("make") or "(ohne Marke)"
        entry = counts.setdefault(marke, {"make": marke, "gesamt": 0, "offen": 0})
        entry["gesamt"] += 1
        if not lot.get("klasse"):
            entry["offen"] += 1
    return sorted(counts.values(), key=lambda e: e["make"])


def list_lots(make: str | None = None, status: str = "offen", q: str = "",
              offset: int = 0, limit: int = 40) -> dict:
    """Anzeigen, gefiltert nach Marke und Zuordnungs-Status, immer als ganze
    Anzeige (nie einzelne Bilder). status: 'offen' | 'zugeordnet' | 'alle'."""
    staging = _staging_index()
    q = q.strip().lower()

    rows = []
    for lot in _load().values():
        if make and (lot.get("make") or "(ohne Marke)") != make:
            continue
        if status == "offen" and lot.get("klasse"):
            continue
        if status == "zugeordnet" and not lot.get("klasse"):
            continue
        if q and q not in " ".join(str(lot.get(f, "")) for f in LOT_FIELDS).lower():
            continue
        rows.append(lot)

    rows.sort(key=lambda l: (l.get("make", ""), l.get("model", ""), l.get("lid", "")))
    total = len(rows)
    page = [_lot_view(lot, staging) for lot in rows[offset:offset + limit]]
    return {"total": total, "offset": offset, "limit": limit, "lots": page}


def get_lot(lid: str) -> dict:
    lots = _load()
    if lid not in lots:
        raise KeyError(f"Anzeige '{lid}' nicht gefunden.")
    return _lot_view(lots[lid], _staging_index())


def stats() -> dict:
    lots = _load()
    staging = _staging_index()
    offen = [l for l in lots.values() if not l.get("klasse")]
    fehlend = sum(
        1 for l in offen for fid in l["file_ids"] if slug_for_file_id(fid) not in staging
    )
    return {
        "anzeigen_gesamt": len(lots),
        "anzeigen_offen": len(offen),
        "anzeigen_zugeordnet": len(lots) - len(offen),
        "bilder_fehlend": fehlend,
        "bilder_staging": len(staging),
    }


def staging_path(lid: str, slug: str) -> Path:
    """Pfad eines noch nicht zugeordneten Anzeigen-Bilds (für Vorschau/Inferenz)."""
    lots = _load()
    lot = lots.get(lid)
    if lot is None:
        raise KeyError(f"Anzeige '{lid}' nicht gefunden.")
    if slug not in {slug_for_file_id(fid) for fid in lot["file_ids"]}:
        raise KeyError(f"Bild '{slug}' gehört nicht zu Anzeige '{lid}'.")
    path = _staging_index().get(slug)
    if path is None:
        raise FileNotFoundError(f"Bild '{slug}' liegt nicht im Staging-Ordner.")
    return path


def staging_images(lid: str) -> list[tuple[str, Path]]:
    """(slug, Pfad) aller vorhandenen Bilder einer noch offenen Anzeige."""
    lots = _load()
    lot = lots.get(lid)
    if lot is None:
        raise KeyError(f"Anzeige '{lid}' nicht gefunden.")
    staging = _staging_index()
    return [(slug_for_file_id(fid), staging[slug_for_file_id(fid)])
            for fid in lot["file_ids"] if slug_for_file_id(fid) in staging]


# ---- Zuordnung ----

def assign(lid: str, klasse: str) -> dict:
    """Ordnet eine Anzeige einer Klasse zu: alle vorhandenen Bilder wandern aus
    dem Staging nach IMAGE_ROOT/<klasse>/ und bekommen die CSV-Spalten der
    Anzeige als Metadaten. Fehlende Bilder bleiben als fehlend vermerkt."""
    with _lock:
        lots = _load()
        lot = lots.get(lid)
        if lot is None:
            raise KeyError(f"Anzeige '{lid}' nicht gefunden.")
        if lot.get("klasse"):
            raise ValueError(f"Anzeige ist bereits Klasse '{lot['klasse']}' zugeordnet.")

        staging = _staging_index()
        vorhanden = [(fid, staging[slug_for_file_id(fid)])
                     for fid in lot["file_ids"] if slug_for_file_id(fid) in staging]
        if not vorhanden:
            raise ValueError("Kein einziges Bild dieser Anzeige liegt vor — zuerst Bilder hochladen.")

        lot_meta = {f: lot.get(f, "") for f in LOT_FIELDS}
        assigned_files = {}
        for fid, src in vorhanden:
            rel = image_service.register_lot_image(klasse, src, {**lot_meta, "file_id": fid})
            assigned_files[fid] = rel

        lot["klasse"] = klasse
        lot["assigned_at"] = datetime.now(timezone.utc).isoformat()
        lot["assigned_files"] = assigned_files
        _save(lots)

    return {"lid": lid, "klasse": klasse, "zugeordnete_bilder": len(assigned_files),
            "fehlende_bilder": len(lot["file_ids"]) - len(assigned_files)}


def unassign(lid: str) -> dict:
    """Macht eine Zuordnung rückgängig: die Bilder wandern zurück ins Staging und
    verschwinden wieder aus der Galerie. Der Store der Baureihe ist danach
    veraltet und sollte neu gebaut werden."""
    with _lock:
        lots = _load()
        lot = lots.get(lid)
        if lot is None:
            raise KeyError(f"Anzeige '{lid}' nicht gefunden.")
        if not lot.get("klasse"):
            raise ValueError("Anzeige ist keiner Klasse zugeordnet.")

        zurueck = 0
        for fid, rel in (lot.get("assigned_files") or {}).items():
            src = config.IMAGE_ROOT / rel
            if src.is_file():
                shutil.move(str(src), str(config.LOT_DIR / f"{slug_for_file_id(fid)}{src.suffix}"))
                zurueck += 1
            image_service.forget_image(rel)

        klasse = lot["klasse"]
        lot.update({"klasse": None, "assigned_at": None, "assigned_files": {}})
        _save(lots)

    return {"lid": lid, "vorherige_klasse": klasse, "zurueck_ins_staging": zurueck}


def delete_lot(lid: str) -> dict:
    """Verwirft eine noch nicht zugeordnete Anzeige samt ihrer Staging-Bilder."""
    with _lock:
        lots = _load()
        lot = lots.get(lid)
        if lot is None:
            raise KeyError(f"Anzeige '{lid}' nicht gefunden.")
        if lot.get("klasse"):
            raise ValueError("Anzeige ist zugeordnet — erst die Zuordnung aufheben.")

        staging = _staging_index()
        geloescht = 0
        for fid in lot["file_ids"]:
            path = staging.get(slug_for_file_id(fid))
            if path is not None:
                path.unlink()
                geloescht += 1

        del lots[lid]
        _save(lots)

    return {"lid": lid, "geloeschte_bilder": geloescht}
