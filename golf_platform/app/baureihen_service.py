"""Marke/Baureihe/Klasse-Hierarchie.

Jede Klasse (z.B. "GOLF VIII") gehört zu genau einer Baureihe (z.B. "Golf"),
jede Baureihe zu genau einer Marke (z.B. "Volkswagen"). Ein Embedding-Store wird
je Baureihe gebaut (store_<baureihe-slug>.*), nicht global — so lassen sich z.B.
ein Golf- und ein Polo-Store unabhängig voneinander verwalten und neu bauen.

Beim allerersten Start (keine baureihen.json vorhanden) werden alle zu diesem
Zeitpunkt vorhandenen Klassen automatisch der Baureihe "Golf" (Marke
"Volkswagen") zugeordnet — das deckt den bestehenden Datenbestand ab.

@author Yousuf Yesil <yousufyesil@icloud.com>
"""
import json
import threading

from . import config

ASSIGNMENTS_FILE = config.DATA_DIR / "baureihen.json"
_lock = threading.Lock()


def slug(baureihe: str) -> str:
    s = "".join(c.lower() if c.isalnum() else "_" for c in baureihe.strip())
    return s.strip("_") or "baureihe"


def _load() -> dict:
    if ASSIGNMENTS_FILE.exists():
        return json.loads(ASSIGNMENTS_FILE.read_text(encoding="utf-8"))

    from . import image_service  # lazy: image_service importiert baureihen_service selbst

    assignments = {
        row["name"]: {"marke": "Volkswagen", "baureihe": "Golf"}
        for row in image_service.list_classes()
    }
    _save(assignments)
    return assignments


def _save(assignments: dict) -> None:
    ASSIGNMENTS_FILE.write_text(json.dumps(assignments, indent=2, ensure_ascii=False), encoding="utf-8")


def baureihe_for_klasse(klasse: str) -> dict | None:
    return _load().get(klasse)


def add_klasse(marke: str, baureihe: str, klasse: str) -> None:
    marke, baureihe, klasse = marke.strip(), baureihe.strip(), klasse.strip()
    if not marke or not baureihe or not klasse:
        raise ValueError("Marke, Baureihe und Klasse dürfen nicht leer sein.")

    with _lock:
        assignments = _load()
        if klasse in assignments:
            existing = assignments[klasse]
            raise ValueError(
                f"Klasse '{klasse}' existiert bereits (Baureihe '{existing['baureihe']}', Marke '{existing['marke']}')."
            )
        assignments[klasse] = {"marke": marke, "baureihe": baureihe}
        _save(assignments)


def update_klasse(old_klasse: str, marke: str, baureihe: str, new_klasse: str) -> dict:
    """Ändert Marke/Baureihe-Zuordnung einer Klasse und benennt sie optional um.
    Umbenennen ist nur erlaubt, wenn die Klasse ausschließlich aus eigenen Uploads
    besteht — handgeprüfte train/test-Bilder hängen fest am bisherigen Ordnernamen
    (siehe verified_set.py) und würden sonst aus der Galerie verschwinden."""
    from . import image_service  # lazy

    old_klasse = old_klasse.strip()
    marke, baureihe, new_klasse = marke.strip(), baureihe.strip(), new_klasse.strip()
    if not marke or not baureihe or not new_klasse:
        raise ValueError("Marke, Baureihe und Klasse dürfen nicht leer sein.")

    with _lock:
        assignments = _load()
        if old_klasse not in assignments:
            raise ValueError(f"Klasse '{old_klasse}' ist keiner Baureihe zugeordnet.")

        renaming = new_klasse != old_klasse
        if renaming:
            if new_klasse in assignments:
                raise ValueError(f"Klasse '{new_klasse}' existiert bereits.")
            if image_service.has_verified_images(old_klasse):
                raise ValueError(
                    f"Klasse '{old_klasse}' enthält handgeprüfte train/test-Bilder und kann "
                    "deshalb nicht umbenannt werden. Marke/Baureihe lassen sich trotzdem ändern."
                )
            image_service.rename_class_folder(old_klasse, new_klasse)

        info = assignments.pop(old_klasse)
        info["marke"], info["baureihe"] = marke, baureihe
        assignments[new_klasse] = info
        _save(assignments)

    return {"marke": marke, "baureihe": baureihe, "klasse": new_klasse}


def remove_klasse(klasse: str) -> dict:
    """Entfernt die Baureihen-Zuordnung einer Klasse und löscht deren hochgeladene
    Bilder + Metadaten. Verifizierte train/test-Bilder bleiben auf der Platte
    (geteilte Daten) — die Klasse verschwindet nur aus der verwalteten Galerie.
    Gibt die entfernte Zuordnung zurück."""
    from . import image_service  # lazy

    with _lock:
        assignments = _load()
        if klasse not in assignments:
            raise ValueError(f"Klasse '{klasse}' ist keiner Baureihe zugeordnet.")
        removed = assignments.pop(klasse)
        _save(assignments)

    image_service.delete_class_uploads(klasse)
    return removed


def remove_baureihe(baureihe: str) -> list[str]:
    """Entfernt eine ganze Baureihe: alle zugehörigen Klassen (inkl. deren Uploads)
    und den Store der Baureihe. Gibt die Liste der entfernten Klassen zurück."""
    from . import image_service, store_service  # lazy

    if baureihe == config.GLOBAL_BAUREIHE:
        raise ValueError(
            f"'{config.GLOBAL_BAUREIHE}' ist der Store über alle Marken/Baureihen hinweg "
            "und keine echte Baureihe — kann nicht gelöscht werden."
        )

    klassen = klassen_for_baureihe(baureihe)
    if not klassen:
        raise ValueError(f"Baureihe '{baureihe}' existiert nicht.")

    with _lock:
        assignments = _load()
        for k in klassen:
            assignments.pop(k, None)
        _save(assignments)

    for k in klassen:
        image_service.delete_class_uploads(k)
    store_service.delete_store(baureihe)
    return klassen


def list_baureihen() -> list[dict]:
    """Gruppiert: [{'marke', 'baureihe', 'klassen': ['GOLF I', ...]}], sortiert."""
    assignments = _load()
    groups: dict[tuple[str, str], list[str]] = {}
    for klasse, info in assignments.items():
        key = (info["marke"], info["baureihe"])
        groups.setdefault(key, []).append(klasse)
    return [
        {"marke": marke, "baureihe": baureihe, "klassen": sorted(klassen)}
        for (marke, baureihe), klassen in sorted(groups.items(), key=lambda kv: kv[0])
    ]


def klassen_for_baureihe(baureihe: str) -> list[str]:
    return [k for k, info in _load().items() if info["baureihe"] == baureihe]


def list_baureihen_detailed() -> list[dict]:
    """list_baureihen() angereichert mit Bild-Anzahl je Klasse und Store-Status,
    plus einem vorangestellten Sonder-Eintrag für config.GLOBAL_BAUREIHE ("Gesamt"):
    ein Store über alle Klassen aller Marken/Baureihen hinweg (is_global=True)."""
    from . import image_service, store_service  # lazy: store_service importiert dieses Modul selbst

    counts = {row["name"]: row["count"] for row in image_service.list_classes()}
    result = []
    for group in list_baureihen():
        klassen = [{"name": k, "count": counts.get(k, 0)} for k in group["klassen"]]
        result.append({
            "marke": group["marke"],
            "baureihe": group["baureihe"],
            "klassen": klassen,
            "bilder_gesamt": sum(k["count"] for k in klassen),
            "store": store_service.get_status(group["baureihe"]),
            "is_global": False,
        })

    alle_klassen = [{"name": k, "count": c} for k, c in sorted(counts.items())]
    result.insert(0, {
        "marke": "Alle Marken",
        "baureihe": config.GLOBAL_BAUREIHE,
        "klassen": alle_klassen,
        "bilder_gesamt": sum(counts.values()),
        "store": store_service.get_status(config.GLOBAL_BAUREIHE),
        "is_global": True,
    })
    return result
