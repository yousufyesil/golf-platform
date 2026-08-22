"""Whitelist der handgeprüften Bilder aus train_data.json + test_data.json.

golf_images/golf_images enthält tausende roh gescrapte Bilder, aber nur die dort
referenzierten (~325) sind manuell geprüft. Über die Plattform hochgeladene Bilder
gelten ebenfalls als geprüft — der Nutzer ordnet sie beim Upload selbst einer
Klasse zu (siehe image_service.add_image). Alles andere auf der Platte bleibt
für Galerie-Anzeige und Store-Build unsichtbar.

@author Yousuf Yesil <yousufyesil@icloud.com>
"""
import json

from . import config

_SPLIT_FILES = {"train": "train_data.json", "test": "test_data.json"}


def class_to_folder(cls: str) -> str:
    """JSON-Key 'GOLF_III Cabriolet' -> Ordnername 'GOLF III Cabriolet' (siehe build_store.py)."""
    return cls.replace("_", " ", 1)


def verified_index() -> dict[str, dict]:
    """relpath ('Klasse/Datei.webp') -> {'class', 'perspektive', 'source', 'lid', 'file_id'}
    für alle in train_data.json/test_data.json gelisteten, handgeprüften Bilder."""
    index: dict[str, dict] = {}
    for source, filename in _SPLIT_FILES.items():
        path = config.REPO_ROOT / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for cls, perspektiven in data.items():
            folder = class_to_folder(cls)
            for persp, eintraege in perspektiven.items():
                for e in eintraege:
                    rel = f"{folder}/{e['fid']}.webp"
                    index[rel] = {"class": cls, "perspektive": persp, "source": source,
                                  "lid": e.get("lid", ""), "file_id": e["fid"]}
    return index
