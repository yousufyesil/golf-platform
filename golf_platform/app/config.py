"""Zentrale Pfad- und Umgebungskonfiguration der Plattform.

@author Yousuf Yesil <yousufyesil@icloud.com>
"""
import os
import sys
from pathlib import Path

# Repo-Root: bei "uvicorn golf_platform.app.main:app" aus bachelor_arbeit/ heraus
# oder im Docker-Image (siehe Dockerfile) ist REPO_ROOT der Ort, an dem EmbeddingStore.py
# und die store_*.{npy,parquet} Dateien liegen.
REPO_ROOT = Path(os.environ.get("GOLF_REPO_ROOT", Path(__file__).resolve().parents[2]))

# Bild-Galerie. Für schnelle lokale Tests kann GOLF_IMAGE_ROOT auf einen kleinen
# Testordner zeigen, ohne die 17k-Bilder anzufassen.
IMAGE_ROOT = Path(os.environ.get("GOLF_IMAGE_ROOT", REPO_ROOT / "golf_images" / "golf_images"))

# Wo Stores (store_<short>.npy/.parquet/.json) gelesen/geschrieben werden.
# Eigener Ordner statt Repo-Root, damit Stores (auch die aus build_store.py)
# an einem Ort gebuendelt sind.
STORE_DIR = Path(os.environ.get("GOLF_STORE_DIR", REPO_ROOT / "stores"))
STORE_DIR.mkdir(parents=True, exist_ok=True)

# Zusatzmetadaten (Perspektive, added_at) für Bilder, keyed by Pfad relativ zu IMAGE_ROOT.
DATA_DIR = Path(os.environ.get("GOLF_DATA_DIR", REPO_ROOT / "golf_platform" / "data"))
METADATA_FILE = DATA_DIR / "metadata.json"

# Lot-Import (lot.csv + zugehörige Bilder): Anzeigen, die noch keiner Klasse
# zugeordnet sind. Die Bilder liegen hier ausserhalb von IMAGE_ROOT, damit sie
# nicht schon vor der Zuordnung in Galerie und Store-Build auftauchen; bei der
# Zuordnung wandern sie nach IMAGE_ROOT/<KLASSE>/ (siehe lot_service.py).
LOT_DIR = Path(os.environ.get("GOLF_LOT_DIR", REPO_ROOT / "lot_staging"))
LOTS_FILE = DATA_DIR / "lots.json"

# GPU-Pinning wie in app.py: GOLF_GPU=cpu für CPU-Betrieb, sonst CUDA-Geräteindex.
# Muss gesetzt werden, BEVOR irgendein Modul torch importiert -> config immer zuerst importieren.
GPU = os.environ.get("GOLF_GPU", "0")
if GPU.lower() != "cpu":
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU)

BATCH_SIZE = int(os.environ.get("GOLF_BATCH_SIZE", "16"))
DEFAULT_K = 5

# Die Plattform verwaltet bewusst nur EINEN Encoder/Store (wie app.py bisher),
# keinen Zoo aus Modellen -> überschreibbar für Docker-Deployments mit anderem Modell.
MODEL_ID = os.environ.get("GOLF_MODEL_ID", "google/siglip2-so400m-patch16-384")

# Sonder-"Baureihe": steht für einen Store über alle Marken/Baureihen hinweg
# (siehe baureihen_service.list_baureihen_detailed), nicht für eine echte Baureihe
# aus baureihen.json. Wird wie jede andere Baureihe über store_service gebaut/geladen.
GLOBAL_BAUREIHE = "Gesamt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Optionaler Proxy nur für den Kleinanzeigen-Link-Abruf (kleinanzeigen_service.py),
# z.B. "http://user:pass@host:port" oder "socks5://host:port" (letzteres braucht
# das Extra httpx[socks]). Nötig, wenn die Server-IP selbst von kleinanzeigen.de
# gesperrt ist (eigenständig von evtl. global gesetzten HTTP(S)_PROXY-Variablen).
KLEINANZEIGEN_PROXY = os.environ.get("GOLF_KLEINANZEIGEN_PROXY") or None

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOT_DIR.mkdir(parents=True, exist_ok=True)

# EmbeddingStore.py liegt im Repo-Root (nicht Teil des golf_platform-Pakets) -> importierbar machen.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
