# Golf-Klassifikations-Plattform

FastAPI-Webanwendung zur Klassifikation von VW-Golf-Baureihen (und weiteren
Marken/Baureihen) per Bild-Embedding + kNN, mit Verwaltung von Trainingsdaten
(Galerie, Lot-Import, Modell-Neubau, UMAP-Cluster-Ansicht).

Extrahiert aus dem größeren Forschungs-Repo der zugehörigen Bachelorarbeit
(SigLIP2/DINOv3/DINOv2-Backbone-Vergleich). Dieses Repo enthält nur den
Anwendungscode, keine Trainings-/Evaluationsskripte.

## Struktur

```
EmbeddingStore.py       # Datenstruktur für Embeddings + Metadaten (Speichern/Laden, kNN-Suche)
golf_platform/
  app/                  # FastAPI-Anwendung (Router, Services, Templates, Static)
  Dockerfile            # Erwartet golf_images/, stores/, data/ als Volume-Mounts zur Laufzeit
  requirements.txt
```

`EmbeddingStore.py` liegt bewusst im Repo-Root (nicht im `golf_platform`-Paket),
weil `golf_platform/app/config.py` es zur Laufzeit per `sys.path.insert` aus
dem Repo-Root importierbar macht.

## Laufzeitdaten (nicht Teil dieses Repos)

Bilder, vorgebaute Embedding-Stores und Laufzeitzustand (Klassen-Zuordnungen,
Upload-Metadaten, Lot-Importe) liegen außerhalb dieses Repos und werden über
Umgebungsvariablen eingebunden:

| Variable | Zweck | Default |
|---|---|---|
| `GOLF_REPO_ROOT` | Basis für den `EmbeddingStore.py`-Import | zwei Ebenen über `config.py` |
| `GOLF_IMAGE_ROOT` | Bild-Galerie (`<Klasse>/<Datei>`) | `<REPO_ROOT>/golf_images/golf_images` |
| `GOLF_STORE_DIR` | Embedding-Stores (`store_<baureihe>.{npy,parquet,json}`) | `<REPO_ROOT>/stores` |
| `GOLF_DATA_DIR` | `baureihen.json`, `metadata.json` | `<REPO_ROOT>/golf_platform/data` |
| `GOLF_LOT_DIR` | Staging-Ordner für Lot-Import-Bilder | `<REPO_ROOT>/lot_staging` |
| `GOLF_GPU` | CUDA-Geräteindex (`cpu` für CPU-Betrieb) | `0` |
| `GOLF_MODEL_ID` | HuggingFace-Encoder-ID | `google/siglip2-so400m-patch16-384` |

## Lokal starten

```bash
pip install -r golf_platform/requirements.txt torch
export GOLF_IMAGE_ROOT=/pfad/zu/golf_images/golf_images
export GOLF_STORE_DIR=/pfad/zu/stores
export GOLF_DATA_DIR=/pfad/zu/golf_platform/data
uvicorn golf_platform.app.main:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -f golf_platform/Dockerfile -t golf-platform .
docker run --rm --gpus all -p 8000:8000 \
  -v /pfad/zu/den/daten:/data \
  -e GOLF_IMAGE_ROOT=/data/golf_images/golf_images \
  -e GOLF_STORE_DIR=/data/stores \
  -e GOLF_DATA_DIR=/data/golf_platform/data \
  golf-platform
```
