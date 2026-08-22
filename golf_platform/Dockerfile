# Build-Context ist das Repo-Root (bachelor_arbeit/), da EmbeddingStore.py dort liegt:
#   docker build -f golf_platform/Dockerfile -t golf-platform .
#
# golf_images/, store_*.{npy,parquet,json} und golf_platform/data/ werden NICHT ins Image
# kopiert, sondern zur Laufzeit als Volumes gemountet (siehe README-Hinweise unten) -
# Bild bleibt schlank, Daten bleiben persistent/austauschbar.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY golf_platform/requirements.txt ./golf_platform/requirements.txt
RUN pip install --no-cache-dir -r golf_platform/requirements.txt

COPY EmbeddingStore.py ./EmbeddingStore.py
COPY golf_platform/__init__.py ./golf_platform/__init__.py
COPY golf_platform/app ./golf_platform/app

ENV GOLF_REPO_ROOT=/app \
    GOLF_IMAGE_ROOT=/app/golf_images/golf_images \
    GOLF_STORE_DIR=/app \
    GOLF_DATA_DIR=/app/golf_platform/data \
    GOLF_GPU=0

EXPOSE 8000

CMD ["uvicorn", "golf_platform.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Beispiel-Run (GPU-Passthrough, ein Mount für den gesamten Datenbestand):
#   docker run --rm --gpus all -p 8000:8000 \
#     -v $(pwd):/data \
#     -e GOLF_IMAGE_ROOT=/data/golf_images/golf_images \
#     -e GOLF_STORE_DIR=/data \
#     -e GOLF_DATA_DIR=/data/golf_platform/data \
#     golf-platform
# So bleiben golf_images/, store_*.{npy,parquet,json} und metadata.json auf dem Host
# persistent und werden nicht ins Image gebacken.
