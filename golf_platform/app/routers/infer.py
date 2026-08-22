"""@author Yousuf Yesil <yousufyesil@icloud.com>"""

import io

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from .. import heatmap_service, inference_service, kleinanzeigen_service

router = APIRouter(prefix="/api", tags=["infer"])


@router.post("/infer")
async def infer(
    files: list[UploadFile] = File(...),
    baureihe: str = Form(...),
    k: int = Form(5),
    metric: str = Form("cosine"),
    group_mode: bool = Form(False),
):
    images = []
    for f in files:
        content = await f.read()
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail=f"Konnte '{f.filename}' nicht als Bild lesen.")
        images.append((f.filename, img))

    if not images:
        raise HTTPException(status_code=400, detail="Keine Bilder übermittelt.")

    try:
        return inference_service.run_inference(baureihe, images, k=k, metric=metric, group_mode=group_mode)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/infer/kleinanzeigen")
def infer_kleinanzeigen(
    url: str = Form(...),
    baureihe: str = Form(...),
    k: int = Form(5),
    metric: str = Form("cosine"),
):
    """Lädt alle Galerie-Bilder einer Kleinanzeigen-Anzeige per Link und bewertet
    sie gemeinsam (wie mehrere Ansichten derselben Anzeige) gegen eine Baureihe.

    Kein 'async def': fetch_ad_images macht blockierendes Netzwerk-IO (httpx sync
    Client) — FastAPI führt sync-Endpunkte automatisch in einem Threadpool aus,
    statt den Event-Loop zu blockieren.
    """
    try:
        ad = kleinanzeigen_service.fetch_ad_images(url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Anzeige konnte nicht geladen werden: {e}")

    try:
        result = inference_service.run_inference(baureihe, ad["images"], k=k, metric=metric, group_mode=True)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    result["image_urls"] = ad["image_urls"]
    result["ad_title"] = ad["ad_title"]
    return result


@router.post("/heatmap")
def heatmap(
    file: UploadFile | None = File(None),
    image_url: str | None = Form(None),
):
    """Attention-Heatmap für EIN Bild — wird von der 'Erweitert'-Ansicht auf
    Abruf nachgeladen (nicht vorab für jedes Ergebnis berechnet). Entweder eine
    lokal hochgeladene Datei (dropzone/infer-Upload) oder eine bereits bekannte
    Kleinanzeigen-Bild-URL (aus /api/infer/kleinanzeigen); genau eines von beiden.

    Kein 'async def' aus demselben Grund wie /infer/kleinanzeigen: sowohl der
    optionale Netzwerk-Abruf als auch die GPU-Inferenz blockieren.
    """
    if file is not None:
        content = file.file.read()
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail=f"Konnte '{file.filename}' nicht als Bild lesen.")
    elif image_url:
        try:
            image = kleinanzeigen_service.fetch_image(image_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Bild konnte nicht geladen werden: {e}")
    else:
        raise HTTPException(status_code=400, detail="Weder Datei noch image_url übermittelt.")

    png = heatmap_service.generate(image)
    if png is None:
        raise HTTPException(status_code=422, detail="Attention-Heatmap für dieses Modell nicht verfügbar.")
    return Response(content=png, media_type="image/png")
