"""@author Yousuf Yesil <yousufyesil@icloud.com>"""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from .. import config, inference_service, lot_service

router = APIRouter(prefix="/api/lots", tags=["lots"])


@router.get("/stats")
def get_stats():
    return lot_service.stats()


@router.get("/makes")
def get_makes():
    return lot_service.list_makes()


@router.get("")
def get_lots(make: str = "", status: str = "offen", q: str = "", offset: int = 0, limit: int = 40):
    return lot_service.list_lots(make=make or None, status=status, q=q, offset=offset, limit=limit)


@router.post("/csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        return lot_service.import_csv(await file.read())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/images")
async def upload_lot_images(files: list[UploadFile] = File(...)):
    """Bilder zu bereits importierten Anzeigen. Dateiname muss die file_id sein
    (Komma oder Unterstrich als Trenner); Zip-Archive werden entpackt."""
    payload = [(f.filename or "", await f.read()) for f in files]
    try:
        return lot_service.import_images(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{lid}")
def get_lot(lid: str):
    try:
        return lot_service.get_lot(lid)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{lid}/images/{slug}/file")
def get_lot_image(lid: str, slug: str):
    try:
        return FileResponse(lot_service.staging_path(lid, slug))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{lid}/suggest")
def suggest_klasse(lid: str, baureihe: str = Form(config.GLOBAL_BAUREIHE), k: int = Form(config.DEFAULT_K),
                   metric: str = Form("cosine")):
    """Klassen-Vorschlag für eine Anzeige: alle vorhandenen Bilder der Anzeige
    laufen gemeinsam (group_mode) gegen den Store der gewählten Baureihe.

    Kein 'async def': die GPU-Inferenz blockiert — FastAPI führt sync-Endpunkte
    im Threadpool aus, statt den Event-Loop anzuhalten.
    """
    try:
        images = lot_service.staging_images(lid)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not images:
        raise HTTPException(status_code=400, detail="Kein Bild dieser Anzeige liegt vor.")

    loaded = [(slug, Image.open(path).convert("RGB")) for slug, path in images]
    try:
        return inference_service.run_inference(baureihe, loaded, k=k, metric=metric, group_mode=True)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{lid}/assign")
def assign_lot(lid: str, klasse: str = Form(...)):
    try:
        return lot_service.assign(lid, klasse)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{lid}/unassign")
def unassign_lot(lid: str):
    try:
        return lot_service.unassign(lid)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{lid}")
def delete_lot(lid: str):
    try:
        return lot_service.delete_lot(lid)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
