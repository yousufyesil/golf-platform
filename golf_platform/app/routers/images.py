"""@author Yousuf Yesil <yousufyesil@icloud.com>"""

import csv
import io
import zipfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from .. import baureihen_service, config, image_service, lot_service

router = APIRouter(prefix="/api", tags=["images"])


@router.get("/classes")
def get_classes():
    return image_service.list_classes()


@router.get("/baureihen")
def get_baureihen():
    return baureihen_service.list_baureihen_detailed()


@router.post("/classes")
def create_klasse(
    marke: str = Form(...),
    baureihe: str = Form(...),
    klasse: str = Form(...),
):
    try:
        baureihen_service.add_klasse(marke, baureihe, klasse)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "created", "marke": marke, "baureihe": baureihe, "klasse": klasse}


@router.put("/classes/{klasse}")
def update_klasse(
    klasse: str,
    marke: str = Form(...),
    baureihe: str = Form(...),
    new_klasse: str = Form(...),
):
    try:
        result = baureihen_service.update_klasse(klasse, marke, baureihe, new_klasse)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "updated", **result}


@router.delete("/classes/{klasse}")
def delete_klasse(klasse: str):
    try:
        baureihen_service.remove_klasse(klasse)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": "deleted"}


@router.delete("/baureihen/{baureihe}")
def delete_baureihe(baureihe: str):
    try:
        removed = baureihen_service.remove_baureihe(baureihe)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": "deleted", "klassen": removed}


@router.get("/classes/{klasse}/images")
def get_images(klasse: str):
    return image_service.list_images(klasse)


@router.post("/images")
async def upload_images(
    files: list[UploadFile] = File(...),
    klasse: str = Form(...),
    perspektive: str = Form(""),
):
    saved, errors = [], []
    for f in files:
        try:
            content = await f.read()
            saved.append(image_service.add_image(klasse, f.filename, content, perspektive or None))
        except ValueError as e:
            errors.append({"filename": f.filename, "error": str(e)})
    return {"saved": saved, "errors": errors}


@router.delete("/classes/{klasse}/images/{filename}")
def remove_image(klasse: str, filename: str):
    try:
        image_service.delete_image(klasse, filename)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": "deleted"}


@router.get("/classes/{klasse}/images/{filename}/file")
def get_image_file(klasse: str, filename: str):
    try:
        path = image_service.image_path(klasse, filename)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    return FileResponse(path)


def _zip_response(rows: list[dict], archive_name: str) -> StreamingResponse:
    """Baut ein Zip aus den Bild-Dateien (in Unterordnern je Klasse) plus einer
    images.csv (file,class,perspektive,quelle,lid,file_id + Anzeigen-Spalten aus dem
    Lot-Import, sofern vorhanden) und streamt es als Download."""
    # "lid" steckt sowohl top-level (aus train/test_data.json) als auch in LOT_FIELDS
    # (aus dem Lot-Import) -- hier einmalig fuehren, damit die Spalte nicht doppelt auftaucht.
    lot_extra_fields = [f for f in lot_service.LOT_FIELDS if f != "lid"]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["file", "class", "perspektive", "quelle", "lid", "file_id", *lot_extra_fields])
        for r in rows:
            arcname = f"{r['class']}/{r['filename']}"
            zf.write(r["path"], arcname)
            lot = r.get("lot") or {}
            writer.writerow([
                arcname, r["class"], r["perspektive"], r["quelle"], r.get("lid", ""), r.get("file_id", ""),
                *[lot.get(f, "") for f in lot_extra_fields],
            ])
        zf.writestr("images.csv", csv_buf.getvalue())

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{archive_name}.zip"'},
    )


@router.get("/classes/{klasse}/download")
def download_class(klasse: str):
    rows = [r for r in image_service.iter_included_files() if r["class"] == klasse]
    if not rows:
        raise HTTPException(status_code=404, detail=f"Keine Bilder in Klasse '{klasse}'.")
    return _zip_response(rows, f"klasse_{klasse}")


@router.get("/baureihen/{baureihe}/download")
def download_baureihe(baureihe: str):
    if baureihe == config.GLOBAL_BAUREIHE:
        rows = image_service.iter_included_files()
    else:
        klassen = set(baureihen_service.klassen_for_baureihe(baureihe))
        if not klassen:
            raise HTTPException(status_code=404, detail=f"Baureihe '{baureihe}' existiert nicht.")
        rows = [r for r in image_service.iter_included_files() if r["class"] in klassen]
    if not rows:
        raise HTTPException(status_code=404, detail=f"Keine Bilder in Baureihe '{baureihe}'.")
    return _zip_response(rows, f"baureihe_{baureihe}")
