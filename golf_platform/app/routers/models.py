"""@author Yousuf Yesil <yousufyesil@icloud.com>"""

from fastapi import APIRouter, HTTPException

from .. import store_service, umap_service

router = APIRouter(prefix="/api", tags=["models"])


@router.post("/baureihen/{baureihe}/rebuild")
def rebuild(baureihe: str):
    try:
        job_id = store_service.start_rebuild(baureihe)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"job_id": job_id}


@router.get("/baureihen/{baureihe}/umap")
def get_umap(baureihe: str):
    try:
        points = umap_service.project(baureihe)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"points": points}


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = store_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")
    return job
