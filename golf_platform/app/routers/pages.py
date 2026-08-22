"""@author Yousuf Yesil <yousufyesil@icloud.com>"""

import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
# Cache-Buster für /static/*: ändert sich bei jedem Prozessstart, damit Browser
# (v.a. mobile Safari) nach einem Deploy nicht auf altem style.css/app.js hängen bleiben.
templates.env.globals["asset_version"] = int(time.time())


@router.get("/")
def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")


@router.get("/images")
def images_page(request: Request):
    return templates.TemplateResponse(request, "images.html")


@router.get("/lots")
def lots_page(request: Request):
    return templates.TemplateResponse(request, "lots.html")


@router.get("/models")
def models_page(request: Request):
    return templates.TemplateResponse(request, "models.html")


@router.get("/infer")
def infer_page(request: Request):
    return templates.TemplateResponse(request, "infer.html")
