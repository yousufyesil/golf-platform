"""@author Yousuf Yesil <yousufyesil@icloud.com>"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routers import images, infer, lots, models, pages

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Golf-Klassifikations-Plattform")

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

app.include_router(pages.router)
app.include_router(images.router)
app.include_router(lots.router)
app.include_router(models.router)
app.include_router(infer.router)
