"""Klassifiziert die Bilder einer Kleinanzeigen-Anzeige direkt per Link.

Lädt die Anzeigen-Seite herunter, extrahiert die Galerie-Bild-URLs aus den
eingebetteten schema.org ld+json-Blöcken (contentUrl je Bild) und lädt die
Bilder, damit sie wie hochgeladene Ansichten einer Baureihe durch
inference_service.run_inference laufen können. Kein HTML-Parser (bs4/lxml)
nötig — die Bild- und Titel-Werte stecken als einfache JSON-Strings in der
Seite und lassen sich robust per Regex herausziehen.

@author Yousuf Yesil <yousufyesil@icloud.com>
"""
import re
from io import BytesIO
from urllib.parse import urlparse

import httpx
from PIL import Image

from . import config

# Nur kleinanzeigen.de selbst wird angefragt (kein offener URL-Fetcher) -> SSRF-Fläche
# auf eine feste, vertrauenswürdige Domain begrenzt.
_ALLOWED_HOSTS = {"www.kleinanzeigen.de", "kleinanzeigen.de"}
_ALLOWED_IMAGE_HOSTS = {"img.kleinanzeigen.de"}
_MAX_IMAGES = 20
_TIMEOUT = 20.0
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; golf-platform-bot/1.0)"}

_CONTENT_URL_RE = re.compile(r'"contentUrl"\s*:\s*"(https://img\.kleinanzeigen\.de/[^"]+)"')
_TITLE_RE = re.compile(r'"title"\s*:\s*"([^"]+)"')


def _validate_ad_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or host not in _ALLOWED_HOSTS:
        raise ValueError("Nur Anzeigen-Links von kleinanzeigen.de werden unterstützt.")
    return url


def _extract_image_urls(html: str) -> list[str]:
    seen: set[str] = set()
    urls = []
    for m in _CONTENT_URL_RE.finditer(html):
        url = m.group(1)
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _extract_title(html: str) -> str | None:
    m = _TITLE_RE.search(html)
    return m.group(1) if m else None


def fetch_ad_images(ad_url: str) -> dict:
    """Lädt eine Kleinanzeigen-Anzeige und ihre Galerie-Bilder.

    Gibt {"images": [(name, PIL.Image), ...], "image_urls": [...], "ad_title": str|None}
    zurück. "images" wird direkt an inference_service.run_inference übergeben,
    "image_urls" dient dem Frontend als Vorschau (öffentliche kleinanzeigen.de-Bild-URLs,
    kein Proxy nötig).
    """
    ad_url = _validate_ad_url(ad_url)
    with httpx.Client(
        headers=_HEADERS, follow_redirects=True, timeout=_TIMEOUT, proxy=config.KLEINANZEIGEN_PROXY
    ) as client:
        resp = client.get(ad_url)
        resp.raise_for_status()
        html = resp.text

        image_urls = _extract_image_urls(html)[:_MAX_IMAGES]
        if not image_urls:
            raise ValueError("Keine Bilder in dieser Anzeige gefunden.")

        images = []
        for i, url in enumerate(image_urls):
            r = client.get(url)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            images.append((f"bild_{i + 1}.jpg", img))

    return {"images": images, "image_urls": image_urls, "ad_title": _extract_title(html)}


def fetch_image(url: str) -> Image.Image:
    """Lädt ein einzelnes öffentliches Kleinanzeigen-Bild — für die Attention-Heatmap
    eines Anzeigen-Bilds in der Dashboard-Detailansicht, ohne die ganze Anzeige (die
    ggf. gesperrte Host-IP-Probleme hat, s. _ALLOWED_HOSTS) neu zu laden."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or host not in _ALLOWED_IMAGE_HOSTS:
        raise ValueError("Nicht unterstützte Bild-Quelle.")
    with httpx.Client(headers=_HEADERS, follow_redirects=True, timeout=_TIMEOUT, proxy=config.KLEINANZEIGEN_PROXY) as client:
        r = client.get(url)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
