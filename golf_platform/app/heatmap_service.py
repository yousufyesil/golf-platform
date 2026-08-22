"""Attention-Heatmap für ein einzelnes Bild — visualisiert, welche Bildregionen
am stärksten in die vom Encoder erzeugte Embedding eingeflossen sind (also in die
kNN-Klassifikation). Nutzt denselben Encoder-Singleton wie inference_service,
daher unabhängig von Baureihe/Store.

Farbskala: dieselbe sequentielle Blau-Rampe (100..700) wie der Rest der Plattform
(--series-1 in style.css, s. dataviz-Skill-Palette) — eine Farbe, hell->dunkel
nach Attention-Stärke, statt eines Regenbogen-Colormaps. Bei Attention 0 ist der
Overlay vollständig transparent (Originalbild unverändert sichtbar).

@author Yousuf Yesil <yousufyesil@icloud.com>
"""
import io

import numpy as np
from PIL import Image

from . import inference_service

_RAMP_HEX = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
_RAMP = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in _RAMP_HEX], dtype=np.float32)

_MAX_ALPHA = 0.70  # Deckkraft des Overlays bei maximaler Attention (0 = unverändertes Originalbild)


def _colorize(weights: np.ndarray) -> np.ndarray:
    """weights: (H, W) float in [0,1] -> (H, W, 3) float RGB entlang der Blau-Rampe."""
    idx_f = weights * (len(_RAMP) - 1)
    idx0 = np.clip(idx_f.astype(int), 0, len(_RAMP) - 2)
    frac = (idx_f - idx0)[..., None]
    return _RAMP[idx0] * (1 - frac) + _RAMP[idx0 + 1] * frac


def generate(image: Image.Image) -> bytes | None:
    """PNG-Bytes des Originalbilds mit blauem Attention-Overlay, oder None, wenn
    der aktuell konfigurierte Encoder keine Attention-Heatmap unterstützt."""
    encoder = inference_service.get_encoder()
    image = image.convert("RGB")
    grid = encoder.attention_map(image)
    if grid is None:
        return None

    w, h = image.size
    grid_img = Image.fromarray((grid * 255).astype(np.uint8), mode="L").resize((w, h), Image.BICUBIC)
    weights = np.clip(np.asarray(grid_img, dtype=np.float32) / 255.0, 0.0, 1.0)

    overlay = _colorize(weights)
    alpha = (weights * _MAX_ALPHA)[..., None]
    base = np.asarray(image, dtype=np.float32)
    blended = np.clip(base * (1 - alpha) + overlay * alpha, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(blended, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()
