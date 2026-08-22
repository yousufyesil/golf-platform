"""Vision-Encoder laden — konsolidiert aus build_store.py und app.py.

build_store.py nutzte für alle "google/siglip*"-IDs immer SiglipVisionModel; app.py
erkennt dagegen über AutoConfig, dass manche "siglip2"-Checkpoints intern noch
v1-Architektur sind. Diese genauere Erkennung wird hier für Store-Build UND
Inferenz gemeinsam genutzt.

@author Yousuf Yesil <yousufyesil@icloud.com>
"""
from . import config  # muss vor torch importiert werden (setzt CUDA_VISIBLE_DEVICES)

import threading

import numpy as np
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection

# Rebuild-Jobs und Inferenz-Requests teilen sich dieselbe GPU -> alle Forward-Passes
# über diesen Lock serialisieren, damit keine parallelen CUDA-Calls kollidieren.
GPU_LOCK = threading.Lock()


def resolve_device() -> str:
    if config.GPU.lower() == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def short_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def _normalize_attention_grid(grid: np.ndarray) -> np.ndarray:
    """0..1-Normierung der rohen Attention-Gewichte, robust gegen zwei bekannte
    Ausreißer-Effekte bei ViTs ohne explizite Register-Tokens (s. Darcet et al.,
    "Vision Transformers Need Registers", 2023): der äußerste Patch-Rand dient
    systematisch als 'Attention-Sink' für globale, bildunabhängige Information
    und wird deshalb ausgeblendet (auf den Bildinneren-Minimalwert gesetzt statt
    einbezogen); zusätzlich wird das obere 2%-Perzentil im Bildinneren gekappt,
    damit nicht ein einzelner Ausreißer-Patch den restlichen Kontrast plättet.
    Ohne diese beiden Schritte zeigt die Heatmap fast ausschließlich den Bildrand
    statt des eigentlichen Objekts (empirisch verifiziert)."""
    grid = grid.astype(np.float32).copy()
    interior = grid[1:-1, 1:-1]
    if interior.size == 0:
        interior = grid

    floor = float(interior.min())
    ceiling = float(np.percentile(interior, 98))
    if ceiling <= floor:
        ceiling = float(interior.max()) if interior.max() > floor else floor + 1e-6

    grid[0, :] = grid[-1, :] = floor
    grid[:, 0] = grid[:, -1] = floor
    grid = np.clip(grid, floor, ceiling)

    span = ceiling - floor
    return (grid - floor) / span if span > 1e-12 else np.zeros_like(grid)


class Encoder:
    """Lädt Processor + Modell für eine model_id und embedded PIL-Bilder."""

    def __init__(self, model_id: str, device: str | None = None):
        self.model_id = model_id
        self.short = short_name(model_id)
        self.device = device or resolve_device()
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self._model, self._forward = self._load(model_id)
        self._attn_kind = self._resolve_attention_kind(model_id)

    def _load(self, model_id: str):
        if model_id.startswith("facebook/dinov2"):
            model = AutoModel.from_pretrained(model_id).eval().to(self.device)
            return model, lambda inputs: model(**inputs).pooler_output

        if model_id.startswith("openai/clip"):
            model = CLIPVisionModelWithProjection.from_pretrained(model_id).eval().to(self.device)
            return model, lambda inputs: model(**inputs).image_embeds

        if "siglip" in model_id:
            cfg = AutoConfig.from_pretrained(model_id)
            if getattr(cfg, "model_type", "") == "siglip2":
                from transformers import Siglip2VisionModel as VisionModel
            else:
                from transformers import SiglipVisionModel as VisionModel
            model = VisionModel.from_pretrained(model_id).eval().to(self.device)
            return model, lambda inputs: model(**inputs).pooler_output

        raise ValueError(f"Unbekannter Encoder: {model_id!r}")

    @torch.no_grad()
    def embed(self, images) -> np.ndarray:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with GPU_LOCK:
            out = self._forward(inputs)
        return out.cpu().numpy()

    @staticmethod
    def _resolve_attention_kind(model_id: str) -> str | None:
        """Welche Attention für die Heatmap eines Bilds extrahiert wird, hängt von
        der Pooling-Architektur ab: DINOv2/CLIP poolen über das CLS-Token (dessen
        Attention auf die Patches in der letzten Schicht zeigt, was eingeflossen
        ist); SigLIP/SigLIP2 haben kein CLS-Token, sondern poolen über einen
        gelernten "Probe"-Query in einem eigenen Attention-Pooling-Head."""
        if model_id.startswith("facebook/dinov2") or model_id.startswith("openai/clip"):
            return "cls"
        if "siglip" in model_id:
            return "probe"
        return None

    @torch.no_grad()
    def attention_map(self, image) -> np.ndarray | None:
        """Attention-Heatmap-Grundlage für EIN Bild: (grid_h, grid_w)-Array,
        0..1-normiert, wie stark jeder Bild-Patch in die gepoolte Embedding
        eingeflossen ist (Grundlage für die kNN-Klassifikation). None, wenn die
        Architektur keine geeignete Attention exponiert oder das Grid nicht
        quadratisch ist."""
        if self._attn_kind is None:
            return None

        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        with GPU_LOCK:
            weights = (
                self._probe_attention(inputs) if self._attn_kind == "probe" else self._cls_attention(inputs)
            )
        if weights is None:
            return None

        side = round(len(weights) ** 0.5)
        if side * side != len(weights) or side < 3:
            return None
        return _normalize_attention_grid(weights.reshape(side, side))

    def _probe_attention(self, inputs) -> np.ndarray | None:
        """SigLIP/SigLIP2: die Attention-Gewichte des Pooling-Heads (Probe-Query
        -> Patches) sind nicht Teil des regulären Outputs -> per Forward-Hook auf
        dem internen nn.MultiheadAttention abgreifen, statt den Head umzubauen."""
        head = getattr(self._model, "head", None)
        if head is None or not hasattr(head, "attention"):
            return None

        captured: dict = {}

        def _hook(_module, _inputs, output):
            captured["weights"] = output[1]

        handle = head.attention.register_forward_hook(_hook)
        try:
            self._model(**inputs)
        finally:
            handle.remove()

        weights = captured.get("weights")
        if weights is None:
            return None
        return weights[0, 0].float().cpu().numpy()

    def _cls_attention(self, inputs) -> np.ndarray | None:
        """DINOv2/CLIP: Self-Attention der letzten Schicht, CLS-Zeile (ohne die
        CLS-Spalte selbst) über alle Heads gemittelt — Standard-Technik zur
        ViT-Attention-Visualisierung."""
        out = self._model(**inputs, output_attentions=True)
        attentions = getattr(out, "attentions", None)
        if not attentions:
            return None
        last_layer = attentions[-1][0]  # (heads, seq, seq)
        cls_to_patches = last_layer.mean(dim=0)[0, 1:]
        return cls_to_patches.float().cpu().numpy()
