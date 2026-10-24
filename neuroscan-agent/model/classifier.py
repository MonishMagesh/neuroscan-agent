
from __future__ import annotations
import base64, json, os, re
from pathlib import Path
from typing import Optional
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image
load_dotenv()

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
RISK_MAP = {"Glioma": "High", "Meningioma": "Medium", "No Tumor": "Low", "Pituitary": "Medium"}
OPENROUTER_MODELS = [
    "google/gemma-3n-e4b-it:free",
    "meta-llama/llama-4-scout:free",
    "meta-llama/llama-4-maverick:free",
    "microsoft/phi-4-reasoning:free",
]
CLASSIFICATION_PROMPT = """You are a neuroradiology AI analyzing a brain MRI scan.
Classify into ONE category:
GLIOMA: Irregular infiltrative mass, heterogeneous signal, edema, indistinct margins.
MENINGIOMA: Well-defined extraaxial mass, dural attachment, dural tail sign.
PITUITARY: Mass in/near sella turcica, suprasellar extension.
NO TUMOR: Normal brain, no mass lesion, symmetric.
Return ONLY this JSON (no markdown):
{"Glioma": <int 0-100>, "Meningioma": <int 0-100>, "No Tumor": <int 0-100>, "Pituitary": <int 0-100>}
All four must sum to 100."""

class MRIClassifier:
    def __init__(self, model_path=None, openrouter_api_key=None, mode=None):
        self.model_path = model_path or os.getenv("MODEL_PATH", "model/weights/neuroscan_model.keras")
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.mode = mode or os.getenv("ANALYSIS_MODE", "api")
        self._keras_model = None
        if self.mode == "local":
            self._load_local_model()

    def _load_local_model(self):
        try:
            import tensorflow as tf
            mf = Path(self.model_path)
            if not mf.exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}. Run model/train.py first.")
            self._keras_model = tf.keras.models.load_model(str(mf))
        except ImportError:
            self.mode = "api"
        except Exception:
            self.mode = "api"

    @staticmethod
    def _preprocess_image(image_path):
        img = Image.open(image_path).convert("RGB").resize((224, 224), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32)
        arr = (arr / 127.5) - 1.0
        return np.expand_dims(arr, axis=0)

    @staticmethod
    def _encode_image(image_path):
        ext = Path(image_path).suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".bmp": "image/bmp"}
        media_type = mime_map.get(ext, "image/jpeg")
        with open(image_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("utf-8")
        return b64, media_type

    def _classify_local(self, image_path):
        arr = self._preprocess_image(image_path)
        preds = self._keras_model.predict(arr, verbose=0)[0]
        return {cls: float(p) for cls, p in zip(CLASSES, preds)}

    def _classify_api(self, image_path):
        if not self.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to .env (free at https://openrouter.ai/keys).")
        b64, media_type = self._encode_image(image_path)
        data_url = f"data:{media_type};base64,{b64}"
        headers = {"Authorization": f"Bearer {self.openrouter_api_key}", "Content-Type": "application/json", "HTTP-Referer": "https://github.com/MonishMagesh/neuroscan-agent", "X-Title": "NeuroScan Agent"}
        payload_base = {"max_tokens": 200, "temperature": 0.05, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}, {"type": "text", "text": CLASSIFICATION_PROMPT}]}]}
        last_error = "All models unavailable."
        for model_id in OPENROUTER_MODELS:
            try:
                resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json={**payload_base, "model": model_id}, timeout=60)
                if resp.status_code != 200:
                    last_error = f"{model_id} HTTP {resp.status_code}"
                    continue
                content = resp.json()["choices"][0]["message"]["content"].strip()
                match = re.search(r"\{[^}]+\}", content, re.DOTALL)
                if not match:
                    last_error = f"{model_id}: no JSON"
                    continue
                raw = json.loads(match.group())
                if not all(k in raw for k in CLASSES):
                    last_error = f"{model_id}: missing keys"
                    continue
                total = sum(float(raw[k]) for k in CLASSES)
                if total == 0:
                    continue
                return {cls: float(raw[cls]) / total for cls in CLASSES}
            except json.JSONDecodeError as exc:
                last_error = str(exc)
            except Exception as exc:
                last_error = str(exc)
        raise RuntimeError(f"Classification API failed. Last error: {last_error}")

    def classify(self, image_path):
        image_path = str(Path(image_path).resolve())
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if self.mode == "local" and self._keras_model is not None:
            confidence_raw = self._classify_local(image_path)
        else:
            confidence_raw = self._classify_api(image_path)
        classification = max(confidence_raw, key=confidence_raw.get)
        top_conf = confidence_raw[classification]
        return {
            "classification": classification,
            "confidence": {k: round(v * 100, 1) for k, v in confidence_raw.items()},
            "risk": RISK_MAP.get(classification, "Unknown"),
            "top_confidence": round(top_conf * 100, 1),
            "mode": self.mode,
        }
