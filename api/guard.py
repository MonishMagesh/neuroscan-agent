from __future__ import annotations
import json, time
from google import genai
from google.genai import types
try:
    from agent.planner import GEMINI_MODELS as MODELS
except ImportError:
    MODELS = ["gemini-3.6-flash","gemini-3.5-flash","gemini-3.5-flash-lite","gemini-3.1-flash-lite","gemini-flash-lite-latest"]
PROMPT = 'Is this image a brain MRI scan? Reply with ONLY JSON: {"is_brain_mri": true or false, "reason": "<one short sentence>"}'

def check_brain_mri(image_bytes, mime_type, api_key):
    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(temperature=0)
    part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    last = None
    for model in MODELS:
        try:
            resp = client.models.generate_content(model=model, contents=[part, PROMPT], config=cfg)
            data = json.loads(resp.text)
            return {"is_brain_mri": bool(data.get("is_brain_mri")), "reason": str(data.get("reason",""))[:200]}
        except (ValueError, TypeError) as exc:
            last = exc
        except Exception as exc:
            if any(t in str(exc) for t in ("503","429","404","UNAVAILABLE","NOT_FOUND")):
                last = exc; time.sleep(1); continue
            raise
    raise RuntimeError(f"All models unavailable: {last}")
