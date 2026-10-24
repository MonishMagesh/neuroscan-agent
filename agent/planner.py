
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
from .memory import AgentMemory
from .prompts import SYSTEM_PROMPT, GEMINI_TOOLS
from .tools import ToolExecutor
load_dotenv()

MAX_ITERATIONS = 12
GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
]

class NeuroScanPlanner:
    def __init__(self, gemini_api_key=None):
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.\nGet a FREE key at: https://aistudio.google.com/apikey")
        self.client = genai.Client(api_key=api_key)

    def run(self, image_path, patient_context="", session_id=None, openrouter_api_key=None):
        image_path = str(Path(image_path).resolve())
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        memory = AgentMemory(image_path=image_path, patient_context=patient_context, openrouter_api_key=openrouter_api_key or os.getenv("OPENROUTER_API_KEY", ""))
        if session_id:
            memory.session_id = session_id
        executor = ToolExecutor(memory=memory)
        img_bytes, mime_type = self._load_image(image_path)
        config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=GEMINI_TOOLS)
        contents = [types.Content(role="user", parts=[types.Part.from_bytes(data=img_bytes, mime_type=mime_type), types.Part.from_text(text=self._build_user_prompt(image_path, patient_context))])]
        import time
        for _ in range(MAX_ITERATIONS):
            response = None
            for model_name in GEMINI_MODELS:
                try:
                    response = self.client.models.generate_content(model=model_name, contents=contents, config=config)
                    break
                except Exception as e:
                    if '503' in str(e) or '429' in str(e):
                        print(f"  {model_name} busy, trying next model...")
                        time.sleep(3)
                        continue
                    else:
                        raise
            if response is None:
                raise RuntimeError("All Gemini models are currently unavailable. Try again in a few minutes.")
            if response.candidates and response.candidates[0].content:
                contents.append(response.candidates[0].content)
            fn_calls = [p.function_call for p in response.parts if p.function_call and p.function_call.name]
            if not fn_calls:
                memory.final_summary = self._get_text(response)
                break
            fn_response_parts = []
            generate_report_called = False
            for fn in fn_calls:
                tool_name = fn.name
                tool_args = dict(fn.args) if fn.args else {}
                result = executor.execute(tool_name, tool_args)
                fn_response_parts.append(types.Part(function_response=types.FunctionResponse(name=tool_name, response={"result": json.dumps(result, ensure_ascii=False)})))
                if tool_name == "generate_report":
                    generate_report_called = True
            contents.append(types.Content(role="user", parts=fn_response_parts))
            if generate_report_called:
                final = None
                for model_name in GEMINI_MODELS:
                    try:
                        final = self.client.models.generate_content(model=model_name, contents=contents, config=config)
                        break
                    except Exception:
                        continue
                if final is None:
                    final = response
                memory.final_summary = self._get_text(final)
                break
        return memory.to_dict()

    @staticmethod
    def _load_image(image_path):
        ext = Path(image_path).suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".bmp": "image/bmp"}
        mime_type = mime_map.get(ext, "image/jpeg")
        with open(image_path, "rb") as fh:
            return fh.read(), mime_type

    @staticmethod
    def _get_text(response):
        try:
            return response.text or ""
        except Exception:
            return ""

    @staticmethod
    def _build_user_prompt(image_path, context):
        filename = Path(image_path).name
        lines = [f"Analyse this brain MRI scan: **{filename}**", "", "Follow workflow exactly:", "1. Examine the MRI image for visual features.", "2. Call classify_mri() - no arguments needed.", "3. Call explain_result with integrated clinical reasoning.", "4. Call flag_for_review if top_confidence < 70% or ambiguous.", "5. Call generate_report to finalise.", "6. Write 2-3 sentence summary. Then STOP."]
        if context:
            lines.insert(1, f"\n**Patient context:** {context}\n")
        return "\n".join(lines)
