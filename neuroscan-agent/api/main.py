
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
from typing import Optional
import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from agent.planner import NeuroScanPlanner

app = FastAPI(title="NeuroScan Agent API", description="Brain tumor classification agent. Research only.", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
_planner: Optional[NeuroScanPlanner] = None

def _output_dir():
    return Path(os.getenv("OUTPUT_DIR", "outputs"))

def get_planner():
    global _planner
    if _planner is None:
        try:
            _planner = NeuroScanPlanner()
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=f"Agent not configured: {exc}. Set GEMINI_API_KEY in .env (free at https://aistudio.google.com/apikey)") from exc
    return _planner

@app.get("/", tags=["health"])
async def root():
    return {"service": "NeuroScan Agent API", "version": "1.0.0", "status": "running", "disclaimer": "Research only."}

@app.post("/analyze", tags=["analysis"])
async def analyze_mri(file: UploadFile = File(...), context: str = Form(default=""), session_id: str = Form(default=""), openrouter_api_key: str = Form(default="")):
    suffix = Path(file.filename or "scan.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())
        planner = get_planner()
        result = planner.run(image_path=tmp_path, patient_context=context, session_id=session_id or None, openrouter_api_key=openrouter_api_key or None)
        return JSONResponse(content=result, status_code=200)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink(missing_ok=True)

@app.get("/report/{session_id}", tags=["reports"])
async def get_report_json(session_id: str):
    rp = _output_dir() / session_id / "report.json"
    if not rp.exists():
        raise HTTPException(status_code=404, detail=f"No report for '{session_id}'.")
    async with aiofiles.open(rp, "r", encoding="utf-8") as fh:
        return JSONResponse(content=json.loads(await fh.read()))

@app.get("/report/{session_id}/md", tags=["reports"], response_class=PlainTextResponse)
async def get_report_markdown(session_id: str):
    rp = _output_dir() / session_id / "report.md"
    if not rp.exists():
        raise HTTPException(status_code=404, detail=f"No Markdown report for '{session_id}'.")
    async with aiofiles.open(rp, "r", encoding="utf-8") as fh:
        return PlainTextResponse(content=await fh.read())

@app.get("/queue", tags=["review"])
async def get_review_queue():
    qp = _output_dir() / "review_queue.jsonl"
    if not qp.exists():
        return JSONResponse(content={"flagged": [], "total": 0})
    entries = []
    async with aiofiles.open(qp, "r", encoding="utf-8") as fh:
        async for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    sev = {"high": 0, "medium": 1, "low": 2}
    entries.sort(key=lambda x: (sev.get(x.get("severity", "low"), 3), x.get("flagged_at", "")))
    return JSONResponse(content={"flagged": entries, "total": len(entries)})

@app.get("/sessions", tags=["sessions"])
async def list_sessions():
    od = _output_dir()
    if not od.exists():
        return JSONResponse(content={"sessions": [], "total": 0})
    sessions = []
    for sd in sorted(od.iterdir(), reverse=True):
        rp = sd / "report.json"
        if rp.exists():
            try:
                data = json.loads(rp.read_text(encoding="utf-8"))
                sessions.append({"session_id": data.get("session_id"), "started_at": data.get("started_at"), "image": data.get("input", {}).get("image_filename"), "classification": data.get("result", {}).get("classification"), "risk": data.get("result", {}).get("risk"), "flagged": data.get("review", {}).get("flagged", False)})
            except Exception:
                continue
    return JSONResponse(content={"sessions": sessions, "total": len(sessions)})
