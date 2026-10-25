from __future__ import annotations
import asyncio, hmac, io, os, shutil, tempfile, time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool
os.environ.setdefault("OUTPUT_DIR", "/tmp/neuroscan-outputs")
from agent.planner import NeuroScanPlanner
from api.guard import check_brain_mri
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
MAX_BYTES = 10 * 1024 * 1024
MIME = {".jpg":"image/jpeg",".jpeg":"image/jpeg",".png":"image/png",".webp":"image/webp",".bmp":"image/bmp"}
ACCESS_CODE = os.getenv("ACCESS_CODE","")
SERVER_KEY = os.getenv("GEMINI_API_KEY","")
LIMITS = ((600,6),(86400,40))
_hits: dict[str,deque] = defaultdict(deque)
_slots = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT","1")))

app = FastAPI(title="NeuroScan Agent", docs_url=None, redoc_url=None, openapi_url=None)
_origins = [o.strip() for o in os.getenv("CORS_ORIGINS","").split(",") if o.strip()]
if _origins:
    app.add_middleware(CORSMiddleware,allow_origins=_origins,allow_methods=["POST","GET"],
                       allow_headers=["X-Gemini-Key","X-Access-Code","Content-Type"])

def _client_ip(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "unknown")

def _check_rate(ip: str) -> None:
    now,q = time.time(),_hits[ip]
    while q and now-q[0]>86400: q.popleft()
    for window,cap in LIMITS:
        if sum(1 for t in q if now-t<window)>=cap:
            raise HTTPException(429,"Rate limit reached. Please try again later.")
    q.append(now)

def _pick_key(gemini_key,access_code):
    if gemini_key and gemini_key.strip(): return gemini_key.strip()
    if ACCESS_CODE and SERVER_KEY and access_code and hmac.compare_digest(access_code.encode(),ACCESS_CODE.encode()):
        return SERVER_KEY
    raise HTTPException(401,"Enter your Gemini API key (free at aistudio.google.com/apikey) or a valid access code.")

def _fail(exc):
    s=str(exc); low=s.lower()
    if "api key" in low and ("not valid" in low or "invalid" in low): return HTTPException(401,"Gemini rejected that API key.")
    if "429" in s or "resource_exhausted" in low: return HTTPException(429,"Gemini quota exhausted. Try again later.")
    if "503" in s or "unavailable" in low: return HTTPException(503,"Gemini is busy. Please retry in a minute.")
    return HTTPException(500,"Analysis failed. Please try again.")

@app.get("/health")
async def health(): return {"status":"ok"}

@app.get("/")
async def index(): return FileResponse(WEB_DIR/"index.html")

@app.post("/analyze")
async def analyze(request: Request, file: UploadFile=File(...), context: str=Form(""),
                  x_gemini_key: Optional[str]=Header(None), x_access_code: Optional[str]=Header(None)):
    key = _pick_key(x_gemini_key, x_access_code)
    _check_rate(_client_ip(request))
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in MIME: raise HTTPException(415,"Upload a JPG, PNG, WEBP or BMP image.")
    data = await file.read(MAX_BYTES+1)
    if len(data)>MAX_BYTES: raise HTTPException(413,"Image is larger than 10 MB.")
    try: Image.open(io.BytesIO(data)).verify()
    except Exception: raise HTTPException(415,"That file is not a valid image.")
    tmp_path=None; result={}
    async with _slots:
        try:
            verdict = await run_in_threadpool(check_brain_mri,data,MIME[suffix],key)
            if not verdict["is_brain_mri"]:
                return JSONResponse(status_code=422,content={"error":"not_brain_mri","detail":verdict["reason"]})
            with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as tmp:
                tmp.write(data); tmp_path=tmp.name
            planner = NeuroScanPlanner(gemini_api_key=key)
            result = await run_in_threadpool(planner.run,image_path=tmp_path,patient_context=context[:500])
        except HTTPException: raise
        except Exception as exc:
            print(f"[analyze] {type(exc).__name__}: {str(exc)[:300]}")
            raise _fail(exc) from exc
        finally:
            if tmp_path: Path(tmp_path).unlink(missing_ok=True)
    sid=result.get("session_id","")
    if sid: shutil.rmtree(Path(os.environ["OUTPUT_DIR"])/sid,ignore_errors=True)
    result.get("input",{}).pop("image_path",None); result["reports"]={}
    return JSONResponse(result)
