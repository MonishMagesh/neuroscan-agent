# 🧠 NeuroScan Agent

> AI-powered brain tumor classification agent — built on IEEE SCEECS 2026 research. **100% FREE** using Google Gemini + OpenRouter.

[![IEEE Published](https://img.shields.io/badge/IEEE-SCEECS%202026-blue)](https://doi.org/10.1109/SCEECS68810.2026.11429884)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![Gemini](https://img.shields.io/badge/powered%20by-Gemini%202.0%20Flash-orange)](https://aistudio.google.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Classifies brain MRI scans into **Glioma, Meningioma, No Tumor, Pituitary** using an
AI agent that reasons through the scan, explains its findings, and generates a structured
diagnostic report. Both API keys are **completely free**.

---

## 🔗 Related Projects

| Project | Description |
|---|---|
| **NeuroScan Agent** ← you are here | Python agent + REST API |
| [NeuroScan AI](https://monishmagesh.github.io/neuroscan-ai) | Interactive web demo |
| [IEEE Paper](https://doi.org/10.1109/SCEECS68810.2026.11429884) | Published research |

---

## 🚀 Quick Start (Windows)

```powershell
# 1. Extract and enter the folder
tar -xzf neuroscan-agent.tar.gz
cd neuroscan-agent

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install google-genai

# 4. Configure API keys
copy .env.example .env
notepad .env
# Add:
#   GEMINI_API_KEY=AIza...     (free at https://aistudio.google.com/apikey)
#   OPENROUTER_API_KEY=sk-or-v1-...  (free at https://openrouter.ai/keys)

# 5. Confirm keys are set (safe — never shows actual keys)
python -c "import os; from dotenv import load_dotenv; load_dotenv(); gk=os.getenv('GEMINI_API_KEY',''); ok=os.getenv('OPENROUTER_API_KEY',''); print('Gemini:', 'SET' if gk.startswith('AIza') else 'MISSING'); print('OpenRouter:', 'SET' if ok.startswith('sk-or') else 'MISSING')"

# 6. Run your first analysis
python cli.py analyze path\to\brain_mri.jpg
```

---

## 🖥️ CLI Commands

```powershell
# Analyse an MRI
python cli.py analyze brain.jpg

# With patient context
python cli.py analyze brain.jpg --context "45F, headaches for 3 months"

# Print raw JSON
python cli.py analyze brain.jpg --json-only

# View a saved report
python cli.py report NSA-20240615-143022-A8F2B1
python cli.py report NSA-20240615-143022-A8F2B1 --md

# Review queue (flagged scans)
python cli.py queue

# List all sessions
python cli.py sessions
```

---

## 🌐 REST API

```powershell
# Start server
uvicorn api.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for interactive Swagger UI.

```powershell
# Analyse via curl
curl -X POST http://localhost:8000/analyze -F "file=@brain.jpg" -F "context=45F"
```

---

## 🤖 How the Agent Works

```
MRI image uploaded
      ↓
Gemini 2.0 Flash sees the image → calls classify_mri()
      ↓
OpenRouter vision model classifies → Glioma 84%, Meningioma 10%...
      ↓
Gemini integrates visual + model scores → calls explain_result()
      ↓
If confidence < 70% → calls flag_for_review()
      ↓
Calls generate_report() → saves JSON + Markdown
      ↓
Gemini writes plain-English summary → DONE
```

---

## 📊 Model Performance (Paper Results)

| Folds | Accuracy | AUC |
|---|---|---|
| 3-fold | 97.8% | 0.997 |
| 5-fold | 98.5% | 0.998 |
| **10-fold** | **99.0%** | **0.999** |

Dataset: 16,087 images across Glioma, Meningioma, No Tumor, Pituitary

---

## 🧪 Tests

```powershell
pytest tests/ -v
```

All 22 tests pass with no API keys needed (everything is mocked).

---

## ⚠️ Disclaimer

Research tool only. Not a medical device. Does not replace diagnosis by a licensed radiologist.

---

## 👤 Author

**Monish Magesh** — B.Tech CSE (AI/ML), SRM Institute of Science and Technology  
IEEE Published Researcher | [GitHub](https://github.com/MonishMagesh)

---

## 📄 Citation

```bibtex
@inproceedings{monish2026neuroscan,
  title     = {Deep Learning Based Brain Tumor Classification Using Orange Data Analytics},
  author    = {Monish Magesh and others},
  booktitle = {2026 IEEE Students Conference on Engineering and Systems (SCEECS)},
  year      = {2026},
  doi       = {10.1109/SCEECS68810.2026.11429884}
}
```
