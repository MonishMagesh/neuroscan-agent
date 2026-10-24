
from __future__ import annotations
import json, os, sys
from datetime import datetime
from pathlib import Path
from .memory import AgentMemory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.classifier import MRIClassifier

RISK_EMOJI = {"High": "🔴", "Medium": "🟠", "Low": "🟢", "Unknown": "⚪"}

def _conf_bar(pct, width=30):
    filled = int(round(pct / 100 * width))
    return "█" * filled + "░" * (width - filled)

def _build_markdown_report(memory):
    conf = memory.confidence or {}
    emoji = RISK_EMOJI.get(memory.risk or "Unknown", "⚪")
    lines = [
        "# NeuroScan AI — Diagnostic Report", "",
        f"> **Session ID:** `{memory.session_id}`  ",
        f"> **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> **Image:** `{Path(memory.image_path).name if memory.image_path else 'unknown'}`",
        "", "---", "", "## Result", "",
        "| Field | Value |", "|---|---|",
        f"| **Classification** | {emoji} **{memory.classification or 'Unknown'}** |",
        f"| **Risk Level** | {memory.risk or 'Unknown'} |",
        f"| **Top Confidence** | {(memory.top_confidence or 0.0):.1f}% |",
        f"| **Review Flagged** | {'⚠ YES — ' + (memory.flag_severity or '').upper() if memory.flagged else 'No'} |",
        "", "---", "", "## Confidence Scores", "",
    ]
    for cls, pct in sorted(conf.items(), key=lambda x: -x[1]):
        lines.append(f"**{cls:<14}** `{_conf_bar(pct)}` {pct:>5.1f}%")
    lines += ["", "---", "", "## Clinical Reasoning", "", memory.clinical_notes or "_Not recorded._", ""]
    if memory.flagged:
        lines += [f"## ⚠ Review Flag — {(memory.flag_severity or '').upper()} Priority", "", f"**Reason:** {memory.flag_reason}", "", "---", ""]
    tool_trace = " → ".join(tc["tool"] for tc in memory.tool_calls) or "None recorded"
    lines += ["## Recommendation", "", memory.recommendation or "_Not recorded._", "", "---", "", f"Tools called: {tool_trace}", "", "> ⚠ Research tool only. Not medical advice.", ""]
    return "\n".join(lines)

class ToolExecutor:
    def __init__(self, memory):
        self.memory = memory
        self._classifier = None

    def _get_classifier(self):
        if self._classifier is None:
            self._classifier = MRIClassifier(openrouter_api_key=self.memory.openrouter_api_key or None)
        return self._classifier

    def execute(self, tool_name, tool_input):
        try:
            if tool_name == "classify_mri":
                result = self._classify_mri(**tool_input)
            elif tool_name == "explain_result":
                result = self._explain_result(**tool_input)
            elif tool_name == "flag_for_review":
                result = self._flag_for_review(**tool_input)
            elif tool_name == "generate_report":
                result = self._generate_report(**tool_input)
            else:
                result = {"error": f"Unknown tool: '{tool_name}'"}
        except TypeError as exc:
            result = {"error": f"Bad arguments for '{tool_name}': {exc}"}
        except Exception as exc:
            result = {"error": f"Tool '{tool_name}' failed: {exc}"}
        self.memory.record_tool_call(tool_name, tool_input, result)
        return result

    def _classify_mri(self, image_path=""):
        actual_path = self.memory.image_path or image_path
        if not actual_path:
            return {"error": "No image path in session memory."}
        classifier = self._get_classifier()
        result = classifier.classify(actual_path)
        self.memory.classification = result["classification"]
        self.memory.confidence = result["confidence"]
        self.memory.top_confidence = result["top_confidence"]
        self.memory.risk = result["risk"]
        self.memory.classifier_mode = result["mode"]
        return {
            "status": "ok",
            "classification": result["classification"],
            "confidence": result["confidence"],
            "top_confidence": result["top_confidence"],
            "risk": result["risk"],
            "mode": result["mode"],
            "note": f"Top confidence {result['top_confidence']:.1f}%. {'Consider flagging.' if result['top_confidence'] < 70 else 'Confidence acceptable.'}",
        }

    def _explain_result(self, classification, clinical_notes, confidence_summary):
        self.memory.clinical_notes = clinical_notes
        self.memory.confidence_summary = confidence_summary
        if classification != self.memory.classification:
            self.memory.classification = classification
        return {"status": "ok", "stored": True, "classification_confirmed": classification}

    def _flag_for_review(self, reason, severity):
        self.memory.flagged = True
        self.memory.flag_reason = reason
        self.memory.flag_severity = severity
        output_dir = Path(os.getenv("OUTPUT_DIR", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        queue_path = output_dir / "review_queue.jsonl"
        entry = {"session_id": self.memory.session_id, "image": Path(self.memory.image_path).name if self.memory.image_path else "", "classification": self.memory.classification, "top_confidence": self.memory.top_confidence, "reason": reason, "severity": severity, "flagged_at": datetime.now().isoformat()}
        with open(queue_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        return {"status": "ok", "flagged": True, "severity": severity, "message": f"Queued for {severity}-priority review."}

    def _generate_report(self, final_classification, risk_level, recommendation):
        m = self.memory
        m.recommendation = recommendation
        m.classification = final_classification
        m.risk = risk_level
        m.completed_at = datetime.now().isoformat()
        output_root = Path(os.getenv("OUTPUT_DIR", "outputs"))
        session_dir = output_root / m.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        json_path = session_dir / "report.json"
        json_path.write_text(json.dumps(m.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        m.report_json_path = str(json_path)
        md_path = session_dir / "report.md"
        md_path.write_text(_build_markdown_report(m), encoding="utf-8")
        m.report_md_path = str(md_path)
        return {"status": "ok", "session_id": m.session_id, "report_json": str(json_path), "report_md": str(md_path), "classification": final_classification, "risk": risk_level, "flagged": m.flagged, "message": "Report saved."}
