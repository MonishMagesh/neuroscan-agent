from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

def _new_session_id() -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"NSA-{ts}-{str(uuid.uuid4())[:6].upper()}"

@dataclass
class AgentMemory:
    image_path: str = ""
    patient_context: str = ""
    openrouter_api_key: str = ""
    session_id: str = field(default_factory=_new_session_id)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    classification: Optional[str] = None
    confidence: dict = field(default_factory=dict)
    top_confidence: Optional[float] = None
    risk: Optional[str] = None
    classifier_mode: Optional[str] = None
    clinical_notes: Optional[str] = None
    confidence_summary: Optional[str] = None
    flagged: bool = False
    flag_reason: Optional[str] = None
    flag_severity: Optional[str] = None
    report_json_path: Optional[str] = None
    report_md_path: Optional[str] = None
    recommendation: Optional[str] = None
    tool_calls: list = field(default_factory=list)
    final_summary: Optional[str] = None
    completed_at: Optional[str] = None

    def record_tool_call(self, tool_name: str, tool_input: dict, result: dict) -> None:
        self.tool_calls.append({"tool": tool_name, "input": tool_input, "result": result, "timestamp": datetime.now().isoformat()})

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at or datetime.now().isoformat(),
            "input": {"image_path": self.image_path, "image_filename": Path(self.image_path).name if self.image_path else None, "patient_context": self.patient_context},
            "result": {"classification": self.classification, "confidence": self.confidence, "top_confidence": self.top_confidence, "risk": self.risk, "clinical_notes": self.clinical_notes, "confidence_summary": self.confidence_summary, "recommendation": self.recommendation, "classifier_mode": self.classifier_mode},
            "review": {"flagged": self.flagged, "flag_reason": self.flag_reason, "flag_severity": self.flag_severity},
            "reports": {"json": self.report_json_path, "markdown": self.report_md_path},
            "agent_trace": {"tool_calls": self.tool_calls, "final_summary": self.final_summary},
            "disclaimer": "This analysis is for research purposes only. Not medical advice.",
        }

    @property
    def is_complete(self) -> bool:
        return self.report_json_path is not None
