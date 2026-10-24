
from __future__ import annotations
import json, os, sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

@pytest.fixture()
def dummy_mri(tmp_path):
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    path = str(tmp_path / "dummy_mri.png")
    img.save(path)
    return path

@pytest.fixture()
def mock_openrouter_response():
    return {"choices": [{"message": {"content": '{"Glioma": 84, "Meningioma": 10, "No Tumor": 4, "Pituitary": 2}'}}]}

class TestAgentMemory:
    def test_session_id_format(self):
        from agent.memory import AgentMemory
        assert AgentMemory().session_id.startswith("NSA-")

    def test_default_state(self):
        from agent.memory import AgentMemory
        m = AgentMemory(image_path="/tmp/test.jpg")
        assert m.classification is None and m.flagged is False and m.tool_calls == []

    def test_record_tool_call(self):
        from agent.memory import AgentMemory
        m = AgentMemory()
        m.record_tool_call("classify_mri", {}, {"status": "ok"})
        assert len(m.tool_calls) == 1 and m.tool_calls[0]["tool"] == "classify_mri"

    def test_to_dict_structure(self):
        from agent.memory import AgentMemory
        d = AgentMemory(image_path="/tmp/scan.jpg").to_dict()
        assert "session_id" in d and "result" in d and "review" in d and "reports" in d and "disclaimer" in d

    def test_is_complete_false(self):
        from agent.memory import AgentMemory
        assert AgentMemory().is_complete is False

    def test_is_complete_true(self):
        from agent.memory import AgentMemory
        m = AgentMemory()
        m.report_json_path = "/tmp/report.json"
        assert m.is_complete is True

class TestMRIClassifier:
    def test_classify_api_success(self, dummy_mri, mock_openrouter_response):
        from model.classifier import MRIClassifier
        with patch("requests.post") as mock_post:
            mr = MagicMock()
            mr.status_code = 200
            mr.json.return_value = mock_openrouter_response
            mock_post.return_value = mr
            result = MRIClassifier(openrouter_api_key="fake-key", mode="api").classify(dummy_mri)
        assert result["classification"] == "Glioma"
        assert result["top_confidence"] == pytest.approx(84.0, abs=1.0)
        assert result["risk"] == "High"
        assert set(result["confidence"].keys()) == {"Glioma", "Meningioma", "No Tumor", "Pituitary"}

    def test_normalises_to_100(self, dummy_mri, mock_openrouter_response):
        from model.classifier import MRIClassifier
        with patch("requests.post") as mock_post:
            mr = MagicMock()
            mr.status_code = 200
            mr.json.return_value = mock_openrouter_response
            mock_post.return_value = mr
            result = MRIClassifier(openrouter_api_key="fake-key", mode="api").classify(dummy_mri)
        assert sum(result["confidence"].values()) == pytest.approx(100.0, abs=1.0)

    def test_missing_image_raises(self):
        from model.classifier import MRIClassifier
        with pytest.raises(FileNotFoundError):
            MRIClassifier(openrouter_api_key="fake-key", mode="api").classify("/no/such/file.jpg")

    def test_no_api_key_raises(self, dummy_mri, monkeypatch):
        from model.classifier import MRIClassifier
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            MRIClassifier(openrouter_api_key=None, mode="api").classify(dummy_mri)

    def test_all_models_fail_raises(self, dummy_mri):
        from model.classifier import MRIClassifier
        with patch("requests.post") as mock_post:
            mr = MagicMock()
            mr.status_code = 500
            mock_post.return_value = mr
            with pytest.raises(RuntimeError, match="Classification API failed"):
                MRIClassifier(openrouter_api_key="fake-key", mode="api").classify(dummy_mri)

    def test_risk_mapping(self, dummy_mri):
        from model.classifier import MRIClassifier, RISK_MAP
        for cls, expected_risk in RISK_MAP.items():
            conf = {c: 5 for c in ["Glioma", "Meningioma", "No Tumor", "Pituitary"]}
            conf[cls] = 85
            mock_data = {"choices": [{"message": {"content": json.dumps(conf)}}]}
            with patch("requests.post") as mock_post:
                mr = MagicMock()
                mr.status_code = 200
                mr.json.return_value = mock_data
                mock_post.return_value = mr
                result = MRIClassifier(openrouter_api_key="fake-key", mode="api").classify(dummy_mri)
            assert result["risk"] == expected_risk

class TestToolExecutor:
    def _make_executor(self, dummy_mri, tmp_path):
        from agent.memory import AgentMemory
        from agent.tools import ToolExecutor
        memory = AgentMemory(image_path=dummy_mri, openrouter_api_key="fake-key")
        os.environ["OUTPUT_DIR"] = str(tmp_path)
        return ToolExecutor(memory=memory), memory

    def test_classify_mri_writes_to_memory(self, dummy_mri, tmp_path):
        executor, memory = self._make_executor(dummy_mri, tmp_path)
        mock_result = {"classification": "Meningioma", "confidence": {"Glioma": 5.0, "Meningioma": 82.0, "No Tumor": 8.0, "Pituitary": 5.0}, "top_confidence": 82.0, "risk": "Medium", "mode": "api"}
        with patch("model.classifier.MRIClassifier.classify", return_value=mock_result):
            result = executor.execute("classify_mri", {})
        assert result["status"] == "ok"
        assert memory.classification == "Meningioma" and memory.top_confidence == 82.0 and memory.risk == "Medium"
        assert len(memory.tool_calls) == 1

    def test_explain_result_stores_notes(self, dummy_mri, tmp_path):
        executor, memory = self._make_executor(dummy_mri, tmp_path)
        memory.classification = "Glioma"
        result = executor.execute("explain_result", {"classification": "Glioma", "clinical_notes": "Irregular mass with edema.", "confidence_summary": "84% confidence."})
        assert result["status"] == "ok" and memory.clinical_notes == "Irregular mass with edema."

    def test_flag_for_review(self, dummy_mri, tmp_path):
        executor, memory = self._make_executor(dummy_mri, tmp_path)
        result = executor.execute("flag_for_review", {"reason": "Low confidence 55%.", "severity": "medium"})
        assert result["status"] == "ok" and memory.flagged is True and memory.flag_severity == "medium"
        assert (Path(tmp_path) / "review_queue.jsonl").exists()

    def test_generate_report_creates_files(self, dummy_mri, tmp_path):
        executor, memory = self._make_executor(dummy_mri, tmp_path)
        memory.classification = "Pituitary"
        memory.confidence = {"Glioma": 5.0, "Meningioma": 5.0, "No Tumor": 10.0, "Pituitary": 80.0}
        memory.top_confidence = 80.0
        memory.risk = "Medium"
        memory.clinical_notes = "Sellar mass noted."
        memory.confidence_summary = "80% confidence."
        result = executor.execute("generate_report", {"final_classification": "Pituitary", "risk_level": "Medium", "recommendation": "Endocrinology referral."})
        assert result["status"] == "ok"
        assert Path(result["report_json"]).exists() and Path(result["report_md"]).exists()

    def test_unknown_tool_returns_error(self, dummy_mri, tmp_path):
        executor, _ = self._make_executor(dummy_mri, tmp_path)
        assert "error" in executor.execute("nonexistent_tool", {})

class TestNeuroScanPlanner:
    def test_missing_api_key_raises(self):
        from agent.planner import NeuroScanPlanner
        original = os.environ.pop("GEMINI_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                NeuroScanPlanner(gemini_api_key=None)
        finally:
            if original:
                os.environ["GEMINI_API_KEY"] = original

    def test_missing_image_raises(self):
        from agent.planner import NeuroScanPlanner
        planner = NeuroScanPlanner(gemini_api_key="fake-key")
        with pytest.raises(FileNotFoundError):
            planner.run("/nonexistent/scan.jpg")

    def test_load_image(self, dummy_mri):
        from agent.planner import NeuroScanPlanner
        img_bytes, mime_type = NeuroScanPlanner._load_image(dummy_mri)
        assert isinstance(img_bytes, bytes) and len(img_bytes) > 100 and mime_type == "image/png"

    def test_build_user_prompt(self):
        from agent.planner import NeuroScanPlanner
        prompt = NeuroScanPlanner._build_user_prompt("/tmp/scan.jpg", "45F headaches")
        assert "scan.jpg" in prompt and "45F headaches" in prompt and "classify_mri" in prompt

    def test_full_run_mocked(self, dummy_mri, tmp_path):
        from agent.planner import NeuroScanPlanner
        os.environ["OUTPUT_DIR"] = str(tmp_path)
        mock_classify_result = {"classification": "Glioma", "confidence": {"Glioma": 84.0, "Meningioma": 10.0, "No Tumor": 4.0, "Pituitary": 2.0}, "top_confidence": 84.0, "risk": "High", "mode": "api"}

        def make_fn_resp(name, args=None):
            fn = MagicMock(); fn.name = name; fn.args = args or {}
            part = MagicMock(); part.function_call = fn
            resp = MagicMock(); resp.parts = [part]; resp.text = ""
            cand = MagicMock(); cand.content = MagicMock(); resp.candidates = [cand]
            return resp

        def make_text_resp(text):
            part = MagicMock(); part.function_call = MagicMock(); part.function_call.name = None
            resp = MagicMock(); resp.parts = [part]; resp.text = text
            cand = MagicMock(); cand.content = MagicMock(); resp.candidates = [cand]
            return resp

        responses = [
            make_fn_resp("classify_mri", {}),
            make_fn_resp("explain_result", {"classification": "Glioma", "clinical_notes": "Irregular mass.", "confidence_summary": "84%."}),
            make_fn_resp("generate_report", {"final_classification": "Glioma", "risk_level": "High", "recommendation": "Urgent consultation."}),
            make_text_resp("Analysis complete. Glioma identified."),
        ]
        call_count = 0
        def mock_generate(**kwargs):
            nonlocal call_count
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return resp

        with patch("model.classifier.MRIClassifier.classify", return_value=mock_classify_result):
            with patch("google.genai.Client") as mock_client_class:
                mc = MagicMock()
                mc.models.generate_content.side_effect = mock_generate
                mock_client_class.return_value = mc
                planner = NeuroScanPlanner(gemini_api_key="fake-key")
                result = planner.run(image_path=dummy_mri, patient_context="Test patient")

        assert "session_id" in result
        assert result["result"]["classification"] == "Glioma"
        assert result["result"]["risk"] == "High"
        assert result["agent_trace"]["final_summary"] != ""

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
