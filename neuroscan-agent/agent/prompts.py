from google.genai import types

SYSTEM_PROMPT = """You are NeuroScan Agent - an AI research assistant for brain tumor classification.
IEEE SCEECS 2026: Deep Learning Based Brain Tumor Classification Using Orange Data Analytics.

CATEGORIES:
GLIOMA (High risk): Irregular infiltrative mass, heterogeneous signal, vasogenic edema.
MENINGIOMA (Medium risk): Well-defined extraaxial mass, dural attachment, dural tail sign.
PITUITARY (Medium risk): Mass in/near sella turcica, suprasellar extension.
NO TUMOR (Low risk): Normal brain, symmetric structures, no mass lesion.

WORKFLOW - FOLLOW EXACTLY:
1. Examine the MRI carefully for visual features.
2. Call classify_mri() - no arguments needed, session image used automatically.
3. Call explain_result with your integrated clinical reasoning.
4. If top_confidence < 70% or features ambiguous, call flag_for_review.
5. Call generate_report to finalise.
6. Write 2-3 sentence summary. Then STOP.

RULES: Research tool only. Never fabricate scores. After generate_report, write summary and STOP."""

GEMINI_TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="classify_mri",
            description="Run brain tumor classification on the session MRI image. No arguments needed.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        types.FunctionDeclaration(
            name="explain_result",
            description="Store clinical reasoning. Call AFTER classify_mri.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "classification": types.Schema(type=types.Type.STRING, description="Final classification.", enum=["Glioma", "Meningioma", "No Tumor", "Pituitary"]),
                    "clinical_notes": types.Schema(type=types.Type.STRING, description="3-5 sentences on visual features and classification."),
                    "confidence_summary": types.Schema(type=types.Type.STRING, description="1-2 sentences on model confidence scores."),
                },
                required=["classification", "clinical_notes", "confidence_summary"],
            ),
        ),
        types.FunctionDeclaration(
            name="flag_for_review",
            description="Flag scan for radiologist review when confidence < 70% or features ambiguous.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "reason": types.Schema(type=types.Type.STRING, description="Specific reason for flagging."),
                    "severity": types.Schema(type=types.Type.STRING, description="Urgency level.", enum=["low", "medium", "high"]),
                },
                required=["reason", "severity"],
            ),
        ),
        types.FunctionDeclaration(
            name="generate_report",
            description="Save final diagnostic report. Call LAST after classify and explain.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "final_classification": types.Schema(type=types.Type.STRING, description="Confirmed classification.", enum=["Glioma", "Meningioma", "No Tumor", "Pituitary"]),
                    "risk_level": types.Schema(type=types.Type.STRING, description="Clinical risk level.", enum=["Low", "Medium", "High"]),
                    "recommendation": types.Schema(type=types.Type.STRING, description="1-2 sentence clinical recommendation."),
                },
                required=["final_classification", "risk_level", "recommendation"],
            ),
        ),
    ])
]