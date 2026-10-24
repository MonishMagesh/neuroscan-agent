# Fix 1: Add confidence fields to prompts.py
src = open('agent/prompts.py', encoding='utf-8').read()
old = '"confidence_summary": types.Schema(type=types.Type.STRING, description="1-2 sentences on model confidence scores."),'
new = '''"confidence_summary": types.Schema(type=types.Type.STRING, description="1-2 sentences on model confidence scores."),
                    "glioma_pct": types.Schema(type=types.Type.INTEGER, description="Your confidence percent that this is Glioma. All four must sum to 100."),
                    "meningioma_pct": types.Schema(type=types.Type.INTEGER, description="Your confidence percent that this is Meningioma."),
                    "notumor_pct": types.Schema(type=types.Type.INTEGER, description="Your confidence percent that this is No Tumor."),
                    "pituitary_pct": types.Schema(type=types.Type.INTEGER, description="Your confidence percent that this is Pituitary."),'''
src = src.replace(old, new)
src = src.replace(
    '3. Call explain_result with your integrated clinical reasoning.',
    '3. Call explain_result with clinical reasoning AND percentage confidence for all four classes (must sum to 100).'
)
open('agent/prompts.py', 'w', encoding='utf-8').write(src)
print('prompts.py patched')

# Fix 2: Update _explain_result in tools.py
src = open('agent/tools.py', encoding='utf-8').read()
src = src.replace(
    'def _explain_result(self, classification, clinical_notes, confidence_summary):',
    'def _explain_result(self, classification, clinical_notes, confidence_summary, glioma_pct=0, meningioma_pct=0, notumor_pct=0, pituitary_pct=0):'
)
src = src.replace(
    'if classification != self.memory.classification:',
    '''scores = {"Glioma": int(glioma_pct), "Meningioma": int(meningioma_pct), "No Tumor": int(notumor_pct), "Pituitary": int(pituitary_pct)}
        total = sum(scores.values())
        if total > 0 and not any(v > 0 for v in self.memory.confidence.values()):
            self.memory.confidence = {k: round(v / total * 100, 1) for k, v in scores.items()}
            self.memory.top_confidence = self.memory.confidence.get(classification, 0.0)
        if classification != self.memory.classification:'''
)
open('agent/tools.py', 'w', encoding='utf-8').write(src)
print('tools.py patched')

print('All done!')