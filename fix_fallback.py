src = open('agent/planner.py', encoding='utf-8').read()

# Replace single model constant with fallback list
old = 'GEMINI_MODEL = "gemini-3.6-flash"'
new = '''GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
]'''
src = src.replace(old, new)

# Replace the retry block with model fallback
old2 = '''            for attempt in range(5):
                try:
                    response = self.client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
                    break
                except Exception as e:
                    if '503' in str(e) and attempt < 4:
                        print(f"  Server busy, retrying in 20s... ({attempt+1}/5)")
                        time.sleep(20)
                    else:
                        raise'''
new2 = '''            response = None
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
                raise RuntimeError("All Gemini models are currently unavailable. Try again in a few minutes.")'''
src = src.replace(old2, new2)

# Also fix the final summary call
old3 = 'final = self.client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)'
new3 = '''final = None
                for model_name in GEMINI_MODELS:
                    try:
                        final = self.client.models.generate_content(model=model_name, contents=contents, config=config)
                        break
                    except Exception:
                        continue
                if final is None:
                    final = response'''
src = src.replace(old3, new3)

open('agent/planner.py', 'w', encoding='utf-8').write(src)
print('Model fallback applied successfully')
print('Models in order:', ['gemini-3.6-flash','gemini-3.5-flash','gemini-3.5-flash-lite','gemini-3.1-flash-lite','gemini-flash-lite-latest','gemini-flash-latest'])