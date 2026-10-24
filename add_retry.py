src = open('agent/planner.py', encoding='utf-8').read()

old = '        for _ in range(MAX_ITERATIONS):'
new = '''        import time
        for _ in range(MAX_ITERATIONS):'''

old2 = '            response = self.client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)'
new2 = '''            for attempt in range(5):
                try:
                    response = self.client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
                    break
                except Exception as e:
                    if '503' in str(e) and attempt < 4:
                        print(f"  Server busy, retrying in 20s... ({attempt+1}/5)")
                        time.sleep(20)
                    else:
                        raise'''

src = src.replace(old, new)
src = src.replace(old2, new2)
open('agent/planner.py', 'w', encoding='utf-8').write(src)
print('Retry logic added')